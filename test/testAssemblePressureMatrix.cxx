#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "DirectionalArray.h"
#include "fixtures/MpiTest.cxx"
#include "GhostExchangePlan.h"
#include "PhysicsBasedPreconditioner.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <mpi.h>

TEST_CASE_METHOD(MpiTest, "PhysicsBasedPreconditioner::assemblePressureMatrix")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 1 && num_ranks != 4) { SKIP("Test requires either 1 or 4 ranks"); }

    // Initialise problem
    int dt = 4.5228571428571426E-004;
    DirectionalArray<BoundaryConditions> global_bcs;
    global_bcs.fill(BoundaryConditions::PERIODIC);
    Slab global_slab({0, 0, 0}, {16, 16, 16});

    // Set domain decompositions
    std::array<int, 3> cart_dims = createCartesianDimensions();
    MPI_Comm cart_comm           = createCartesianTopology(cart_dims, global_bcs);
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);
    DirectionalArray<int> nghbrs = neighbours(cart_comm);
    GhostExchangePlan plan(global_slab, global_bcs, nghbrs, cart_dims, cart_comm);

    // Populate starting states
    // Laplacian coefficients
    std::vector<DirectionalArray<double>> laplacian_coefficients_data;
    for (int cell_idx = 0; cell_idx < global_slab.volume(); cell_idx++)
    {
        DirectionalArray<double> laplacian_coefficient;
        laplacian_coefficient.fill(2.5600000000000000E+002);
        laplacian_coefficients_data.push_back(laplacian_coefficient);
    }
    Array3D<DirectionalArray<double>> laplacian_coefficients(
        global_slab.end(0), global_slab.end(1), global_slab.end(2), laplacian_coefficients_data);
    std::vector<double> solution_data(global_slab.volume() * PhysicsBasedPreconditioner::num_sol_vars());
    std::vector<double> solution_dPd_data(global_slab.volume() * PhysicsBasedPreconditioner::num_sol_dPd_vars());
    PackedVariablesWithGhosts solution(solution_data, global_slab.end(0), global_slab.end(1), global_slab.end(2), plan);
    PackedVariables solution_dPd(solution_dPd_data, global_slab.end(0), global_slab.end(1), global_slab.end(2));

    // Fill arrays
    solution.singleVariableView(SolutionVariableIndices::RHO).fill(1.0);
    // Fill with something easier to spot as these shouldn't be used by assemblePressureMatrix.
    solution.singleVariableView(SolutionVariableIndices::PRESSURE).fill(0);
    solution.singleVariableView(SolutionVariableIndices::ENERGY).fill(0);
    solution_dPd.singleVariableView(SolutionDpdVariableIndices::DPDENERGY).fill(0);
    solution_dPd.singleVariableView(SolutionDpdVariableIndices::DPDRHO).fill(0);

    // Pressure
    Array3DWithGhosts<double> solution_rho_with_ghosts = solution.singleVariableView(SolutionVariableIndices::RHO);
    // Squared sound speed
    Array3D<double> squared_sound_speed(global_slab.end(0), global_slab.end(1), global_slab.end(2));

    PhysicsBasedPreconditioner pbp(3, 3, true, solution, solution_dPd, squared_sound_speed);

    pbp.assemblePressureMatrix(solution_rho_with_ghosts, laplacian_coefficients);

    const std::vector<double>& actual_pressure_matrix_for_solver        = pbp.pressureMatrixForSolver();
    const std::vector<long long>& actual_global_pressure_column_indices = pbp.globalPressureColumnIndices();
    const std::vector<long long>& actual_global_pressure_row_indices    = pbp.globalPressureRowIndices();
    const std::vector<long long>& actual_global_pressure_row_offsets    = pbp.globalPressureRowOffsets();

    // MUSIC matrix rows of this rank (before the D A D scaling): global row, global column, value
    std::string expected_output_file = TEST_DATA_DIR "/assemble_pressure_matrix_roberts_dynamo_" + std::to_string(num_ranks) +
                                       "rank_rank" + std::to_string(my_rank) + "_expected_outputs.dat";
    FILE* file                       = fopen(expected_output_file.c_str(), "r");
    REQUIRE(file != nullptr);
    long long row, column;
    double value;
    size_t entry = 0, local_row_index = 0;
    while (fscanf(file, "%lld %lld %lf", &row, &column, &value) == 3)
    {
        REQUIRE(entry < actual_pressure_matrix_for_solver.size());
        while (actual_global_pressure_row_offsets[local_row_index + 1] <= entry)
        {
            local_row_index++;
        }
        INFO(std::to_string(actual_global_pressure_row_indices[local_row_index]) + ", " + std::to_string(entry) + ", " +
             std::to_string(local_row_index) + ", " + std::to_string(row));
        REQUIRE(actual_global_pressure_row_indices[local_row_index] == row);
        REQUIRE(actual_global_pressure_column_indices[entry] == column);
        REQUIRE_THAT(actual_pressure_matrix_for_solver[entry], Catch::Matchers::WithinRel(value, 1e-14));
        entry++;
    }
    fclose(file);
    REQUIRE(entry == actual_pressure_matrix_for_solver.size());
}
