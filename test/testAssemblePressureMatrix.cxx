#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "DirectionalArray.h"
#include "fixtures/MpiTest.cxx"
#include "fixtures/PhysicsBasedPreconditionerTest.cxx"
#include "GhostExchangePlan.h"
#include "PhysicsBasedPreconditioner.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <mpi.h>

struct AssemblePressureMatrixTestSetup
{
    DirectionalArray<BoundaryConditions> global_bcs;
    std::array<int, 3> dimensions;
    MPI_Comm communicator;
    GhostExchangePlan plan;
    std::array<unsigned int, 3> nc;
    size_t size;
    std::vector<DirectionalArray<double>> laplacian_coefficients_data;
    Array3D<DirectionalArray<double>> laplacian_coefficients;
    std::vector<double> solution_data;
    std::vector<double> solution_dPd_data;
    PackedVariablesWithGhosts solution;
    PackedVariables solution_dPd;
    std::vector<double> squared_sound_speed_data;
    Array3D<double> squared_sound_speed;
    PhysicsBasedPreconditioner pbp;

    AssemblePressureMatrixTestSetup(const double dt,
                                    const int ndim,
                                    const int nmom,
                                    const bool mhd_enabled,
                                    const DirectionalArray<BoundaryConditions> global_bcs,
                                    const std::array<int, 3>& cell_counts,
                                    const double laplacian_coefficients_value,
                                    const double sol_rho_value,
                                    const double squared_sound_speed_value) :
        dimensions(MpiTest::createCartesianDimensions()),
        communicator(MpiTest::createCartesianTopology(dimensions, global_bcs)),
        plan(Slab({0, 0, 0}, cell_counts), global_bcs, MpiTest::neighbours(communicator), dimensions, communicator),
        nc({plan.localDomain().extent(0), plan.localDomain().extent(1), plan.localDomain().extent(2)}),
        size(plan.localDomain().volume()),
        laplacian_coefficients_data(size),
        laplacian_coefficients(nc[0], nc[1], nc[2], laplacian_coefficients_data),
        solution_data(PhysicsBasedPreconditioner::num_sol_vars() * size),
        solution_dPd_data(PhysicsBasedPreconditioner::num_sol_dPd_vars() * size),
        squared_sound_speed_data(size),
        solution(solution_data, nc[0], nc[1], nc[2], plan),
        solution_dPd(solution_dPd_data, nc[0], nc[1], nc[2]),
        squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_data),
        pbp(dt, ndim, nmom, mhd_enabled, solution, solution_dPd, squared_sound_speed)
    {
        for (int cell_idx = 0; cell_idx < size; cell_idx++)
        {
            DirectionalArray<double> laplacian_coefficient;
            laplacian_coefficient.fill(laplacian_coefficients_value);
            laplacian_coefficients_data[cell_idx] = laplacian_coefficient;
        }
        solution.singleVariableView(PhysicsBasedPreconditioner::n_rho()).fill(sol_rho_value);
        squared_sound_speed.fill(squared_sound_speed_value);
    }

    void checkRow(int local_row, int expected_row, std::vector<int> expected_columns, std::vector<double> expected_values)
    {
        REQUIRE(pbp.globalPressureRowIndices().at(local_row) == expected_row);
        size_t begin = pbp.globalPressureRowOffsets().at(local_row);
        size_t end   = pbp.globalPressureRowOffsets().at(local_row + 1);
        REQUIRE(end - begin == expected_columns.size());
        for (size_t n = 0; n < expected_columns.size(); n++)
        {
            CHECK(pbp.globalPressureColumnIndices()[begin + n] == expected_columns[n]);
            CHECK(pbp.pressureMatrixForSolver()[begin + n] == expected_values[n]);
        }
    }
};

TEST_CASE("PhysicsBasedPreconditioner::assemblePressureMatrix 3D")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 1) { SKIP("Test requires exactly 1 rank"); }

    const double dt         = 0.5;
    const unsigned int ndim = 3;
    const bool mhd_enabled  = true;
    int ni = 3, nj = 3, nk = 3;

    // Periodic in x and y; the k faces are never read in 2D
    DirectionalArray<BoundaryConditions> global_bcs;
    global_bcs.fill(BoundaryConditions::PERIODIC);

    AssemblePressureMatrixTestSetup setup(dt, ndim, ndim, mhd_enabled, global_bcs, {ni, nj, nk}, 4.0, 1.0, 2.0);
    Array3DWithGhosts<double> solution_rho_with_ghosts = setup.solution.singleVariableView(PhysicsBasedPreconditioner::n_rho());
    solution_rho_with_ghosts(1, 1, 1)                  = 3.0;

    setup.pbp.assemblePressureMatrix(solution_rho_with_ghosts, setup.laplacian_coefficients, setup.squared_sound_speed);

    // 4 faces and one centre value for each cell
    REQUIRE(setup.pbp.pressureMatrixForSolver().size() == ni * nj * nk * 7);
    setup.checkRow(0, 0, {0, 2, 1, 6, 3, 18, 9}, {26, -4, -4, -4, -4, -4, -4});

    // Verify coefficients are weighted by rho / rho_face
    // Cell (1, 1, 1): rho 3, every face rho_f = 2, coef = 4 * 3 / 2 = 6
    setup.checkRow(13, 13, {13, 12, 14, 10, 16, 4, 22}, {38, -6, -6, -6, -6, -6, -6});
    // Cell (0, 1, 1): rho 1, its positive x face has rho_f = 2, coef = 4 * 1 / 2 = 2
    setup.checkRow(12, 12, {12, 14, 13, 9, 15, 3, 21}, {24, -4, -2, -4, -4, -4, -4});
}

TEST_CASE("PhysicsBasedPreconditioner::assemblePressureMatrix 2D")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 1) { SKIP("Test requires exactly 1 rank"); }

    const double dt         = 0.5;
    const unsigned int ndim = 2;
    const bool mhd_enabled  = true;
    int ni = 3, nj = 3, nk = 1;

    // Periodic in x and y; the k faces are never read in 2D
    DirectionalArray<BoundaryConditions> global_bcs;
    global_bcs.fill(BoundaryConditions::PERIODIC);
    global_bcs.get(2, Direction::NEGATIVE) = BoundaryConditions::ZERO_DERIVATIVE;
    global_bcs.get(2, Direction::POSITIVE) = BoundaryConditions::ZERO_DERIVATIVE;

    AssemblePressureMatrixTestSetup setup(dt, ndim, ndim, mhd_enabled, global_bcs, {ni, nj, nk}, 4.0, 1.0, 2.0);
    Array3DWithGhosts<double> solution_rho_with_ghosts = setup.solution.singleVariableView(PhysicsBasedPreconditioner::n_rho());

    setup.pbp.assemblePressureMatrix(solution_rho_with_ghosts, setup.laplacian_coefficients, setup.squared_sound_speed);

    // 4 faces and one centre value for each cell
    REQUIRE(setup.pbp.pressureMatrixForSolver().size() == ni * nj * 5);
    setup.checkRow(0, 0, {0, 2, 1, 6, 3}, {18, -4, -4, -4, -4});
    setup.checkRow(4, 4, {4, 3, 5, 1, 7}, {18, -4, -4, -4, -4});
}

TEST_CASE("PhysicsBasedPreconditioner::assemblePressureMatrix open boundary keeps the face in the diagonal")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 1) { SKIP("Test requires exactly 1 rank"); }

    const double dt         = 0.5;
    const unsigned int ndim = 3;
    const bool mhd_enabled  = true;
    int ni = 3, nj = 3, nk = 3;

    DirectionalArray<BoundaryConditions> global_bcs;
    double expected_pressure_at_edge;
    SECTION("ZERO_DERIVATIVE boundary")
    {
        global_bcs.fill(BoundaryConditions::ZERO_DERIVATIVE);
        expected_pressure_at_edge = 26.0;
    }
    SECTION("REFLECTIVE boundary")
    {
        global_bcs.fill(BoundaryConditions::REFLECTIVE);
        expected_pressure_at_edge = 14.0;
    }

    AssemblePressureMatrixTestSetup setup(dt, ndim, ndim, mhd_enabled, global_bcs, {ni, nj, nk}, 4.0, 1.0, 2.0);
    Array3DWithGhosts<double> solution_rho_with_ghosts = setup.solution.singleVariableView(PhysicsBasedPreconditioner::n_rho());

    setup.pbp.assemblePressureMatrix(solution_rho_with_ghosts, setup.laplacian_coefficients, setup.squared_sound_speed);

    // All centres and faces minus the faces on each global boundary
    REQUIRE(setup.pbp.pressureMatrixForSolver().size() == 135);
    // Corner cell: no neighbours outside the domain, but the diagonal still has all 6 faces
    setup.checkRow(0, 0, {0, 1, 3, 9}, {expected_pressure_at_edge, -4, -4, -4});
    setup.checkRow(26, 26, {26, 25, 23, 17}, {expected_pressure_at_edge, -4, -4, -4});
    setup.checkRow(13, 13, {13, 12, 14, 10, 16, 4, 22}, {26, -4, -4, -4, -4, -4, -4});
}

TEST_CASE("PhysicsBasedPreconditioner::assemblePressureMatrix satisfies the Roberts Dynamo test from MUSIC")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 1 && num_ranks != 4) { SKIP("Test requires either 1 or 4 ranks"); }

    const double dt         = 4.5228571428571410E-004;  // Taken from roberts dynamo
    const unsigned int ndim = 3;
    const bool mhd_enabled  = true;
    int global_ni = 16, global_nj = 16, global_nk = 16;

    // Initialise problem
    DirectionalArray<BoundaryConditions> global_bcs;
    global_bcs.fill(BoundaryConditions::PERIODIC);

    AssemblePressureMatrixTestSetup setup(
        dt, ndim, ndim, mhd_enabled, global_bcs, {global_ni, global_nj, global_nk}, 1.0, 1.0, 2.5266187266788758E007);

    int my_rank;
    MPI_Comm_rank(setup.communicator, &my_rank);

    // Populate Laplacian coefficients
    std::vector<DirectionalArray<double>> laplacian_coefficients_data;
    for (int cell_idx = 0; cell_idx < setup.plan.localDomain().volume(); cell_idx++)
        setup.laplacian_coefficients_data[cell_idx].fill(2.5600000000000000E+002);

    // populate solution arrays
    Array3DWithGhosts<double> solution_rho_with_ghosts = setup.solution.singleVariableView(PhysicsBasedPreconditioner::n_rho());
    solution_rho_with_ghosts.fill(1.0);
    // Fill with something easier to spot as these shouldn't be used by assemblePressureMatrix.
    setup.solution.singleVariableView(PhysicsBasedPreconditioner::n_pressure()).fill(0);
    setup.solution.singleVariableView(PhysicsBasedPreconditioner::n_energy()).fill(0);
    setup.solution_dPd.singleVariableView(PhysicsBasedPreconditioner::n_dPdenergy()).fill(0);
    setup.solution_dPd.singleVariableView(PhysicsBasedPreconditioner::n_dPdrho()).fill(0);

    setup.pbp.assemblePressureMatrix(solution_rho_with_ghosts, setup.laplacian_coefficients, setup.squared_sound_speed);

    const std::vector<double>& actual_pressure_matrix_for_solver        = setup.pbp.pressureMatrixForSolver();
    const std::vector<long long>& actual_global_pressure_column_indices = setup.pbp.globalPressureColumnIndices();
    const std::vector<long long>& actual_global_pressure_row_indices    = setup.pbp.globalPressureRowIndices();
    const std::vector<long long>& actual_global_pressure_row_offsets    = setup.pbp.globalPressureRowOffsets();

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
