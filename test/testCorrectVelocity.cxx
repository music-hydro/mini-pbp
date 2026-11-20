#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "fixtures/GhostExchangePlanTest.cxx"
#include "fixtures/PhysicsBasedPreconditionerTest.cxx"
#include "PackedVariables.h"
#include "PhysicsBasedPreconditioner.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

struct CorrectVelocityTestSetup
{
    int ndim, nmom;
    std::array<int, 3> nc;
    size_t size;
    GhostExchangePlan<double> plan;
    std::vector<double> solution_raw;
    std::vector<double> solution_dPd_raw;
    std::vector<double> squared_sound_speed_raw;
    std::vector<double> no_variables_raw;
    PackedVariablesWithGhosts solution;
    PackedVariables solution_dPd;
    Array3D<double> squared_sound_speed;
    PackedVariables no_variables;
    PhysicsBasedPreconditioner pbp;

    /**
     * @brief Allocate shared test variables using a periodic ghost-exchange plan.
     *
     * @param cell_counts Number of cells in the i, j, and k directions.
     * @param primitive_initial_value Initial value for all primitive-variable storage.
     */
    explicit CorrectVelocityTestSetup(const int ndim,
                                      const int nmom,
                                      const bool mhd_enabled,
                                      const std::array<int, 3>& cell_counts) :
        CorrectVelocityTestSetup(ndim, nmom, mhd_enabled, cell_counts, GhostExchangePlanTest::periodicPlan())
    {
    }

    /**
     * @brief Allocate shared test variables using the supplied ghost-exchange plan.
     *
     * @param cell_counts Number of cells in the i, j, and k directions.
     * @param exchange_plan Ghost-exchange plan used by the solution variables.
     * @param primitive_initial_value Initial value for all primitive-variable storage.
     * @param conserved_initial_value Initial value for all conserved-variable storage.
     */
    CorrectVelocityTestSetup(const int ndim,
                             const int nmom,
                             const bool mhd_enabled,
                             const std::array<int, 3>& cell_counts,
                             const GhostExchangePlan<double>& exchange_plan) :
        nc(cell_counts),
        size(static_cast<size_t>(nc[0]) * nc[1] * nc[2]),
        plan(exchange_plan),
        solution_raw(PhysicsBasedPreconditioner::num_sol_vars() * size),
        solution_dPd_raw(PhysicsBasedPreconditioner::num_sol_dPd_vars() * size),
        squared_sound_speed_raw(size),
        solution(solution_raw, nc[0], nc[1], nc[2], plan),
        solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]),
        squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_raw),
        no_variables(no_variables_raw, nc[0], nc[1], nc[2]),
        pbp(4.5228571428571410E-004,
            ndim,
            nmom,
            mhd_enabled,
            {0.0625, 0.0625, 0.0625},
            solution,
            solution_dPd,
            squared_sound_speed)
    {
    }
};

/**
 * @brief Run PhysicsBasedPreconditioner::correctVelocity on the Roberts Dynamo data of one rank and compare to expected results
 *
 * @details Inputs and expected outputs are read from `.dat` files with prefix `file_prefix`.
 *
 * @param[in] file_prefix - The prefix of the files containing inputs and expected outputs.
 * @param[in] plan - The ghost exchange plan required for the chosen number opf ranks (should be periodic in all directions).
 */
void checkAgainstRobertsDynamo(const std::string& file_prefix, const GhostExchangePlan<double>& plan)
{
    FILE* inputs_file = fopen((file_prefix + "_inputs.dat").c_str(), "r");
    REQUIRE(inputs_file != nullptr);

    // Static definition of the Roberts Dynamo problem
    unsigned int ndim = 3;
    unsigned int nmom = 3;
    bool use_mhd      = false;
    // The local dimensions change per rank and thus must be read from the file
    std::array<int, 3> start, end;
    REQUIRE(fscanf(inputs_file, " local slab %d %d %d %d %d %d", &start[0], &start[1], &start[2], &end[0], &end[1], &end[2]) ==
            6);
    Slab local_slab(start, end);
    int ni = local_slab.extent(0);
    int nj = local_slab.extent(1);
    int nk = local_slab.extent(2);
    CorrectVelocityTestSetup setup(ndim, nmom, use_mhd, end, plan);

    // Read in solution density
    Array3DWithGhosts<double> sol_rho = setup.pbp.solution().singleVariableView(PhysicsBasedPreconditioner::n_rho());
    for (int k = -1; k <= nk; k++)
    {
        for (int j = -1; j <= nj; j++)
        {
            for (int i = -1; i <= ni; i++)
            {
                // Skip edges and corners
                if ((i == -1 && (j == -1 || k == -1 || j == nj || k == nk)) ||
                    (i == ni && (j == -1 || k == -1 || j == nj || k == nk)) || (j == -1 && (k == -1 || k == nk)) ||
                    (j == nj && (k == -1 || k == nk)))
                    continue;
                REQUIRE(fscanf(inputs_file, " %lf", &sol_rho(i, j, k)));
            }
        }
    }

    // Read in delta pressure
    REQUIRE(fscanf(inputs_file, " Pressure delta") == 0);
    std::vector<double> delta_pressure_raw(local_slab.volume());
    Array3D<double> delta_pressure(ni, nj, nk, delta_pressure_raw);
    for (int k = 0; k < nk; k++)
        for (int j = 0; j < nj; j++)
            for (int i = 0; i < ni; i++)
                REQUIRE(fscanf(inputs_file, " %lf", &delta_pressure(i, j, k)) == 1);

    Array3DWithGhosts<double> delta_pressure_g(delta_pressure, plan);
    delta_pressure_g.exchangeGhosts();

    // Read in velocity starting state
    std::vector<double> actual_delta_velocities_raw(ndim * local_slab.volume());
    PackedVariables actual_delta_velocities(actual_delta_velocities_raw, ni, nj, nk);
    for (unsigned int axis = 0; axis < ndim; axis++)
    {
        int ax;
        REQUIRE(fscanf(inputs_file, " Velocity %i", &ax) == 1);
        REQUIRE(ax == axis);
        for (int k = 0; k < nk; k++)
            for (int j = 0; j < nj; j++)
                for (int i = 0; i < ni; i++)
                    REQUIRE(fscanf(inputs_file, " %lf", &actual_delta_velocities.singleVariableView(axis)(i, j, k)));
    }

    // Run correctVelocity
    setup.pbp.correctVelocity(delta_pressure_g, actual_delta_velocities);

    FILE* expected_outputs_file = fopen((file_prefix + "_expected_outputs.dat").c_str(), "r");
    REQUIRE(expected_outputs_file != nullptr);

    // Read in expected output velocities
    std::vector<double> expected_delta_velocities_raw(ndim * local_slab.volume());
    PackedVariables expected_delta_velocities(expected_delta_velocities_raw, ni, nj, nk);
    for (unsigned int axis = 0; axis < ndim; axis++)
    {
        for (int k = 0; k < nk; k++)
        {
            for (int j = 0; j < nj; j++)
            {
                for (int i = 0; i < ni; i++)
                {
                    double expected_value;
                    REQUIRE(fscanf(expected_outputs_file, " %lf", &expected_value) == 1);
                    CHECK_THAT(actual_delta_velocities.singleVariableView(axis)(i, j, k),
                               Catch::Matchers::WithinRel(expected_value, 1e-12));
                }
            }
        }
    }

    // Validate results
}

TEST_CASE("PhysicsBasedPreconditioner::correctVelocity roberts dynamo problem 1 rank")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 1) { SKIP("Test requires exactly 1 rank"); }

    checkAgainstRobertsDynamo(std::string(TEST_DATA_DIR) + "/velocity_correction_roberts_dynamo_1rank_rank0",
                              GhostExchangePlanTest::periodicPlan());
}
