#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "DirectionalArray.h"
#include "fixtures/GhostExchangePlanTest.cxx"
#include "GhostExchangePlan.h"
#include "PackedVariables.h"
#include "PhysicsBasedPreconditioner.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    /**
     * @brief Initialize the density, pressure, energy, and velocity variables.
     *
     * @param variables Packed variables to initialize.
     * @param pbp Preconditioner providing the variable indices.
     * @param rho Value for density.
     * @param pressure Value for pressure.
     * @param energy Value for energy.
     * @param velocities Values for the velocity components.
     */
    template<typename T>
    void initializeVariables(T& variables,
                             PhysicsBasedPreconditioner& pbp,
                             const double rho,
                             const double pressure,
                             const double energy,
                             const std::array<double, 3>& velocities)
    {
        variables.singleVariableView(pbp.n_rho()).fill(rho);
        variables.singleVariableView(pbp.n_pressure()).fill(pressure);
        variables.singleVariableView(pbp.n_energy()).fill(energy);
        const std::array<unsigned int, 3> velocity_indices = {pbp.n_vel_i(), pbp.n_vel_j(), pbp.n_vel_k()};
        for (unsigned int axis = 0; axis < velocity_indices.size(); axis++)
        {
            variables.singleVariableView(velocity_indices[axis]).fill(velocities[axis]);
        }
    }

    /**
     * @brief A wrapper to initialise the variables shared by most tests
     */
    struct ConservedToPrimitiveTestSetup
    {
        int ndim, nmom;
        std::array<int, 3> nc;
        size_t size;
        GhostExchangePlan<double> plan;
        std::vector<double> solution_raw;
        std::vector<double> solution_dPd_raw;
        std::vector<double> squared_sound_speed_raw;
        std::vector<double> conserved_raw;
        std::vector<double> primitive_raw;
        std::vector<double> no_variables_raw;
        PackedVariablesWithGhosts solution;
        PackedVariables solution_dPd;
        Array3D<double> squared_sound_speed;
        PackedVariables conserved;
        PackedVariables primitive;
        PackedVariables no_variables;
        PhysicsBasedPreconditioner pbp;

        /**
         * @brief Allocate shared test variables using a periodic ghost-exchange plan.
         *
         * @param cell_counts Number of cells in the i, j, and k directions.
         * @param primitive_initial_value Initial value for all primitive-variable storage.
         */
        explicit ConservedToPrimitiveTestSetup(const int ndim,
                                               const int nmom,
                                               const bool mhd_enabled,
                                               const std::array<int, 3>& cell_counts,
                                               const double primitive_initial_value = 0.0) :
            ConservedToPrimitiveTestSetup(ndim,
                                          nmom,
                                          mhd_enabled,
                                          cell_counts,
                                          GhostExchangePlanTest::periodicPlan(),
                                          primitive_initial_value)
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
        ConservedToPrimitiveTestSetup(const int ndim,
                                      const int nmom,
                                      const bool mhd_enabled,
                                      const std::array<int, 3>& cell_counts,
                                      const GhostExchangePlan<double>& exchange_plan,
                                      const double primitive_initial_value = 0.0,
                                      const double conserved_initial_value = 1.0) :
            nc(cell_counts),
            size(static_cast<size_t>(nc[0]) * nc[1] * nc[2]),
            plan(exchange_plan),
            solution_raw(PhysicsBasedPreconditioner::num_sol_vars() * size),
            solution_dPd_raw(PhysicsBasedPreconditioner::num_sol_dPd_vars() * size),
            squared_sound_speed_raw(size),
            conserved_raw(PhysicsBasedPreconditioner::num_sol_vars() * size, conserved_initial_value),
            primitive_raw(PhysicsBasedPreconditioner::num_sol_vars() * size, primitive_initial_value),
            solution(solution_raw, nc[0], nc[1], nc[2], plan),
            solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]),
            squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_raw),
            conserved(conserved_raw, nc[0], nc[1], nc[2]),
            primitive(primitive_raw, nc[0], nc[1], nc[2]),
            no_variables(no_variables_raw, nc[0], nc[1], nc[2]),
            pbp(ndim, nmom, mhd_enabled, solution, solution_dPd, squared_sound_speed)
        {
        }
    };

    /// @brief Every value of `a` equals `expected` (or is NaN when `expected` is NaN, i.e. not written)
    void requireAll(const Array3D<double>& a, const double expected)
    {
        const Slab s = a.cellSlab();
        for (int k = s.start(2); k < s.end(2); k++)
        {
            for (int j = s.start(1); j < s.end(1); j++)
            {
                for (int i = s.start(0); i < s.end(0); i++)
                {
                    if (std::isnan(expected)) { REQUIRE(std::isnan(a(i, j, k))); }
                    else
                    {
                        REQUIRE_THAT(a(i, j, k), Catch::Matchers::WithinRel(expected, 1e-14));
                    }
                }
            }
        }
    }

    /**
     * @brief Run PhysicsBasedPreconditioner::conservedToPrimitive on the Roberts Dynamo data of one rank and compare to expected results
     *
     * @details Inputs and expected outputs are read from `.dat` files with prefix `file_prefix`.
     *
     * @param[in] file_prefix - The prefix of the files containing inputs and expected outputs.
     * @param[in] plan - The ghost exchange plan required for the chosen number opf ranks (should be periodic in all directions).
     */
    void checkAgainstRobertsDynamo(const std::string& file_prefix, GhostExchangePlan<double>& plan)
    {
        FILE* file = fopen((file_prefix + "_inputs.dat").c_str(), "r");
        REQUIRE(file != nullptr);

        // Static definition of the Roberts Dynamo problem
        unsigned int ndim = 3;
        unsigned int nmom = 3;
        bool use_mhd      = false;
        int ni, nj, nk;
        // The local dimensions change per rank and thus must be read from the file
        REQUIRE(fscanf(file, " local_cells %d %d %d", &ni, &nj, &nk) == 3);
        Slab local_slab({0, 0, 0}, {ni, nj, nk});

        ConservedToPrimitiveTestSetup setup(ndim, nmom, use_mhd, {ni, nj, nk}, plan, NaN, 0.0);
        std::vector<double> expected_primitive_variables_raw(PhysicsBasedPreconditioner::num_sol_vars() * setup.size);
        PackedVariables expected_primitive_variables(expected_primitive_variables_raw, ni, nj, nk);
        auto& actual_primitive_variables = setup.primitive;

        // Read the solution values (including the ghosts)
        Array3DWithGhosts<double> solution_rho    = setup.solution.singleVariableView(setup.pbp.n_rho());
        Array3DWithGhosts<double> solution_energy = setup.solution.singleVariableView(setup.pbp.n_energy());
        Array3DWithGhosts<double> solution_vel_i  = setup.solution.singleVariableView(setup.pbp.n_vel_i());
        Array3DWithGhosts<double> solution_vel_j  = setup.solution.singleVariableView(setup.pbp.n_vel_j());
        Array3DWithGhosts<double> solution_vel_k  = setup.solution.singleVariableView(setup.pbp.n_vel_k());
        for (int k = -1; k < nk + 1; k++)
        {
            for (int j = -1; j < nj + 1; j++)
            {
                for (int i = -1; i < ni + 1; i++)
                {
                    // Skip corner cells
                    if ((i == -1 && (j == -1 || k == -1 || j == nj || k == nk)) ||
                        (i == ni && (j == -1 || k == -1 || j == nj || k == nk)) || (j == -1 && (k == -1 || k == nk)) ||
                        (j == nj && (k == -1 || k == nk)))
                        continue;

                    REQUIRE(fscanf(file,
                                   " %lf %lf %lf %lf %lf",
                                   &solution_rho(i, j, k),
                                   &solution_energy(i, j, k),
                                   &solution_vel_i(i, j, k),
                                   &solution_vel_j(i, j, k),
                                   &solution_vel_k(i, j, k)) == 5);
                }
            }
        }

        // Convenient store of the relevant variable indicies for this problem
        std::vector<unsigned int> relevant_var_indices(5);
        relevant_var_indices = {
            setup.pbp.n_rho(), setup.pbp.n_energy(), setup.pbp.n_vel_i(), setup.pbp.n_vel_j(), setup.pbp.n_vel_k()};
        // Read the conserved values as a flat list of one element per row
        for (unsigned int idx = 0; idx < relevant_var_indices.size(); idx++)
        {
            unsigned int ivar        = relevant_var_indices[idx];
            Array3D<double> var_view = setup.conserved.singleVariableView(ivar);
            for (int k = 0; k < nk; k++)
            {
                for (int j = 0; j < nj; j++)
                {
                    for (int i = 0; i < ni; i++)
                    {
                        REQUIRE(fscanf(file, " %lf", &var_view(i, j, k)) == 1);
                    }
                }
            }
        }
        // Read the conserved rho halo values in the negative direction as this is the only direction we read from in conservedToPrimitive
        Array3DWithGhosts<double> delta_rho_g(setup.conserved.singleVariableView(setup.pbp.n_rho()), plan);
        delta_rho_g.exchangeGhosts();  // Populate all halos to start with but negative ones will be overwritten shortly
        for (unsigned int axis = 0; axis < 3; axis++)
        {
            unsigned int axis_read_from_file, halo_size;
            REQUIRE(fscanf(file, " halo %u %u", &axis_read_from_file, &halo_size) == 2);
            REQUIRE(axis == axis_read_from_file - 1);
            REQUIRE(halo_size == local_slab.volume() / local_slab.extent(axis));
            for (unsigned int n = 0; n < halo_size; n++)
            {
                int i, j, k;
                double value;
                REQUIRE(fscanf(file, " %d %d %d %lf", &i, &j, &k, &value) == 4);
                REQUIRE(delta_rho_g(i - 1, j - 1, k - 1) == value);  // Offset by -1 to map from MUSIC to mini-app indexing
            }
        }
        fclose(file);

        // Read the expected outputs from file
        file = fopen((file_prefix + "_expected_outputs.dat").c_str(), "r");
        REQUIRE(file != nullptr);

        // Read the primitive values as a flat list of one element per row
        for (unsigned int idx = 0; idx < relevant_var_indices.size(); idx++)
        {
            unsigned int ivar        = relevant_var_indices[idx];
            Array3D<double> var_view = expected_primitive_variables.singleVariableView(ivar);
            for (unsigned int k = 0; k < nk; k++)
            {
                for (unsigned int j = 0; j < nj; j++)
                {
                    for (unsigned int i = 0; i < ni; i++)
                    {
                        REQUIRE(fscanf(file, " %lf", &var_view(i, j, k)) == 1);
                    }
                }
            }
        }
        fclose(file);

        // The magnetic field is a copy and is not in the data; it is covered by the cases above
        Array3D<double> actual_primitive_rho    = actual_primitive_variables.singleVariableView(setup.pbp.n_rho());
        Array3D<double> actual_primitive_energy = actual_primitive_variables.singleVariableView(setup.pbp.n_energy());
        setup.pbp.conservedToPrimitive(plan,
                                       setup.no_variables,
                                       setup.conserved,
                                       setup.no_variables,
                                       setup.no_variables,
                                       actual_primitive_variables,
                                       setup.no_variables,
                                       setup.no_variables);

        // Verify the actual primitive variable views have been populated with the expected values
        for (unsigned int k = 0; k < nk; k++)
        {
            for (unsigned int j = 0; j < nj; j++)
            {
                for (unsigned int i = 0; i < ni; i++)
                {
                    REQUIRE_THAT(actual_primitive_rho(i, j, k),
                                 Catch::Matchers::WithinRel(
                                     expected_primitive_variables.singleVariableView(setup.pbp.n_rho())(i, j, k), 1e-12));
                    REQUIRE_THAT(actual_primitive_energy(i, j, k),
                                 Catch::Matchers::WithinRel(
                                     expected_primitive_variables.singleVariableView(setup.pbp.n_energy())(i, j, k), 1e-12));
                    REQUIRE_THAT(actual_primitive_variables.singleVariableView(setup.pbp.n_vel_i())(i, j, k),
                                 Catch::Matchers::WithinRel(
                                     expected_primitive_variables.singleVariableView(setup.pbp.n_vel_i())(i, j, k), 1e-12));
                    REQUIRE_THAT(actual_primitive_variables.singleVariableView(setup.pbp.n_vel_j())(i, j, k),
                                 Catch::Matchers::WithinRel(
                                     expected_primitive_variables.singleVariableView(setup.pbp.n_vel_j())(i, j, k), 1e-12));
                    REQUIRE_THAT(actual_primitive_variables.singleVariableView(setup.pbp.n_vel_k())(i, j, k),
                                 Catch::Matchers::WithinRel(
                                     expected_primitive_variables.singleVariableView(setup.pbp.n_vel_k())(i, j, k), 1e-12));
                }
            }
        }
    }
}  // namespace

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive throws if active scalar inputs don't have matching counts")
{
    ConservedToPrimitiveTestSetup setup(3, 3, true, {3, 3, 3});
    std::vector<double> scalar_raw(setup.size);
    PackedVariables scalar(scalar_raw, setup.nc[0], setup.nc[1], setup.nc[2]);
    REQUIRE_THROWS_WITH(setup.pbp.conservedToPrimitive(setup.plan,
                                                       scalar,
                                                       setup.conserved,
                                                       setup.no_variables,
                                                       setup.no_variables,
                                                       setup.primitive,
                                                       setup.no_variables,
                                                       setup.no_variables),
                        Catch::Matchers::ContainsSubstring("The number of active scalars differs between the inputs"));
}

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive 3D with active scalars and MHD")
{
    ConservedToPrimitiveTestSetup setup(3, 3, true, {3, 4, 5}, NaN);

    // Inputs
    const double input_sol_rho                           = 2.0;
    const double input_conserved_rho                     = 3.0;
    const double input_sol_energy                        = 3.0;
    const double input_conserved_energy                  = 10.0;
    const double input_conserved_scalar                  = 50.0;
    const std::array<double, 3> input_sol_vel            = {4.0, 5.0, 6.0};
    const std::array<double, 3> input_conserved_vel      = {20.0, 30.0, 40.0};
    const std::array<double, 3> input_magnetic_primitive = {1.0, 2.0, 3.0};

    // Expected outputs
    double expected_primitive_rho                           = input_conserved_rho;
    const double expected_primitive_energy                  = 0.5;
    const std::array<double, 2> expected_primitive_scalars  = {14.5, 13.0};
    const std::array<double, 3> expected_primitive_vel      = {4.0, 7.5, 11.0};
    const std::array<double, 3> expected_magnetic_primitive = input_magnetic_primitive;

    initializeVariables(setup.solution, setup.pbp, input_sol_rho, 0.0, input_sol_energy, input_sol_vel);
    initializeVariables(setup.conserved, setup.pbp, input_conserved_rho, 0.0, input_conserved_energy, input_conserved_vel);

    std::vector<double> scalar_solution_raw(2 * setup.size);
    std::vector<double> scalar_conserved_raw(2 * setup.size);
    std::vector<double> scalar_primitive_raw(2 * setup.size, NaN);
    PackedVariables scalar_solution(scalar_solution_raw, setup.nc[0], setup.nc[1], setup.nc[2]);
    PackedVariables scalar_conserved(scalar_conserved_raw, setup.nc[0], setup.nc[1], setup.nc[2]);
    PackedVariables scalar_primitive(scalar_primitive_raw, setup.nc[0], setup.nc[1], setup.nc[2]);
    for (unsigned int scalar = 0; scalar < 2; scalar++)
    {
        scalar_solution.singleVariableView(scalar).fill(7.0 + scalar);
        scalar_conserved.singleVariableView(scalar).fill(input_conserved_scalar);
    }

    std::vector<double> magnetic_conserved_raw(3 * setup.size);
    std::vector<double> magnetic_primitive_raw(3 * setup.size, NaN);
    PackedVariables magnetic_conserved(magnetic_conserved_raw, setup.nc[0], setup.nc[1], setup.nc[2]);
    PackedVariables magnetic_primitive(magnetic_primitive_raw, setup.nc[0], setup.nc[1], setup.nc[2]);
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        magnetic_conserved.singleVariableView(axis).fill(input_magnetic_primitive[axis]);
    }

    setup.pbp.conservedToPrimitive(setup.plan,
                                   scalar_solution,
                                   setup.conserved,
                                   scalar_conserved,
                                   magnetic_conserved,
                                   setup.primitive,
                                   scalar_primitive,
                                   magnetic_primitive);

    requireAll(setup.primitive.singleVariableView(setup.pbp.n_rho()), expected_primitive_rho);
    requireAll(setup.primitive.singleVariableView(setup.pbp.n_energy()), expected_primitive_energy);
    for (unsigned int scalar = 0; scalar < 2; scalar++)
    {
        requireAll(scalar_primitive.singleVariableView(scalar), expected_primitive_scalars[scalar]);
    }

    requireAll(setup.primitive.singleVariableView(setup.pbp.n_vel_i()), expected_primitive_vel[0]);
    requireAll(setup.primitive.singleVariableView(setup.pbp.n_vel_j()), expected_primitive_vel[1]);
    requireAll(setup.primitive.singleVariableView(setup.pbp.n_vel_k()), expected_primitive_vel[2]);

    for (unsigned int axis = 0; axis < 3; axis++)
    {
        requireAll(magnetic_primitive.singleVariableView(axis), expected_magnetic_primitive[axis]);
    }
}

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive 2D with two velocity components and no MHD leaves the rest untouched")
{
    ConservedToPrimitiveTestSetup setup(2, 2, false, {3, 3, 1}, NaN);

    // Inputs
    const double input_sol_rho                      = 2.0;
    const double input_conserved_rho                = 3.0;
    const double input_sol_energy                   = 3.0;
    const double input_conserved_energy             = 10.0;
    const std::array<double, 3> input_sol_vel       = {4.0, 5.0, 6.0};
    const std::array<double, 3> input_conserved_vel = {20.0, 30.0, 40.0};

    // Expected outputs
    const double expected_primitive_rho                = input_conserved_rho;
    const double expected_primitive_energy             = 0.5;
    const std::array<double, 3> expected_primitive_vel = {4.0, 7.5, NaN};

    initializeVariables(setup.solution, setup.pbp, input_sol_rho, 0.0, input_sol_energy, input_sol_vel);
    initializeVariables(setup.conserved, setup.pbp, input_conserved_rho, 0.0, input_conserved_energy, input_conserved_vel);

    std::vector<double> magnetic_conserved_raw(3 * setup.size);
    std::vector<double> magnetic_primitive_raw(3 * setup.size, NaN);
    PackedVariables magnetic_conserved(magnetic_conserved_raw, setup.nc[0], setup.nc[1], setup.nc[2]);
    PackedVariables magnetic_primitive(magnetic_primitive_raw, setup.nc[0], setup.nc[1], setup.nc[2]);
    setup.pbp.conservedToPrimitive(setup.plan,
                                   setup.no_variables,
                                   setup.conserved,
                                   setup.no_variables,
                                   magnetic_conserved,
                                   setup.primitive,
                                   setup.no_variables,
                                   magnetic_primitive);

    requireAll(setup.primitive.singleVariableView(setup.pbp.n_rho()), expected_primitive_rho);
    requireAll(setup.primitive.singleVariableView(setup.pbp.n_energy()), expected_primitive_energy);
    requireAll(setup.primitive.singleVariableView(setup.pbp.n_vel_i()), expected_primitive_vel[0]);
    requireAll(setup.primitive.singleVariableView(setup.pbp.n_vel_j()), expected_primitive_vel[1]);
    requireAll(setup.primitive.singleVariableView(setup.pbp.n_vel_k()), expected_primitive_vel[2]);
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        requireAll(magnetic_primitive.singleVariableView(axis), NaN);
    }
}

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive converts each velocity component")
{
    ConservedToPrimitiveTestSetup setup(3, 3, false, {2, 2, 2}, NaN);

    initializeVariables(setup.solution, setup.pbp, 2.0, 0.0, 3.0, {1.0, 2.0, 3.0});
    initializeVariables(setup.conserved, setup.pbp, 2.0, 0.0, 10.0, {8.0, 16.0, 24.0});
    setup.pbp.conservedToPrimitive(setup.plan,
                                   setup.no_variables,
                                   setup.conserved,
                                   setup.no_variables,
                                   setup.no_variables,
                                   setup.primitive,
                                   setup.no_variables,
                                   setup.no_variables);

    requireAll(setup.primitive.singleVariableView(setup.pbp.n_vel_i()), 3.0);
    requireAll(setup.primitive.singleVariableView(setup.pbp.n_vel_j()), 6.0);
    requireAll(setup.primitive.singleVariableView(setup.pbp.n_vel_k()), 9.0);
}

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive matches MUSIC cons_to_prim on the Roberts dynamo test on 1 rank")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 1) { SKIP("Test requires exactly 1 rank"); }

    GhostExchangePlan<double> plan = GhostExchangePlanTest::periodicPlan();
    checkAgainstRobertsDynamo(std::string(TEST_DATA_DIR) + "/cons_to_prim_roberts_dynamo_1rank_rank0", plan);
}

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive matches MUSIC cons_to_prim on the Roberts dynamo test on 4 ranks")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 4) { SKIP("Test requires exactly 4 rank"); }

    // MUSIC splits the periodic 16^3 domain 2 x 2 x 1 with rank = 2 * (x block) + (y block), which is the order of a
    // cartesian communicator of dims {2, 2, 1} without reordering
    std::array<int, 3> dims    = {2, 2, 1};
    std::array<int, 3> periods = {1, 1, 1};
    MPI_Comm cartesian;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods.data(), false, &cartesian);
    int rank;
    MPI_Comm_rank(cartesian, &rank);

    DirectionalArray<int> neighbours;
    for (int axis = 0; axis < 3; axis++)
    {
        MPI_Cart_shift(
            cartesian, axis, 1, &neighbours.get(axis, Direction::NEGATIVE), &neighbours.get(axis, Direction::POSITIVE));
    }
    DirectionalArray<BoundaryConditions> inner;
    inner.fill(BoundaryConditions::INNER);
    GhostExchangePlan<double> plan(inner, neighbours, cartesian);
    checkAgainstRobertsDynamo(std::string(TEST_DATA_DIR) + "/cons_to_prim_roberts_dynamo_4ranks_rank" + std::to_string(rank),
                              plan);
    MPI_Comm_free(&cartesian);
}
