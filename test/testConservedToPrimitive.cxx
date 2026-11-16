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

    std::array<unsigned int, 3> velocityIndices(PhysicsBasedPreconditioner& pbp)
    { return {pbp.n_vel_i(), pbp.n_vel_j(), pbp.n_vel_k()}; }

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
        const std::array<unsigned int, 3> velocity_indices = velocityIndices(pbp);
        for (unsigned int axis = 0; axis < velocity_indices.size(); axis++)
        {
            variables.singleVariableView(velocity_indices[axis]).fill(velocities[axis]);
        }
    }

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

        size_t size                 = ni * nj * nk;
        unsigned int num_total_vars = 6;
        std::vector<double> solution_raw(num_total_vars * size);
        std::vector<double> solution_dPd_raw(2 * size);
        std::vector<double> conserved_variables_raw(num_total_vars * size);
        std::vector<double> expected_primitive_variables_raw(num_total_vars * size);
        std::vector<double> actual_primitive_variables_raw(num_total_vars * size);
        PackedVariablesWithGhosts solution(solution_raw, ni, nj, nk, plan);
        PackedVariables solution_dPd(solution_dPd_raw, ni, nj, nk);
        PackedVariables conserved_variables(conserved_variables_raw, ni, nj, nk);
        PackedVariables expected_primitive_variables(expected_primitive_variables_raw, ni, nj, nk);
        PackedVariables actual_primitive_variables(actual_primitive_variables_raw, ni, nj, nk);
        Array3D<double> squared_sound_speed(ni, nj, nk);
        PhysicsBasedPreconditioner pbp(solution, solution_dPd, squared_sound_speed);

        // Read the solution values (including the ghosts)
        Array3DWithGhosts<double> solution_rho    = solution.singleVariableView(pbp.n_rho());
        Array3DWithGhosts<double> solution_energy = solution.singleVariableView(pbp.n_energy());
        Array3DWithGhosts<double> solution_vel_i  = solution.singleVariableView(pbp.n_vel_i());
        Array3DWithGhosts<double> solution_vel_j  = solution.singleVariableView(pbp.n_vel_j());
        Array3DWithGhosts<double> solution_vel_k  = solution.singleVariableView(pbp.n_vel_k());
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
        relevant_var_indices = {pbp.n_rho(), pbp.n_energy(), pbp.n_vel_i(), pbp.n_vel_j(), pbp.n_vel_k()};
        // Read the conserved values as a flat list of one element per row
        for (unsigned int idx = 0; idx < relevant_var_indices.size(); idx++)
        {
            unsigned int ivar        = relevant_var_indices[idx];
            Array3D<double> var_view = conserved_variables.singleVariableView(ivar);
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
        Array3DWithGhosts<double> delta_rho_g(conserved_variables.singleVariableView(pbp.n_rho()), plan);
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
        Array3D<double> actual_primitive_rho    = actual_primitive_variables.singleVariableView(pbp.n_rho());
        Array3D<double> actual_primitive_energy = actual_primitive_variables.singleVariableView(pbp.n_energy());
        std::vector<double> no_variables_raw;
        PackedVariables no_variables(no_variables_raw, ni, nj, nk);
        pbp.conservedToPrimitive(ndim,
                                 nmom,
                                 use_mhd,
                                 plan,
                                 no_variables,
                                 conserved_variables,
                                 no_variables,
                                 no_variables,
                                 actual_primitive_variables,
                                 no_variables,
                                 no_variables);

        // Verify the actual primitive variable views have been populated with the expected values
        for (unsigned int k = 0; k < nk; k++)
        {
            for (unsigned int j = 0; j < nj; j++)
            {
                for (unsigned int i = 0; i < ni; i++)
                {
                    REQUIRE_THAT(
                        actual_primitive_rho(i, j, k),
                        Catch::Matchers::WithinRel(expected_primitive_variables.singleVariableView(pbp.n_rho())(i, j, k), 1e-12));
                    REQUIRE_THAT(actual_primitive_energy(i, j, k),
                                 Catch::Matchers::WithinRel(
                                     expected_primitive_variables.singleVariableView(pbp.n_energy())(i, j, k), 1e-12));
                    REQUIRE_THAT(actual_primitive_variables.singleVariableView(pbp.n_vel_i())(i, j, k),
                                 Catch::Matchers::WithinRel(
                                     expected_primitive_variables.singleVariableView(pbp.n_vel_i())(i, j, k), 1e-12));
                    REQUIRE_THAT(actual_primitive_variables.singleVariableView(pbp.n_vel_j())(i, j, k),
                                 Catch::Matchers::WithinRel(
                                     expected_primitive_variables.singleVariableView(pbp.n_vel_j())(i, j, k), 1e-12));
                    REQUIRE_THAT(actual_primitive_variables.singleVariableView(pbp.n_vel_k())(i, j, k),
                                 Catch::Matchers::WithinRel(
                                     expected_primitive_variables.singleVariableView(pbp.n_vel_k())(i, j, k), 1e-12));
                }
            }
        }
    }
}  // namespace

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive rejects invalid input")
{
    const std::array<int, 3> nc    = {3, 3, 3};
    const size_t size              = static_cast<size_t>(nc[0]) * nc[1] * nc[2];
    GhostExchangePlan<double> plan = GhostExchangePlanTest::periodicPlan();
    std::vector<double> solution_raw(6 * size);
    std::vector<double> solution_dPd_raw(2 * size);
    std::vector<double> squared_sound_speed_raw(size);
    PackedVariablesWithGhosts solution(solution_raw, nc[0], nc[1], nc[2], plan);
    PackedVariables solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]);
    Array3D<double> squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_raw);
    PhysicsBasedPreconditioner pbp(solution, solution_dPd, squared_sound_speed);
    std::vector<double> conserved_raw(6 * size);
    std::vector<double> primitive_raw(6 * size);
    PackedVariables conserved(conserved_raw, nc[0], nc[1], nc[2]);
    PackedVariables primitive(primitive_raw, nc[0], nc[1], nc[2]);
    std::vector<double> no_variables_raw;
    PackedVariables no_variables(no_variables_raw, nc[0], nc[1], nc[2]);

    SECTION("invalid input ndim and/or nmom")
    {
        for (const auto [ndim, nmom, expected_err] : std::array<std::tuple<unsigned int, unsigned int, std::string>, 5> {
                 {{1, 3, "ndim must be 2 or 3"},
                  {4, 3, "ndim must be 2 or 3"},
                  {3, 1, "nmom must be 2 or 3"},
                  {3, 4, "nmom must be 2 or 3"},
                  {3, 2, "nmom must be at least ndim"}}
        })
        {
            REQUIRE_THROWS_WITH(pbp.conservedToPrimitive(ndim,
                                                         nmom,
                                                         false,
                                                         plan,
                                                         no_variables,
                                                         conserved,
                                                         no_variables,
                                                         no_variables,
                                                         primitive,
                                                         no_variables,
                                                         no_variables),
                                Catch::Matchers::ContainsSubstring(expected_err));
        }
    }

    SECTION("active scalar inputs have matching counts")
    {
        std::vector<double> scalar_raw(size);
        PackedVariables scalar(scalar_raw, nc[0], nc[1], nc[2]);
        REQUIRE_THROWS_WITH(
            pbp.conservedToPrimitive(
                3, 3, false, plan, scalar, conserved, no_variables, no_variables, primitive, no_variables, no_variables),
            Catch::Matchers::ContainsSubstring("The number of active scalars differs between the inputs"));
    }
}

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive 3D with active scalars and MHD")
{
    const std::array<int, 3> nc    = {3, 4, 5};
    const size_t size              = static_cast<size_t>(nc[0]) * nc[1] * nc[2];
    GhostExchangePlan<double> plan = GhostExchangePlanTest::periodicPlan();
    std::vector<double> solution_raw(6 * size);
    std::vector<double> solution_dPd_raw(2 * size);
    std::vector<double> squared_sound_speed_raw(size);
    PackedVariablesWithGhosts solution(solution_raw, nc[0], nc[1], nc[2], plan);
    PackedVariables solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]);
    Array3D<double> squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_raw);
    PhysicsBasedPreconditioner pbp(solution, solution_dPd, squared_sound_speed);

    initializeVariables(solution, pbp, 2.0, 0.0, 3.0, {4.0, 5.0, 6.0});

    std::vector<double> conserved_raw(6 * size, 1.0);
    std::vector<double> primitive_raw(6 * size, NaN);
    PackedVariables conserved(conserved_raw, nc[0], nc[1], nc[2]);
    PackedVariables primitive(primitive_raw, nc[0], nc[1], nc[2]);
    initializeVariables(conserved, pbp, 1.0, 0.0, 10.0, {20.0, 30.0, 40.0});

    std::vector<double> scalar_solution_raw(2 * size);
    std::vector<double> scalar_conserved_raw(2 * size);
    std::vector<double> scalar_primitive_raw(2 * size, NaN);
    PackedVariables scalar_solution(scalar_solution_raw, nc[0], nc[1], nc[2]);
    PackedVariables scalar_conserved(scalar_conserved_raw, nc[0], nc[1], nc[2]);
    PackedVariables scalar_primitive(scalar_primitive_raw, nc[0], nc[1], nc[2]);
    for (unsigned int scalar = 0; scalar < 2; scalar++)
    {
        scalar_solution.singleVariableView(scalar).fill(7.0 + scalar);
        scalar_conserved.singleVariableView(scalar).fill(50.0);
    }

    std::vector<double> magnetic_conserved_raw(3 * size);
    std::vector<double> magnetic_primitive_raw(3 * size, NaN);
    PackedVariables magnetic_conserved(magnetic_conserved_raw, nc[0], nc[1], nc[2]);
    PackedVariables magnetic_primitive(magnetic_primitive_raw, nc[0], nc[1], nc[2]);
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        magnetic_conserved.singleVariableView(axis).fill(8.0 + axis);
    }

    pbp.conservedToPrimitive(3,
                             3,
                             true,
                             plan,
                             scalar_solution,
                             conserved,
                             scalar_conserved,
                             magnetic_conserved,
                             primitive,
                             scalar_primitive,
                             magnetic_primitive);

    requireAll(primitive.singleVariableView(pbp.n_rho()), 1.0);
    requireAll(primitive.singleVariableView(pbp.n_energy()), (10.0 - 3.0) / 2.0);
    for (unsigned int scalar = 0; scalar < 2; scalar++)
    {
        requireAll(scalar_primitive.singleVariableView(scalar), (50.0 - (7.0 + scalar)) / 2.0);
    }
    const std::array<unsigned int, 3> velocity_indices = velocityIndices(pbp);
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        requireAll(primitive.singleVariableView(velocity_indices[axis]), (20.0 + 10.0 * axis - (4.0 + axis)) / 2.0);
        requireAll(magnetic_primitive.singleVariableView(axis), 8.0 + axis);
    }
}

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive 2D with two velocity components and no MHD leaves the rest untouched")
{
    const std::array<int, 3> nc    = {3, 3, 1};
    const size_t size              = static_cast<size_t>(nc[0]) * nc[1] * nc[2];
    GhostExchangePlan<double> plan = GhostExchangePlanTest::periodicPlan();
    std::vector<double> solution_raw(6 * size);
    std::vector<double> solution_dPd_raw(2 * size);
    std::vector<double> squared_sound_speed_raw(size);
    PackedVariablesWithGhosts solution(solution_raw, nc[0], nc[1], nc[2], plan);
    PackedVariables solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]);
    Array3D<double> squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_raw);
    PhysicsBasedPreconditioner pbp(solution, solution_dPd, squared_sound_speed);
    initializeVariables(solution, pbp, 2.0, 0.0, 3.0, {4.0, 5.0, 6.0});

    std::vector<double> conserved_raw(6 * size, 1.0);
    std::vector<double> primitive_raw(6 * size, NaN);
    PackedVariables conserved(conserved_raw, nc[0], nc[1], nc[2]);
    PackedVariables primitive(primitive_raw, nc[0], nc[1], nc[2]);
    initializeVariables(conserved, pbp, 1.0, 0.0, 10.0, {20.0, 30.0, 40.0});

    std::vector<double> magnetic_conserved_raw(3 * size);
    std::vector<double> magnetic_primitive_raw(3 * size, NaN);
    PackedVariables magnetic_conserved(magnetic_conserved_raw, nc[0], nc[1], nc[2]);
    PackedVariables magnetic_primitive(magnetic_primitive_raw, nc[0], nc[1], nc[2]);
    std::vector<double> no_variables_raw;
    PackedVariables no_variables(no_variables_raw, nc[0], nc[1], nc[2]);

    pbp.conservedToPrimitive(2,
                             2,
                             false,
                             plan,
                             no_variables,
                             conserved,
                             no_variables,
                             magnetic_conserved,
                             primitive,
                             no_variables,
                             magnetic_primitive);

    requireAll(primitive.singleVariableView(pbp.n_rho()), 1.0);
    requireAll(primitive.singleVariableView(pbp.n_energy()), (10.0 - 3.0) / 2.0);
    requireAll(primitive.singleVariableView(pbp.n_vel_i()), (20.0 - 4.0) / 2.0);
    requireAll(primitive.singleVariableView(pbp.n_vel_j()), (30.0 - 5.0) / 2.0);
    requireAll(primitive.singleVariableView(pbp.n_vel_k()), NaN);
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        requireAll(magnetic_primitive.singleVariableView(axis), NaN);
    }
}

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive converts each velocity component")
{
    const std::array<int, 3> nc    = {4, 3, 3};
    const size_t size              = static_cast<size_t>(nc[0]) * nc[1] * nc[2];
    GhostExchangePlan<double> plan = GhostExchangePlanTest::periodicPlan();
    std::vector<double> solution_raw(6 * size);
    std::vector<double> solution_dPd_raw(2 * size);
    std::vector<double> squared_sound_speed_raw(size);
    PackedVariablesWithGhosts solution(solution_raw, nc[0], nc[1], nc[2], plan);
    PackedVariables solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]);
    Array3D<double> squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_raw);
    PhysicsBasedPreconditioner pbp(solution, solution_dPd, squared_sound_speed);
    Array3DWithGhosts<double> solution_rho = solution.singleVariableView(pbp.n_rho());
    initializeVariables(solution, pbp, 2.0, 0.0, 3.0, {1.0, 2.0, 3.0});

    std::vector<double> conserved_raw(6 * size, 1.0);
    std::vector<double> primitive_raw(6 * size, NaN);
    PackedVariables conserved(conserved_raw, nc[0], nc[1], nc[2]);
    PackedVariables primitive(primitive_raw, nc[0], nc[1], nc[2]);
    initializeVariables(conserved, pbp, 1.0, 0.0, 10.0, {20.0, 30.0, 40.0});
    Array3D<double> conserved_rho = conserved.singleVariableView(pbp.n_rho());
    for (int k = 0; k < nc[2]; k++)
    {
        for (int j = 0; j < nc[1]; j++)
        {
            for (int i = 0; i < nc[0]; i++)
            {
                solution_rho(i, j, k)  = 2.0 + i + 2.0 * j + 3.0 * k;
                conserved_rho(i, j, k) = 1.0 + i + 2.0 * j + 3.0 * k;
            }
        }
    }
    std::vector<double> no_variables_raw;
    PackedVariables no_variables(no_variables_raw, nc[0], nc[1], nc[2]);
    pbp.conservedToPrimitive(
        3, 3, false, plan, no_variables, conserved, no_variables, no_variables, primitive, no_variables, no_variables);

    const std::array<unsigned int, 3> velocity_indices = velocityIndices(pbp);
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        Array3D<double> primitive_velocity = primitive.singleVariableView(velocity_indices[axis]);
        const int di                       = axis == 0 ? 1 : 0;
        const int dj                       = axis == 1 ? 1 : 0;
        const int dk                       = axis == 2 ? 1 : 0;
        for (int k = 0; k < nc[2]; k++)
        {
            for (int j = 0; j < nc[1]; j++)
            {
                for (int i = 0; i < nc[0]; i++)
                {
                    const int im                = (i - di + nc[0]) % nc[0];
                    const int jm                = (j - dj + nc[1]) % nc[1];
                    const int km                = (k - dk + nc[2]) % nc[2];
                    const double rho_face       = 0.5 * (solution_rho(i, j, k) + solution_rho(im, jm, km));
                    const double delta_rho_face = 0.5 * (conserved_rho(i, j, k) + conserved_rho(im, jm, km));
                    const double expected       = (20.0 + 10.0 * axis - (1.0 + axis) * delta_rho_face) / rho_face;
                    REQUIRE_THAT(primitive_velocity(i, j, k), Catch::Matchers::WithinRel(expected, 1e-14));
                }
            }
        }
    }
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
