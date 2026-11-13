#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "DirectionalArray.h"
#include "GhostExchangePlan.h"
#include "PackedVariables.h"
#include "PhysicsBasedPreconditioner.h"
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace
{
    // const double NaN = std::numeric_limits<double>::quiet_NaN();

    /**
     * @brief A field on the nodes [start, end) of the rank, stored from node 0 so that the view keeps the local
     * indices. Nodes outside [start, end) hold NaN. A field starting at node 0 is a whole array, so that it can be
     * wrapped in an Array3DWithGhosts.
     */
    // struct Field
    // {
    // Field(const std::array<int, 3> start, const std::array<int, 3> end, const double value = NaN) :
    // data(static_cast<size_t>(end[0]) * end[1] * end[2], NaN),
    // whole(std::make_unique<Array3D<double>>(end[0], end[1], end[2], data))
    // {
    // if (start == std::array<int, 3> {0, 0, 0}) { view = std::make_unique<Array3D<double>>(end[0], end[1], end[2], data); }
    // else
    // {
    // view = std::make_unique<Array3D<double>>(*whole, Slab(start, end));
    // }
    // forEach(*view, [&](int i, int j, int k) { (*view)(i, j, k) = value; });
    // }

    // template<typename Function>
    // static void forEach(const Array3D<double>& a, Function f)
    // {
    // const Slab s = a.cellSlab();
    // for (int k = s.start(2); k < s.end(2); k++)
    // for (int j = s.start(1); j < s.end(1); j++)
    // for (int i = s.start(0); i < s.end(0); i++)
    // f(i, j, k);
    // }

    // std::vector<double> data;
    // std::unique_ptr<Array3D<double>> whole, view;
    // };

    // /// @brief Every value of `a` equals `expected` (or is NaN when `expected` is NaN, i.e. not written)
    // void requireAll(const Array3D<double>& a, const double expected)
    // {
    // const Slab s = a.cellSlab();
    // for (int k = s.start(2); k < s.end(2); k++)
    // {
    // for (int j = s.start(1); j < s.end(1); j++)
    // {
    // for (int i = s.start(0); i < s.end(0); i++)
    // {
    // if (std::isnan(expected))
    // REQUIRE(std::isnan(a(i, j, k)));
    // else
    // REQUIRE_THAT(a(i, j, k), Catch::Matchers::WithinRel(expected, 1e-14));
    // }
    // }
    // }
    // }

    /// Ghost exchange plan of a periodic domain held by this rank alone, which is then its own neighbour
    GhostExchangePlan<double> periodicPlan()
    {
        DirectionalArray<BoundaryConditions> bcs;
        bcs.fill(BoundaryConditions::PERIODIC);
        DirectionalArray<int> neighbours;
        neighbours.fill(0);
        return GhostExchangePlan<double>(bcs, neighbours, MPI_COMM_SELF);
    }

    /// A PhysicsBasedPreconditioner on nc cells whose stored solution holds the given rho and e
    // struct Preconditioner
    // {
    // Preconditioner(const std::array<int, 3> nc,
    // const std::function<double(int, int, int)>& rho,
    // const std::function<double(int, int, int)>& e) :
    // size(static_cast<size_t>(nc[0]) * nc[1] * nc[2]),
    // solution_raw(3 * size),
    // solution_dPd_raw(2 * size),
    // squared_sound_speed_raw(size),
    // solution(solution_raw, nc[0], nc[1], nc[2]),
    // solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]),
    // squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_raw),
    // pbp(solution, solution_dPd, squared_sound_speed)
    // {
    // Array3D<double> sol_rho = solution.singleVariableView(pbp.n_rho());
    // Array3D<double> sol_e   = solution.singleVariableView(pbp.n_energy());

    // Field::forEach(sol_rho,
    // [&](int i, int j, int k)
    // {
    // sol_rho(i, j, k) = rho(i, j, k);
    // sol_e(i, j, k)   = e(i, j, k);
    // });
    // }

    // size_t size;
    // std::vector<double> solution_raw, solution_dPd_raw, squared_sound_speed_raw;
    // PackedVariables solution, solution_dPd;
    // Array3D<double> squared_sound_speed;
    // PhysicsBasedPreconditioner pbp;
    // };

    /**
     * @brief Constant fields on nc periodic cells, so the expected result is a few numbers. The x velocity and B can be
     * given other faces than the cells. The indices and ghosts are checked against MUSIC further down.
     */
    // struct ConstantProblem
    // {
    // ConstantProblem(const unsigned int ndim_,
    // const unsigned int nmom_,
    // const std::array<int, 3> nc,
    // const unsigned int nscalars,
    // const std::array<int, 3> x_faces_start = {0, 0, 0},
    // std::array<int, 3> x_faces_end         = {0, 0, 0}) :
    // ndim(ndim_),
    // nmom(nmom_),
    // plan(periodicPlan()),
    // preconditioner(
    // nc,
    // [](int, int, int) { return 2.0; },
    // [](int, int, int) { return 3.0; }),
    // cv_rho({0, 0, 0}, nc, 1.0),
    // cv_e({0, 0, 0}, nc, 10.0),
    // pv_rho({0, 0, 0}, nc),
    // pv_e({0, 0, 0}, nc)
    // {
    // if (x_faces_end == std::array<int, 3> {0, 0, 0}) { x_faces_end = nc; }
    // for (unsigned int ax = 0; ax < 3; ax++)
    // {
    // const std::array<int, 3> start = (ax == 0) ? x_faces_start : std::array<int, 3> {0, 0, 0};
    // const std::array<int, 3> end   = (ax == 0) ? x_faces_end : nc;
    // sol_vel.push_back(std::make_unique<Field>(start, end, 4.0 + ax));
    // cv_vel.push_back(std::make_unique<Field>(start, end, 20.0 + 10.0 * ax));
    // pv_vel.push_back(std::make_unique<Field>(start, end));
    // cv_b.push_back(std::make_unique<Field>(start, end, 8.0 + ax));
    // pv_b.push_back(std::make_unique<Field>(start, end));
    // }
    // for (unsigned int isc = 0; isc < nscalars; isc++)
    // {
    // sol_phi.push_back(std::make_unique<Field>(std::array<int, 3> {0, 0, 0}, nc, 7.0 + isc));
    // cv_phi.push_back(std::make_unique<Field>(std::array<int, 3> {0, 0, 0}, nc, 50.0));
    // pv_phi.push_back(std::make_unique<Field>(std::array<int, 3> {0, 0, 0}, nc));
    // }
    // }

    // template<typename T>
    // static std::array<T*, 3> three(const std::vector<std::unique_ptr<Field>>& fields)
    // { return {fields[0]->view.get(), fields[1]->view.get(), fields[2]->view.get()}; }

    // template<typename T>
    // static std::vector<T*> all(const std::vector<std::unique_ptr<Field>>& fields)
    // {
    // std::vector<T*> result;
    // for (const auto& field : fields)
    // {
    // result.push_back(field->view.get());
    // }
    // return result;
    // }

    // /// Run conservedToPrimitive, optionally with another delta_primitive e or without the primitive deltas of the active scalars
    // void run(const bool mhd, Array3D<double>* primitive_e = nullptr, const bool with_primitive_scalars = true)
    // {
    // preconditioner.pbp.conservedToPrimitive(ndim,
    // nmom,
    // mhd,
    // plan,
    // three<const Array3D<double>>(sol_vel),
    // all<const Array3D<double>>(sol_phi),
    // *cv_rho.view,
    // *cv_e.view,
    // three<const Array3D<double>>(cv_vel),
    // all<const Array3D<double>>(cv_phi),
    // three<const Array3D<double>>(cv_b),
    // *pv_rho.view,
    // primitive_e ? *primitive_e : *pv_e.view,
    // three<Array3D<double>>(pv_vel),
    // with_primitive_scalars ? all<Array3D<double>>(pv_phi) :
    // std::vector<Array3D<double>*> {},
    // three<Array3D<double>>(pv_b));
    // }

    // unsigned int ndim, nmom;
    // GhostExchangePlan<double> plan;
    // Preconditioner preconditioner;
    // Field cv_rho, cv_e, pv_rho, pv_e;
    // std::vector<std::unique_ptr<Field>> sol_vel, cv_vel, pv_vel, cv_b, pv_b, sol_phi, cv_phi, pv_phi;
    // };
}  // namespace

// TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive rejects invalid input")
// {
// const std::array<int, 3> nc = {3, 3, 3};
// for (const auto& [ndim, nmom] : std::vector<std::pair<unsigned int, unsigned int>> {
// {1, 3},
// {4, 3},
// {3, 1},
// {3, 4},
// {3, 2}
// })
// {
// ConstantProblem p(ndim, nmom, nc, 0);
// REQUIRE_THROWS_AS(p.run(true), std::invalid_argument);
// }

// SECTION("delta_primitive e on other nodes than delta_conserved rho")
// {
// ConstantProblem p(3, 3, nc, 0);
// Field smaller({0, 0, 0}, {2, 3, 3});
// REQUIRE_THROWS_AS(p.run(true, smaller.view.get()), std::invalid_argument);
// }

// SECTION("active scalars missing in delta_primitive")
// {
// ConstantProblem p(3, 3, nc, 1);
// REQUIRE_THROWS_AS(p.run(true, nullptr, false), std::invalid_argument);
// }
// }

// TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive 3D with active scalars and MHD")
// {
// ConstantProblem p(3, 3, {3, 4, 5}, 2);
// p.run(true);

// requireAll(*p.pv_rho.view, 1.0);
// requireAll(*p.pv_e.view, (10.0 - 3.0 * 1.0) / 2.0);
// for (unsigned int isc = 0; isc < 2; isc++)
// {
// requireAll(*p.pv_phi[isc]->view, (50.0 - (7.0 + isc) * 1.0) / 2.0);
// }
// for (unsigned int ax = 0; ax < 3; ax++)
// {
// CAPTURE(ax);
// requireAll(*p.pv_vel[ax]->view, (20.0 + 10.0 * ax - (4.0 + ax) * 1.0) / 2.0);
// requireAll(*p.pv_b[ax]->view, 8.0 + ax);
// }
// }

// TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive 2D with two velocity components and no MHD leaves the rest untouched")
// {
// ConstantProblem p(2, 2, {3, 3, 1}, 0);
// p.run(false);

// requireAll(*p.pv_vel[0]->view, (20.0 - 4.0) / 2.0);
// requireAll(*p.pv_vel[1]->view, (30.0 - 5.0) / 2.0);
// requireAll(*p.pv_vel[2]->view, NaN);
// for (unsigned int ax = 0; ax < 3; ax++)
// {
// requireAll(*p.pv_b[ax]->view, NaN);
// }
// }

// TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive converts each velocity over its own solved faces")
// {
// // Along x the negative boundary face is not solved (faces start at 1) and the positive one is (faces end at n + 1),
// // as MUSIC `solved_local_domain_nodes` gives on a rank touching both boundaries
// const std::array<int, 3> nc = {4, 3, 3};
// ConstantProblem p(3, 3, nc, 0, {1, 0, 0}, {nc[0] + 1, nc[1], nc[2]});
// p.run(true);

// requireAll(*p.pv_vel[0]->view, (20.0 - 4.0) / 2.0);
// requireAll(*p.pv_b[0]->view, 8.0);
// for (int k = 0; k < nc[2]; k++)
// {
// for (int j = 0; j < nc[1]; j++)
// {
// REQUIRE(std::isnan((*p.pv_vel[0]->whole)(0, j, k)));
// REQUIRE(std::isnan((*p.pv_b[0]->whole)(0, j, k)));
// }
// }
// }

namespace
{
    /**
     * @brief Run PhysicsBasedPreconditioner::conservedToPrimitive on the MUSIC data of one rank (rho, e and the velocity components, written by the MUSIC diff
     * given in the pull request; MUSIC indices start at 1) and compare with the MUSIC outputs. Also check that the
     * exchange of delta rho on the plan gives the halo MUSIC read.
     */
    void checkAgainstMusic(const std::string& file_prefix, GhostExchangePlan<double>& plan)
    {
        FILE* file = fopen((file_prefix + "_inputs.dat").c_str(), "r");
        REQUIRE(file != nullptr);

        // Static definition of the Roberts Dynamo problem
        unsigned int ndim = 3;
        unsigned int nmom = 3;
        bool use_mhd      = false;
        int ni, nj, nk;
        const DirectionalArray<BoundaryConditions> global_bcs({
            BoundaryConditions::PERIODIC,
            BoundaryConditions::PERIODIC,
            BoundaryConditions::PERIODIC,
            BoundaryConditions::PERIODIC,
            BoundaryConditions::PERIODIC,
            BoundaryConditions::PERIODIC,
        });

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
        pbp.conservedToPrimitive(ndim, nmom, use_mhd, plan, {}, conserved_variables, {}, {}, actual_primitive_variables, {}, {});

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

TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive matches MUSIC cons_to_prim on the Roberts dynamo test on 1 rank")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 1) { SKIP("Test requires exactly 1 rank"); }

    GhostExchangePlan<double> plan = periodicPlan();
    checkAgainstMusic(std::string(TEST_DATA_DIR) + "/cons_to_prim_roberts_dynamo_1rank_rank0", plan);
}

// TEST_CASE("PhysicsBasedPreconditioner::conservedToPrimitive matches MUSIC cons_to_prim on the Roberts dynamo test on 4 ranks")
// {
// int num_ranks;
// MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
// if (num_ranks != 4) { SKIP("The 4 rank MUSIC data needs exactly 4 ranks"); }

// // MUSIC splits the periodic 16^3 domain 2 x 2 x 1 with rank = 2 * (x block) + (y block), which is the order of a
// // cartesian communicator of dims {2, 2, 1} without reordering
// std::array<int, 3> dims    = {2, 2, 1};
// std::array<int, 3> periods = {1, 1, 1};
// MPI_Comm cartesian;
// MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods.data(), false, &cartesian);
// int rank;
// MPI_Comm_rank(cartesian, &rank);
// DirectionalArray<int> neighbours;
// for (int axis = 0; axis < 3; axis++)
// {
// MPI_Cart_shift(
// cartesian, axis, 1, &neighbours.get(axis, Direction::NEGATIVE), &neighbours.get(axis, Direction::POSITIVE));
// }
// DirectionalArray<BoundaryConditions> inner;
// inner.fill(BoundaryConditions::INNER);
// GhostExchangePlan<double> plan(inner, neighbours, cartesian);
// checkAgainstMusic(std::string(TEST_DATA_DIR) + "/cons_to_prim_roberts_dynamo_4ranks_rank" + std::to_string(rank), plan);
// MPI_Comm_free(&cartesian);
// }
