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
#include <limits>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    /**
     * @brief A field on the nodes [start, end) of the rank, stored from node 0 so that the view keeps the local
     * indices. Nodes outside [start, end) hold NaN. A field starting at node 0 is a whole array, so that it can be
     * wrapped in an Array3DWithGhosts.
     */
    struct Field
    {
        Field(const std::array<int, 3> start, const std::array<int, 3> end, const double value = NaN) :
            data(static_cast<size_t>(end[0]) * end[1] * end[2], NaN),
            whole(std::make_unique<Array3D<double>>(end[0], end[1], end[2], data))
        {
            if (start == std::array<int, 3> {0, 0, 0}) { view = std::make_unique<Array3D<double>>(end[0], end[1], end[2], data); }
            else
            {
                view = std::make_unique<Array3D<double>>(*whole, Slab(start, end));
            }
            forEach(*view, [&](int i, int j, int k) { (*view)(i, j, k) = value; });
        }

        template<typename Function>
        static void forEach(const Array3D<double>& a, Function f)
        {
            const Slab s = a.cellSlab();
            for (int k = s.start(2); k < s.end(2); k++)
                for (int j = s.start(1); j < s.end(1); j++)
                    for (int i = s.start(0); i < s.end(0); i++)
                        f(i, j, k);
        }

        std::vector<double> data;
        std::unique_ptr<Array3D<double>> whole, view;
    };

    /// Every value of `a` equals `expected` (or is NaN when `expected` is NaN, i.e. not written)
    void requireAll(const Array3D<double>& a, const double expected)
    {
        Field::forEach(a,
                       [&](int i, int j, int k)
                       {
                           CAPTURE(i, j, k);
                           if (std::isnan(expected)) { REQUIRE(std::isnan(a(i, j, k))); }
                           else
                           {
                               REQUIRE_THAT(a(i, j, k), Catch::Matchers::WithinRel(expected, 1e-14));
                           }
                       });
    }

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
    struct Preconditioner
    {
        Preconditioner(const std::array<int, 3> nc,
                       const std::function<double(int, int, int)>& rho,
                       const std::function<double(int, int, int)>& e) :
            size(static_cast<size_t>(nc[0]) * nc[1] * nc[2]),
            solution_raw(3 * size),
            solution_dPd_raw(2 * size),
            squared_sound_speed_raw(size),
            solution(solution_raw, nc[0], nc[1], nc[2], periodicPlan()),
            solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]),
            squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_raw),
            pbp(solution, solution_dPd, squared_sound_speed)
        {
            Array3D<double> sol_rho = solution.singleVariableView(pbp.n_rho());
            Array3D<double> sol_e   = solution.singleVariableView(pbp.n_energy());
            Field::forEach(sol_rho,
                           [&](int i, int j, int k)
                           {
                               sol_rho(i, j, k) = rho(i, j, k);
                               sol_e(i, j, k)   = e(i, j, k);
                           });
        }

        size_t size;
        std::vector<double> solution_raw, solution_dPd_raw, squared_sound_speed_raw;
        PackedVariablesWithGhosts solution;
        PackedVariables solution_dPd;
        Array3D<double> squared_sound_speed;
        PhysicsBasedPreconditioner pbp;
    };

    /**
     * @brief Constant fields on nc periodic cells, so the expected result is a few numbers. The x velocity and B can be
     * given other faces than the cells. The indices and ghosts are checked against MUSIC further down.
     */
    struct ConstantProblem
    {
        ConstantProblem(const unsigned int ndim_,
                        const unsigned int nmom_,
                        const std::array<int, 3> nc,
                        const unsigned int nscalars,
                        const std::array<int, 3> x_faces_start = {0, 0, 0},
                        std::array<int, 3> x_faces_end         = {0, 0, 0}) :
            ndim(ndim_),
            nmom(nmom_),
            plan(periodicPlan()),
            preconditioner(
                nc,
                [](int, int, int) { return 2.0; },
                [](int, int, int) { return 3.0; }),
            cv_rho({0, 0, 0}, nc, 1.0),
            cv_e({0, 0, 0}, nc, 10.0),
            pv_rho({0, 0, 0}, nc),
            pv_e({0, 0, 0}, nc)
        {
            if (x_faces_end == std::array<int, 3> {0, 0, 0}) { x_faces_end = nc; }
            for (unsigned int ax = 0; ax < 3; ax++)
            {
                const std::array<int, 3> start = (ax == 0) ? x_faces_start : std::array<int, 3> {0, 0, 0};
                const std::array<int, 3> end   = (ax == 0) ? x_faces_end : nc;
                sol_vel.push_back(std::make_unique<Field>(start, end, 4.0 + ax));
                cv_vel.push_back(std::make_unique<Field>(start, end, 20.0 + 10.0 * ax));
                pv_vel.push_back(std::make_unique<Field>(start, end));
                cv_b.push_back(std::make_unique<Field>(start, end, 8.0 + ax));
                pv_b.push_back(std::make_unique<Field>(start, end));
            }
            for (unsigned int isc = 0; isc < nscalars; isc++)
            {
                sol_phi.push_back(std::make_unique<Field>(std::array<int, 3> {0, 0, 0}, nc, 7.0 + isc));
                cv_phi.push_back(std::make_unique<Field>(std::array<int, 3> {0, 0, 0}, nc, 50.0));
                pv_phi.push_back(std::make_unique<Field>(std::array<int, 3> {0, 0, 0}, nc));
            }
        }

        template<typename T>
        static std::array<T*, 3> three(const std::vector<std::unique_ptr<Field>>& fields)
        { return {fields[0]->view.get(), fields[1]->view.get(), fields[2]->view.get()}; }

        template<typename T>
        static std::vector<T*> all(const std::vector<std::unique_ptr<Field>>& fields)
        {
            std::vector<T*> result;
            for (const auto& field : fields)
            {
                result.push_back(field->view.get());
            }
            return result;
        }

        /// Run consToPrim, optionally with another delta_prim e or without the primitive deltas of the active scalars
        void run(const bool mhd, Array3D<double>* prim_e = nullptr, const bool with_prim_scalars = true)
        {
            preconditioner.pbp.consToPrim(ndim,
                                          nmom,
                                          mhd,
                                          plan,
                                          three<const Array3D<double>>(sol_vel),
                                          all<const Array3D<double>>(sol_phi),
                                          *cv_rho.view,
                                          *cv_e.view,
                                          three<const Array3D<double>>(cv_vel),
                                          all<const Array3D<double>>(cv_phi),
                                          three<const Array3D<double>>(cv_b),
                                          *pv_rho.view,
                                          prim_e ? *prim_e : *pv_e.view,
                                          three<Array3D<double>>(pv_vel),
                                          with_prim_scalars ? all<Array3D<double>>(pv_phi) : std::vector<Array3D<double>*> {},
                                          three<Array3D<double>>(pv_b));
        }

        unsigned int ndim, nmom;
        GhostExchangePlan<double> plan;
        Preconditioner preconditioner;
        Field cv_rho, cv_e, pv_rho, pv_e;
        std::vector<std::unique_ptr<Field>> sol_vel, cv_vel, pv_vel, cv_b, pv_b, sol_phi, cv_phi, pv_phi;
    };
}  // namespace

TEST_CASE("PhysicsBasedPreconditioner::consToPrim rejects invalid input")
{
    const std::array<int, 3> nc = {3, 3, 3};
    for (const auto& [ndim, nmom] : std::vector<std::pair<unsigned int, unsigned int>> {
             {1, 3},
             {4, 3},
             {3, 1},
             {3, 4},
             {3, 2}
    })
    {
        CAPTURE(ndim, nmom);
        ConstantProblem p(ndim, nmom, nc, 0);
        REQUIRE_THROWS_AS(p.run(true), std::invalid_argument);
    }

    SECTION("delta_prim e on other nodes than delta_cons rho")
    {
        ConstantProblem p(3, 3, nc, 0);
        Field smaller({0, 0, 0}, {2, 3, 3});
        REQUIRE_THROWS_AS(p.run(true, smaller.view.get()), std::invalid_argument);
    }

    SECTION("active scalars missing in delta_prim")
    {
        ConstantProblem p(3, 3, nc, 1);
        REQUIRE_THROWS_AS(p.run(true, nullptr, false), std::invalid_argument);
    }
}

TEST_CASE("PhysicsBasedPreconditioner::consToPrim 3D with active scalars and MHD")
{
    ConstantProblem p(3, 3, {3, 4, 5}, 2);
    p.run(true);

    requireAll(*p.pv_rho.view, 1.0);
    requireAll(*p.pv_e.view, (10.0 - 3.0 * 1.0) / 2.0);
    for (unsigned int isc = 0; isc < 2; isc++)
    {
        requireAll(*p.pv_phi[isc]->view, (50.0 - (7.0 + isc) * 1.0) / 2.0);
    }
    for (unsigned int ax = 0; ax < 3; ax++)
    {
        CAPTURE(ax);
        requireAll(*p.pv_vel[ax]->view, (20.0 + 10.0 * ax - (4.0 + ax) * 1.0) / 2.0);
        requireAll(*p.pv_b[ax]->view, 8.0 + ax);
    }
}

TEST_CASE("PhysicsBasedPreconditioner::consToPrim 2D with two velocity components and no MHD leaves the rest untouched")
{
    ConstantProblem p(2, 2, {3, 3, 1}, 0);
    p.run(false);

    requireAll(*p.pv_vel[0]->view, (20.0 - 4.0) / 2.0);
    requireAll(*p.pv_vel[1]->view, (30.0 - 5.0) / 2.0);
    requireAll(*p.pv_vel[2]->view, NaN);
    for (unsigned int ax = 0; ax < 3; ax++)
    {
        requireAll(*p.pv_b[ax]->view, NaN);
    }
}

TEST_CASE("PhysicsBasedPreconditioner::consToPrim converts each velocity over its own solved faces")
{
    // Along x the negative boundary face is not solved (faces start at 1) and the positive one is (faces end at n + 1),
    // as MUSIC `solved_local_domain_nodes` gives on a rank touching both boundaries
    const std::array<int, 3> nc = {4, 3, 3};
    ConstantProblem p(3, 3, nc, 0, {1, 0, 0}, {nc[0] + 1, nc[1], nc[2]});
    p.run(true);

    requireAll(*p.pv_vel[0]->view, (20.0 - 4.0) / 2.0);
    requireAll(*p.pv_b[0]->view, 8.0);
    for (int k = 0; k < nc[2]; k++)
    {
        for (int j = 0; j < nc[1]; j++)
        {
            REQUIRE(std::isnan((*p.pv_vel[0]->whole)(0, j, k)));
            REQUIRE(std::isnan((*p.pv_b[0]->whole)(0, j, k)));
        }
    }
}

namespace
{
    /**
     * @brief Run PhysicsBasedPreconditioner::consToPrim on the MUSIC data of one rank (rho, e and the velocity components, written by the MUSIC diff
     * given in the pull request; MUSIC indices start at 1) and compare with the MUSIC outputs. Also check that the
     * exchange of delta rho on the plan gives the halo MUSIC read.
     */
    void checkAgainstMusic(const std::string& prefix, GhostExchangePlan<double>& plan)
    {
        FILE* file = fopen((prefix + "_inputs.dat").c_str(), "r");
        REQUIRE(file != nullptr);
        unsigned int ndim, nmom;
        int mhd, nscalars;
        std::array<int, 3> nc;
        REQUIRE(fscanf(file, " ndim_nmom_mhd %u %u %d", &ndim, &nmom, &mhd) == 3);
        REQUIRE(fscanf(file, " active_scalars %d", &nscalars) == 1);
        REQUIRE(nscalars == 0);
        REQUIRE(fscanf(file, " local_cells %d %d %d", &nc[0], &nc[1], &nc[2]) == 3);
        const unsigned int nvar = 2 + nmom;
        std::vector<std::array<std::array<int, 3>, 2>> slabs(nvar);  // local indices
        for (auto& [start, end] : slabs)
        {
            char name[16];
            REQUIRE(
                fscanf(file, " slab %15s %d %d %d %d %d %d", name, &start[0], &start[1], &start[2], &end[0], &end[1], &end[2]) ==
                7);
            for (int& s : start)
            {
                s--;
            }
        }

        // The solution with MUSIC's ghost layer (along k only in 3D), five values per line, i fastest. Each field is
        // kept without the ghost layer on the nodes of its delta (a velocity component beyond nmom on the cells).
        const int kg = (ndim == 3) ? 1 : 0, gi = nc[0] + 2, gj = nc[1] + 2;
        std::vector<std::array<double, 5>> values(static_cast<size_t>(gi) * gj * (nc[2] + 2 * kg));
        for (auto& v : values)
        {
            REQUIRE(fscanf(file, " %lf %lf %lf %lf %lf", &v[0], &v[1], &v[2], &v[3], &v[4]) == 5);
        }
        const auto music = [&](unsigned int v)
        { return [&, v](int i, int j, int k) { return values[(i + 1) + gi * ((j + 1) + gj * (k + kg))][v]; }; };
        Preconditioner preconditioner(nc, music(0), music(1));
        std::vector<std::unique_ptr<Field>> vel, cons, expected, prim;
        for (unsigned int v = 2; v < 5; v++)
        {
            vel.push_back(v < nvar ? std::make_unique<Field>(slabs[v][0], slabs[v][1]) :
                                     std::make_unique<Field>(std::array<int, 3> {0, 0, 0}, nc));
            Array3D<double>& s = *vel.back()->view;
            Field::forEach(s, [&](int i, int j, int k) { s(i, j, k) = music(v)(i, j, k); });
        }

        // delta_cons over the slab of each variable, i fastest
        for (unsigned int v = 0; v < nvar; v++)
        {
            cons.push_back(std::make_unique<Field>(slabs[v][0], slabs[v][1]));
            Array3D<double>& c = *cons[v]->view;
            Field::forEach(c, [&](int i, int j, int k) { REQUIRE(fscanf(file, " %lf", &c(i, j, k)) == 1); });
        }

        // The delta rho halo MUSIC read outside the rank equals the ghosts after the exchange on the plan
        Array3DWithGhosts<double> delta_rho_g(*cons[0]->view, plan);
        delta_rho_g.exchangeGhosts();
        for (unsigned int ax = 0; ax < ndim; ax++)
        {
            unsigned int axis, count;
            REQUIRE(fscanf(file, " halo %u %u", &axis, &count) == 2);
            REQUIRE(axis == ax + 1);
            REQUIRE(count > 0);
            for (unsigned int n = 0; n < count; n++)
            {
                int i, j, k;
                double value;
                REQUIRE(fscanf(file, " %d %d %d %lf", &i, &j, &k, &value) == 4);
                CAPTURE(i, j, k);
                REQUIRE(delta_rho_g(i - 1, j - 1, k - 1) == value);
            }
        }
        fclose(file);

        // Expected delta_prim over the slab of each variable, i fastest
        file = fopen((prefix + "_expected_outputs.dat").c_str(), "r");
        REQUIRE(file != nullptr);
        for (unsigned int v = 0; v < nvar; v++)
        {
            expected.push_back(std::make_unique<Field>(slabs[v][0], slabs[v][1]));
            prim.push_back(std::make_unique<Field>(slabs[v][0], slabs[v][1]));
            Array3D<double>& e = *expected[v]->view;
            Field::forEach(e, [&](int i, int j, int k) { REQUIRE(fscanf(file, " %lf", &e(i, j, k)) == 1); });
        }
        fclose(file);

        // The magnetic field is a copy and is not in the data; it is covered by the cases above
        preconditioner.pbp.consToPrim(ndim,
                                      nmom,
                                      false,
                                      plan,
                                      {vel[0]->view.get(), vel[1]->view.get(), vel[2]->view.get()},
                                      {},
                                      *cons[0]->view,
                                      *cons[1]->view,
                                      {cons[2]->view.get(), cons[3]->view.get(), cons[4]->view.get()},
                                      {},
                                      {},
                                      *prim[0]->view,
                                      *prim[1]->view,
                                      {prim[2]->view.get(), prim[3]->view.get(), prim[4]->view.get()},
                                      {},
                                      {});

        // The velocity terms v delta_rho_f and delta_rho_v nearly cancel at some faces, so the round-off of the two
        // compilers is compared with the size of each field rather than with each value
        for (unsigned int v = 0; v < nvar; v++)
        {
            CAPTURE(v);
            const Array3D<double>& e = *expected[v]->view;
            double scale             = 0;
            Field::forEach(e, [&](int i, int j, int k) { scale = std::max(scale, std::abs(e(i, j, k))); });
            REQUIRE(scale > 0);
            Field::forEach(e,
                           [&](int i, int j, int k)
                           {
                               CAPTURE(i, j, k);
                               REQUIRE_THAT((*prim[v]->view)(i, j, k), Catch::Matchers::WithinAbs(e(i, j, k), 1e-14 * scale));
                           });
        }
    }
}  // namespace

TEST_CASE("PhysicsBasedPreconditioner::consToPrim matches MUSIC cons_to_prim on the Roberts dynamo test on 1 rank")
{
    GhostExchangePlan<double> plan = periodicPlan();
    checkAgainstMusic(std::string(TEST_DATA_DIR) + "/cons_to_prim_roberts_dynamo_1rank_rank0", plan);
}

TEST_CASE("PhysicsBasedPreconditioner::consToPrim matches MUSIC cons_to_prim on the Roberts dynamo test on 4 ranks")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 4) { SKIP("The 4 rank MUSIC data needs exactly 4 ranks"); }

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
    checkAgainstMusic(std::string(TEST_DATA_DIR) + "/cons_to_prim_roberts_dynamo_4ranks_rank" + std::to_string(rank), plan);
    MPI_Comm_free(&cartesian);
}
