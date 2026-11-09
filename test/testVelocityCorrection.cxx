#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "GhostExchangePlan.h"
#include "VelocityCorrection.h"
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mpi.h>
#include <string>
#include <tuple>
#include <vector>

namespace
{
    using Function       = std::function<double(int, int, int)>;
    using FaceConditions = std::array<BoundaryConditions, 6>;

    const double NaN = std::numeric_limits<double>::quiet_NaN();

    const FaceConditions ALL_PERIODIC = {BoundaryConditions::PERIODIC,
                                         BoundaryConditions::PERIODIC,
                                         BoundaryConditions::PERIODIC,
                                         BoundaryConditions::PERIODIC,
                                         BoundaryConditions::PERIODIC,
                                         BoundaryConditions::PERIODIC};

    /**
     * @brief A plan for a single rank on MPI_COMM_SELF: the rank is its own neighbour across a periodic face, and has
     * no neighbour across any other face.
     */
    GhostExchangePlan<double> singleRankPlan(const FaceConditions& bcs)
    {
        std::array<int, 6> neighbours;
        for (unsigned int face = 0; face < 6; face++)
        {
            neighbours[face] = (bcs[face] == BoundaryConditions::PERIODIC) ? 0 : MPI_PROC_NULL;
        }
        return GhostExchangePlan<double>(bcs, neighbours, MPI_COMM_SELF);
    }

    /**
     * @brief A solution field with one ghost layer (none along k in 2D), filled from f at the local positions.
     */
    struct GhostedField
    {
        GhostedField(const std::array<int, 3> nc, const int kg, const Function& f) :
            data(static_cast<size_t>(nc[0] + 2) * (nc[1] + 2) * (nc[2] + 2 * kg)),
            view(nc[0] + 2, nc[1] + 2, nc[2] + 2 * kg, data)
        {
            for (int k = -kg; k < nc[2] + kg; k++)
                for (int j = -1; j < nc[1] + 1; j++)
                    for (int i = -1; i < nc[0] + 1; i++)
                        view(i + 1, j + 1, k + kg) = f(i, j, k);
        }

        std::vector<double> data;
        Array3D<double> view;
    };

    /**
     * @brief A field on the nodes [start, end) of the rank, stored from node 0 so that the view keeps the local indices.
     * Nodes outside [start, end) hold NaN. A field starting at node 0 is a whole array rather than a sub array, so that
     * it can be wrapped in an Array3DWithGhosts.
     */
    struct NodeField
    {
        NodeField(const std::array<int, 3> start, const std::array<int, 3> end, const Function& f = nullptr) :
            data(static_cast<size_t>(end[0]) * end[1] * end[2], NaN),
            parent(std::make_unique<Array3D<double>>(end[0], end[1], end[2], data)),
            view(start == std::array<int, 3> {0, 0, 0} ? std::make_unique<Array3D<double>>(end[0], end[1], end[2], data) :
                                                         std::make_unique<Array3D<double>>(*parent, Slab(start, end)))
        {
            if (f)
            {
                for (int k = start[2]; k < end[2]; k++)
                    for (int j = start[1]; j < end[1]; j++)
                        for (int i = start[0]; i < end[0]; i++)
                            (*view)(i, j, k) = f(i, j, k);
            }
        }

        std::vector<double> data;
        std::unique_ptr<Array3D<double>> parent;
        std::unique_ptr<Array3D<double>> view;
    };
}  // namespace

TEST_CASE("VelocityCorrection subtracts dt grad(delta P) / rho_f on every solved face")
{
    const double dt = 0.25, span_x = 0.5, span_y = 2.0, span_z = 4.0;
    // Linear pressure delta along each axis, so the gradient is the same on every face, also across the periodic wrap
    // of the ghosts (the wrap is not linear, so the faces next to the wrap are checked against the wrapped value)
    const std::array<int, 3> nc = {4, 3, 2};
    const Function pressure     = [](int i, int j, int k) { return 3.0 * i - 2.0 * j + 5.0 * k; };
    const Function density      = [](int i, int j, int k) { return 1.0 + 0.1 * i + 0.2 * j + 0.3 * k; };
    const Function initial      = [](int i, int j, int k) { return 0.01 * (i + 10 * j + 100 * k); };

    for (const unsigned int ndim : {2U, 3U})
    {
        CAPTURE(ndim);
        const int kg                   = (ndim == 3) ? 1 : 0;
        const std::array<int, 3> cells = {nc[0], nc[1], (ndim == 3) ? nc[2] : 1};
        GhostExchangePlan<double> plan = singleRankPlan(ALL_PERIODIC);
        const auto wrapped             = [&](int i, int j, int k)
        { return pressure((i + cells[0]) % cells[0], (j + cells[1]) % cells[1], (k + cells[2]) % cells[2]); };
        GhostedField rho(cells, kg, [&](int i, int j, int k) { return density(i, j, k); });
        NodeField delta_pressure({0, 0, 0}, cells, pressure);
        const std::array<double, 3> spans = {span_x, span_y, span_z};
        std::array<std::unique_ptr<NodeField>, 3> span, delta_v;
        for (unsigned int ax = 0; ax < 3; ax++)
        {
            span[ax] =
                std::make_unique<NodeField>(std::array<int, 3> {0, 0, 0}, cells, [&, ax](int, int, int) { return spans[ax]; });
            delta_v[ax] = std::make_unique<NodeField>(std::array<int, 3> {0, 0, 0}, cells, initial);
        }

        VelocityCorrection::correctVelocity(ndim,
                                            dt,
                                            rho.view,
                                            plan,
                                            *delta_pressure.view,
                                            {span[0]->view.get(), span[1]->view.get(), span[2]->view.get()},
                                            {delta_v[0]->view.get(), delta_v[1]->view.get(), delta_v[2]->view.get()});

        for (unsigned int ax = 0; ax < 3; ax++)
        {
            CAPTURE(ax);
            const int di = (ax == 0) ? 1 : 0, dj = (ax == 1) ? 1 : 0, dk = (ax == 2) ? 1 : 0;
            for (int k = 0; k < cells[2]; k++)
            {
                for (int j = 0; j < cells[1]; j++)
                {
                    for (int i = 0; i < cells[0]; i++)
                    {
                        CAPTURE(i, j, k);
                        double expected = initial(i, j, k);
                        if (ax < ndim)
                        {
                            const double grad  = (wrapped(i, j, k) - wrapped(i - di, j - dj, k - dk)) / spans[ax];
                            const double rho_f = 0.5 * (density(i - di, j - dj, k - dk) + density(i, j, k));
                            expected += -dt * grad / rho_f;
                        }
                        REQUIRE(std::abs((*delta_v[ax]->view)(i, j, k) - expected) <= 1e-15 * (1.0 + std::abs(expected)));
                    }
                }
            }
        }
    }
}

TEST_CASE("VelocityCorrection sets the velocity delta to 0 on a solved negative domain boundary face only")
{
    // Walls along x: no neighbour on either x face. Periodic along y and z.
    const FaceConditions walls_x   = {BoundaryConditions::ZERO_DERIVATIVE,
                                      BoundaryConditions::ZERO_DERIVATIVE,
                                      BoundaryConditions::PERIODIC,
                                      BoundaryConditions::PERIODIC,
                                      BoundaryConditions::PERIODIC,
                                      BoundaryConditions::PERIODIC};
    const std::array<int, 3> cells = {4, 3, 2};
    const Function pressure        = [](int i, int j, int k) { return 1.0 + i * i + j + k; };

    // The x faces solved on the rank: from face 0 (the negative wall face is solved) or from face 1 (it is not)
    for (const int first_face : {0, 1})
    {
        CAPTURE(first_face);
        GhostExchangePlan<double> plan = singleRankPlan(walls_x);
        GhostedField rho(cells, 1, [](int, int, int) { return 2.0; });
        NodeField delta_pressure({0, 0, 0}, cells, pressure);
        std::array<std::unique_ptr<NodeField>, 3> span, delta_v;
        for (unsigned int ax = 0; ax < 3; ax++)
        {
            span[ax] = std::make_unique<NodeField>(std::array<int, 3> {0, 0, 0}, cells, [](int, int, int) { return 1.0; });
        }
        delta_v[0] = std::make_unique<NodeField>(std::array<int, 3> {first_face, 0, 0}, cells, [](int, int, int) { return 7.0; });
        delta_v[1] = std::make_unique<NodeField>(std::array<int, 3> {0, 0, 0}, cells, [](int, int, int) { return 7.0; });
        delta_v[2] = std::make_unique<NodeField>(std::array<int, 3> {0, 0, 0}, cells, [](int, int, int) { return 7.0; });

        VelocityCorrection::correctVelocity(3,
                                            1.0,
                                            rho.view,
                                            plan,
                                            *delta_pressure.view,
                                            {span[0]->view.get(), span[1]->view.get(), span[2]->view.get()},
                                            {delta_v[0]->view.get(), delta_v[1]->view.get(), delta_v[2]->view.get()});

        const Array3D<double>& vx = *delta_v[0]->parent;
        for (int k = 0; k < cells[2]; k++)
        {
            for (int j = 0; j < cells[1]; j++)
            {
                CAPTURE(j, k);
                if (first_face == 0)
                {
                    // The wall face is solved: its velocity delta is set to 0
                    REQUIRE(vx(0, j, k) == 0.0);
                }
                else
                {
                    // The wall face is not solved: it is left alone
                    REQUIRE(std::isnan(vx(0, j, k)));
                }
                // Inner faces: 7 - (P(i) - P(i - 1)) / rho_f, with rho_f = 2
                for (int i = 1; i < cells[0]; i++)
                {
                    CAPTURE(i);
                    REQUIRE(vx(i, j, k) == 7.0 - (pressure(i, j, k) - pressure(i - 1, j, k)) / 2.0);
                }
            }
        }
    }
}

namespace
{
    /**
     * @brief The velocity correction inputs and outputs MUSIC wrote for one rank.
     */
    struct MusicData
    {
        unsigned int ndim;
        double dt;
        std::array<int, 3> nc;
        int kg;
        std::unique_ptr<GhostedField> rho;
        std::unique_ptr<NodeField> delta_pressure;
        std::array<std::unique_ptr<NodeField>, 3> span, delta_v, expected;
        std::vector<std::tuple<int, int, int, double>> halo;
        std::array<int, 3> boundary;
    };

    void readData(const std::string& inputs_path, const std::string& outputs_path, MusicData& m)
    {
        FILE* file = fopen(inputs_path.c_str(), "r");
        REQUIRE(file != nullptr);
        REQUIRE(fscanf(file, " ndim %u", &m.ndim) == 1);
        REQUIRE(fscanf(file, " dt %lf", &m.dt) == 1);
        REQUIRE(fscanf(file, " local_cells %d %d %d", &m.nc[0], &m.nc[1], &m.nc[2]) == 3);
        m.kg = (m.ndim == 3) ? 1 : 0;

        // Solution rho with one ghost layer, i fastest
        const size_t num_ghosted = static_cast<size_t>(m.nc[0] + 2) * (m.nc[1] + 2) * (m.nc[2] + 2 * m.kg);
        std::vector<double> rho(num_ghosted);
        for (double& value : rho)
        {
            REQUIRE(fscanf(file, " %lf", &value) == 1);
        }
        const int gi = m.nc[0] + 2, gj = m.nc[1] + 2;
        m.rho = std::make_unique<GhostedField>(
            m.nc, m.kg, [&](int i, int j, int k) { return rho[(i + 1) + gi * ((j + 1) + gj * (k + m.kg))]; });

        // delta pressure on the cells, i fastest (MUSIC indices from 1)
        std::array<int, 6> slab;
        REQUIRE(fscanf(file, " delta_press_slab %d %d %d %d %d %d", &slab[0], &slab[1], &slab[2], &slab[3], &slab[4], &slab[5]) ==
                6);
        REQUIRE(slab == std::array<int, 6> {1, 1, 1, m.nc[0], m.nc[1], m.nc[2]});
        m.delta_pressure = std::make_unique<NodeField>(std::array<int, 3> {0, 0, 0}, m.nc);
        for (int k = 0; k < m.nc[2]; k++)
            for (int j = 0; j < m.nc[1]; j++)
                for (int i = 0; i < m.nc[0]; i++)
                    REQUIRE(fscanf(file, " %lf", &(*m.delta_pressure->view)(i, j, k)) == 1);

        // Per axis: the velocity faces, whether the negative face is a domain boundary, the grid spacing, the delta
        // pressure halo MUSIC read and the velocity delta before the correction
        for (unsigned int ax = 0; ax < m.ndim; ax++)
        {
            unsigned int axis;
            std::array<int, 3> beg, end;
            double delta;
            REQUIRE(fscanf(file,
                           " axis %u %d %d %d %d %d %d %d %lf",
                           &axis,
                           &beg[0],
                           &beg[1],
                           &beg[2],
                           &end[0],
                           &end[1],
                           &end[2],
                           &m.boundary[ax],
                           &delta) == 9);
            REQUIRE(axis == ax + 1);
            REQUIRE(delta > 0);
            // MUSIC slab [beg, end] from 1 is the local half open range [beg - 1, end)
            std::array<int, 3> start;
            for (unsigned int d = 0; d < 3; d++)
                start[d] = beg[d] - 1;
            m.span[ax]     = std::make_unique<NodeField>(start, end, [delta](int, int, int) { return delta; });
            m.delta_v[ax]  = std::make_unique<NodeField>(start, end);
            m.expected[ax] = std::make_unique<NodeField>(start, end);

            unsigned int count;
            REQUIRE(fscanf(file, " halo %u %u", &axis, &count) == 2);
            REQUIRE(axis == ax + 1);
            for (unsigned int n = 0; n < count; n++)
            {
                int i, j, k;
                double value;
                REQUIRE(fscanf(file, " %d %d %d %lf", &i, &j, &k, &value) == 4);
                m.halo.push_back({i - 1, j - 1, k - 1, value});
            }

            const Slab s = m.delta_v[ax]->view->cellSlab();
            for (int k = s.start(2); k < s.end(2); k++)
                for (int j = s.start(1); j < s.end(1); j++)
                    for (int i = s.start(0); i < s.end(0); i++)
                        REQUIRE(fscanf(file, " %lf", &(*m.delta_v[ax]->view)(i, j, k)) == 1);
        }
        fclose(file);

        // Expected velocity deltas after the correction, per axis, i fastest
        file = fopen(outputs_path.c_str(), "r");
        REQUIRE(file != nullptr);
        for (unsigned int ax = 0; ax < m.ndim; ax++)
        {
            const Slab s = m.expected[ax]->view->cellSlab();
            for (int k = s.start(2); k < s.end(2); k++)
                for (int j = s.start(1); j < s.end(1); j++)
                    for (int i = s.start(0); i < s.end(0); i++)
                        REQUIRE(fscanf(file, " %lf", &(*m.expected[ax]->view)(i, j, k)) == 1);
        }
        fclose(file);
    }

    /**
     * @brief Run correctVelocity on the MUSIC data of one rank with the given plan, and compare with the MUSIC outputs.
     * Also check that the ghost exchange of the plan gives the delta pressure halo MUSIC read.
     */
    void checkAgainstMusic(const std::string& prefix, GhostExchangePlan<double>& plan)
    {
        MusicData m;
        readData(prefix + "_inputs.dat", prefix + "_expected_outputs.dat", m);

        // The ghosts of delta pressure after the exchange are the halo MUSIC read
        Array3DWithGhosts<double> delta_pressure_g(*m.delta_pressure->view, plan);
        delta_pressure_g.exchangeGhosts();
        REQUIRE(!m.halo.empty());
        for (const auto& [i, j, k, value] : m.halo)
        {
            CAPTURE(i, j, k);
            REQUIRE(delta_pressure_g(i, j, k) == value);
        }

        // The Roberts dynamo test is periodic, so no rank has a domain boundary face
        for (unsigned int ax = 0; ax < m.ndim; ax++)
        {
            REQUIRE(m.boundary[ax] == 0);
        }

        VelocityCorrection::correctVelocity(m.ndim,
                                            m.dt,
                                            m.rho->view,
                                            plan,
                                            *m.delta_pressure->view,
                                            {m.span[0]->view.get(), m.span[1]->view.get(), m.span[2]->view.get()},
                                            {m.delta_v[0]->view.get(), m.delta_v[1]->view.get(), m.delta_v[2]->view.get()});

        // The two compilers may round the gradient differently, so compare with the size of each component
        for (unsigned int ax = 0; ax < m.ndim; ax++)
        {
            CAPTURE(ax);
            const Slab s = m.expected[ax]->view->cellSlab();
            double scale = 0;
            for (int k = s.start(2); k < s.end(2); k++)
                for (int j = s.start(1); j < s.end(1); j++)
                    for (int i = s.start(0); i < s.end(0); i++)
                        scale = std::max(scale, std::abs((*m.expected[ax]->view)(i, j, k)));
            REQUIRE(scale > 0);
            for (int k = s.start(2); k < s.end(2); k++)
            {
                for (int j = s.start(1); j < s.end(1); j++)
                {
                    for (int i = s.start(0); i < s.end(0); i++)
                    {
                        CAPTURE(i, j, k);
                        REQUIRE(std::abs((*m.delta_v[ax]->view)(i, j, k) - (*m.expected[ax]->view)(i, j, k)) <= 1e-14 * scale);
                    }
                }
            }
        }
    }
}  // namespace

TEST_CASE("VelocityCorrection matches the MUSIC velocity correction on the Roberts dynamo test on 1 rank")
{
    // Periodic in every direction, so the single rank is its own neighbour on every face
    GhostExchangePlan<double> plan = singleRankPlan(ALL_PERIODIC);
    checkAgainstMusic(std::string(TEST_DATA_DIR) + "/velocity_correction_roberts_dynamo_1rank_rank0", plan);
}

TEST_CASE("VelocityCorrection matches the MUSIC velocity correction on the Roberts dynamo test on 4 ranks")
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
    std::array<int, 6> neighbours;
    for (int axis = 0; axis < 3; axis++)
    {
        MPI_Cart_shift(cartesian, axis, 1, &neighbours[axis * 2], &neighbours[axis * 2 + 1]);
    }
    const FaceConditions inner = {BoundaryConditions::INNER,
                                  BoundaryConditions::INNER,
                                  BoundaryConditions::INNER,
                                  BoundaryConditions::INNER,
                                  BoundaryConditions::INNER,
                                  BoundaryConditions::INNER};
    GhostExchangePlan<double> plan(inner, neighbours, cartesian);
    checkAgainstMusic(std::string(TEST_DATA_DIR) + "/velocity_correction_roberts_dynamo_4ranks_rank" + std::to_string(rank),
                      plan);
    MPI_Comm_free(&cartesian);
}
