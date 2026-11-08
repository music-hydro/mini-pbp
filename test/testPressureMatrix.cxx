#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "DirectionalArray.h"
#include "GhostExchangePlan.h"
#include "PackedVariables.h"
#include "PhysicsBasedPreconditioner.h"
#include "Slab.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdio>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using GI = long long;

    const DirectionalArray<BoundaryConditions> ALL_PERIODIC({BoundaryConditions::PERIODIC,
                                                             BoundaryConditions::PERIODIC,
                                                             BoundaryConditions::PERIODIC,
                                                             BoundaryConditions::PERIODIC,
                                                             BoundaryConditions::PERIODIC,
                                                             BoundaryConditions::PERIODIC});

    /**
     * @brief The same boundary condition on every face.
     */
    DirectionalArray<BoundaryConditions> allFaces(const BoundaryConditions bc)
    { return DirectionalArray<BoundaryConditions>({bc, bc, bc, bc, bc, bc}); }

    /**
     * @brief Neighbours of a single rank on MPI_COMM_SELF: itself across a periodic face, none across any other face.
     */
    DirectionalArray<int> singleRankNeighbours(const DirectionalArray<BoundaryConditions>& bcs)
    {
        DirectionalArray<int> neighbours;
        for (unsigned int face = 0; face < 6; face++)
        {
            neighbours[face] = (bcs[face] == BoundaryConditions::PERIODIC) ? 0 : MPI_PROC_NULL;
        }
        return neighbours;
    }

    /**
     * @brief Everything assemblePressureMatrix needs on one rank: the preconditioner, rho with its plan, the squared
     * sound speed and the Laplacian coefficients. Fields are filled with 1, the Laplacian coefficients with 1 along the
     * first ndim axes and 0 along the others.
     */
    struct Problem
    {
        Problem(const unsigned int ndim_,
                const Slab& global_domain,
                const Slab& local_domain,
                const DirectionalArray<BoundaryConditions>& bcs,
                const DirectionalArray<int>& neighbours,
                const MPI_Comm communicator,
                const unsigned int prec_nvar = 1) :
            ndim(ndim_),
            nc({int(local_domain.extent(0)), int(local_domain.extent(1)), int(local_domain.extent(2))}),
            num_cells(size_t(nc[0]) * nc[1] * nc[2]),
            solution_raw(3 * num_cells, 1.0),
            solution_dPd_raw(2 * num_cells, 1.0),
            asqr_raw(num_cells, 1.0),
            rho_raw(num_cells, 1.0),
            laplacian_raw(num_cells),
            solution(solution_raw, nc[0], nc[1], nc[2]),
            solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]),
            squared_sound_speed(nc[0], nc[1], nc[2], asqr_raw),
            rho_cells(nc[0], nc[1], nc[2], rho_raw),
            laplacian(nc[0], nc[1], nc[2], laplacian_raw),
            plan(global_domain, local_domain, bcs, neighbours, communicator),
            rho(rho_cells, plan),
            preconditioner(std::make_unique<PhysicsBasedPreconditioner>(solution, solution_dPd, squared_sound_speed, prec_nvar))
        { setLaplacian(1.0); }

        /// The Laplacian coefficients of every face along the first ndim axes, 0 along the others
        void setLaplacian(const double value)
        {
            for (DirectionalArray<double>& faces : laplacian_raw)
            {
                for (unsigned int axis = 0; axis < 3; axis++)
                {
                    faces.get(axis, Direction::NEGATIVE) = (axis < ndim) ? value : 0.0;
                    faces.get(axis, Direction::POSITIVE) = (axis < ndim) ? value : 0.0;
                }
            }
        }

        void assemble(const double dt)
        { preconditioner->assemblePressureMatrix(ndim, dt, rho, squared_sound_speed, laplacian); }

        unsigned int ndim;
        std::array<int, 3> nc;
        size_t num_cells;
        std::vector<double> solution_raw, solution_dPd_raw, asqr_raw, rho_raw;
        std::vector<DirectionalArray<double>> laplacian_raw;
        PackedVariables solution, solution_dPd;
        Array3D<double> squared_sound_speed, rho_cells;
        Array3D<DirectionalArray<double>> laplacian;
        GhostExchangePlan<double> plan;
        Array3DWithGhosts<double> rho;
        std::unique_ptr<PhysicsBasedPreconditioner> preconditioner;
    };

    /**
     * @brief A single rank holding the whole domain of the given number of cells.
     */
    std::unique_ptr<Problem> singleRank(const unsigned int ndim,
                                        const std::array<int, 3> nc,
                                        const DirectionalArray<BoundaryConditions>& bcs,
                                        const unsigned int prec_nvar = 1)
    {
        const Slab domain({0, 0, 0}, nc);
        return std::make_unique<Problem>(ndim, domain, domain, bcs, singleRankNeighbours(bcs), MPI_COMM_SELF, prec_nvar);
    }

    /**
     * @brief Check one row: its global index, columns in order and values.
     */
    void checkRow(const PhysicsBasedPreconditioner& p,
                  const size_t local_row,
                  const GI expected_row,
                  const std::vector<GI>& expected_columns,
                  const std::vector<double>& expected_values)
    {
        REQUIRE(p.pressureMatrixRowGlobalIndices().at(local_row) == expected_row);
        const size_t begin = p.pressureMatrixRowOffsets().at(local_row);
        const size_t end   = p.pressureMatrixRowOffsets().at(local_row + 1);
        REQUIRE(end - begin == expected_columns.size());
        for (size_t n = 0; n < expected_columns.size(); n++)
        {
            CHECK(p.pressureMatrixColumnGlobalIndices()[begin + n] == expected_columns[n]);
            CHECK(p.pressureMatrixValues()[begin + n] == expected_values[n]);
        }
    }

    // dt = 0.5, c^2 = 2 and Laplacian coefficients 4 give J_PP = -1, 1/dt = 2 and off diagonals -4
    const double dt  = 0.5;
    const double cs2 = 2.0;
    const double lap = 4.0;

    void setUniform(Problem& p)
    {
        p.setLaplacian(lap);
        p.squared_sound_speed.fill(cs2);
    }
}  // namespace

TEST_CASE("assemblePressureMatrix rejects invalid input")
{
    // prec_nvar 2 is not translated (Thomas on #27: assume prec_nvar 1)
    std::unique_ptr<Problem> p2 = singleRank(3, {3, 3, 3}, ALL_PERIODIC, 2);
    REQUIRE_THROWS_AS(p2->assemble(dt), std::invalid_argument);

    std::unique_ptr<Problem> p = singleRank(3, {3, 3, 3}, ALL_PERIODIC);
    REQUIRE_THROWS_AS(p->preconditioner->assemblePressureMatrix(1, dt, p->rho, p->squared_sound_speed, p->laplacian),
                      std::invalid_argument);

    // 2D needs a single cell along k
    std::unique_ptr<Problem> p3 = singleRank(2, {3, 3, 2}, ALL_PERIODIC);
    REQUIRE_THROWS_AS(p3->assemble(dt), std::invalid_argument);

    // The cells of a rank must lie inside the global domain
    const DirectionalArray<BoundaryConditions> open = allFaces(BoundaryConditions::ZERO_DERIVATIVE);
    REQUIRE_THROWS_AS(
        GhostExchangePlan<double>(
            Slab({0, 0, 0}, {3, 3, 3}), Slab({1, 0, 0}, {4, 3, 3}), open, singleRankNeighbours(open), MPI_COMM_SELF),
        std::invalid_argument);
}

TEST_CASE("assemblePressureMatrix numbers unknowns as MUSIC")
{
    // Right half of a 4 x 3 x 2 domain: global_dof_idx0 = i + ni * (j + nj * k) over the whole domain
    const DirectionalArray<BoundaryConditions> open = allFaces(BoundaryConditions::ZERO_DERIVATIVE);
    Problem p(3, Slab({0, 0, 0}, {4, 3, 2}), Slab({2, 0, 0}, {4, 3, 2}), open, singleRankNeighbours(open), MPI_COMM_SELF);
    p.assemble(dt);

    // Rows in the local order of the cells, i fastest
    const std::vector<GI> expected = {2, 3, 6, 7, 10, 11, 14, 15};
    for (size_t n = 0; n < expected.size(); n++)
    {
        CHECK(p.preconditioner->pressureMatrixRowGlobalIndices()[n] == expected[n]);
    }
    REQUIRE(p.preconditioner->pressureMatrixRowGlobalIndices().size() == 2 * 3 * 2);
    REQUIRE(p.preconditioner->pressureMatrixRowOffsets().size() == p.preconditioner->pressureMatrixRowGlobalIndices().size() + 1);
}

TEST_CASE("assemblePressureMatrix periodic 3D stencil")
{
    std::unique_ptr<Problem> p = singleRank(3, {3, 3, 3}, ALL_PERIODIC);
    setUniform(*p);
    p->assemble(dt);

    REQUIRE(p->preconditioner->pressureMatrixRowGlobalIndices().size() == 27);
    REQUIRE(p->preconditioner->pressureMatrixValues().size() == 27 * 7);

    // Diagonal 1/dt + dt c^2 * 6 lap, neighbours across the periodic boundaries wrap around
    checkRow(*p->preconditioner, 0, 0, {0, 2, 1, 6, 3, 18, 9}, {26, -4, -4, -4, -4, -4, -4});
    // Centre cell (1, 1, 1)
    checkRow(*p->preconditioner, 13, 13, {13, 12, 14, 10, 16, 4, 22}, {26, -4, -4, -4, -4, -4, -4});

    // A second call replaces the matrix rather than appending to it
    p->assemble(dt);
    REQUIRE(p->preconditioner->pressureMatrixValues().size() == 27 * 7);
}

TEST_CASE("assemblePressureMatrix open boundary keeps the face in the diagonal")
{
    std::unique_ptr<Problem> p = singleRank(3, {3, 3, 3}, allFaces(BoundaryConditions::ZERO_DERIVATIVE));
    setUniform(*p);
    p->assemble(dt);

    // Corner cell: no neighbours outside the domain, but the diagonal still has all 6 faces
    checkRow(*p->preconditioner, 0, 0, {0, 1, 3, 9}, {26, -4, -4, -4});
    checkRow(*p->preconditioner, 13, 13, {13, 12, 14, 10, 16, 4, 22}, {26, -4, -4, -4, -4, -4, -4});
}

TEST_CASE("assemblePressureMatrix reflective boundary drops the face")
{
    std::unique_ptr<Problem> p = singleRank(3, {3, 3, 3}, allFaces(BoundaryConditions::REFLECTIVE));
    setUniform(*p);
    p->assemble(dt);

    // Corner cell: the 3 boundary faces have weight 0, diagonal 2 + 3 * 4
    checkRow(*p->preconditioner, 0, 0, {0, 1, 3, 9}, {14, -4, -4, -4});
    // Last cell (2, 2, 2)
    checkRow(*p->preconditioner, 26, 26, {26, 25, 23, 17}, {14, -4, -4, -4});
    checkRow(*p->preconditioner, 13, 13, {13, 12, 14, 10, 16, 4, 22}, {26, -4, -4, -4, -4, -4, -4});
}

TEST_CASE("assemblePressureMatrix weights the coefficients by rho / rho_face")
{
    std::unique_ptr<Problem> p = singleRank(3, {3, 3, 3}, ALL_PERIODIC);
    setUniform(*p);
    p->rho_cells(1, 1, 1) = 3.0;
    p->assemble(dt);

    // Cell (1, 1, 1): rho 3, every face rho_f = 2, coef = 4 * 3 / 2 = 6
    checkRow(*p->preconditioner, 13, 13, {13, 12, 14, 10, 16, 4, 22}, {38, -6, -6, -6, -6, -6, -6});
    // Cell (0, 1, 1): rho 1, its positive x face has rho_f = 2, coef = 4 * 1 / 2 = 2
    checkRow(*p->preconditioner, 12, 12, {12, 14, 13, 9, 15, 3, 21}, {24, -4, -2, -4, -4, -4, -4});
}

TEST_CASE("assemblePressureMatrix 2D stencil")
{
    // Periodic in x and y; the k faces are never read in 2D
    DirectionalArray<BoundaryConditions> bcs = ALL_PERIODIC;
    bcs.get(2, Direction::NEGATIVE)          = BoundaryConditions::ZERO_DERIVATIVE;
    bcs.get(2, Direction::POSITIVE)          = BoundaryConditions::ZERO_DERIVATIVE;
    std::unique_ptr<Problem> p               = singleRank(2, {3, 3, 1}, bcs);
    setUniform(*p);
    p->assemble(dt);

    REQUIRE(p->preconditioner->pressureMatrixValues().size() == 9 * 5);
    checkRow(*p->preconditioner, 0, 0, {0, 2, 1, 6, 3}, {18, -4, -4, -4, -4});
    checkRow(*p->preconditioner, 4, 4, {4, 3, 5, 1, 7}, {18, -4, -4, -4, -4});
}

TEST_CASE("assemblePressureMatrix neighbours on another rank")
{
    // Right half of a non periodic 4 x 2 x 2 domain
    const DirectionalArray<BoundaryConditions> open = allFaces(BoundaryConditions::ZERO_DERIVATIVE);
    Problem p(3, Slab({0, 0, 0}, {4, 2, 2}), Slab({2, 0, 0}, {4, 2, 2}), open, singleRankNeighbours(open), MPI_COMM_SELF);
    setUniform(p);
    p.assemble(dt);

    // Local cell (0, 0, 0) is global (2, 0, 0): its negative x neighbour (1, 0, 0) is on the other rank
    checkRow(*p.preconditioner, 0, 2, {2, 1, 3, 6, 10}, {26, -4, -4, -4, -4});
    // Local cell (1, 0, 0) is global (3, 0, 0), at the positive end of the domain
    checkRow(*p.preconditioner, 1, 3, {3, 2, 7, 11}, {26, -4, -4, -4});
}

namespace
{
    /**
     * @brief The set_matrix inputs MUSIC wrote for one rank.
     */
    struct MusicInputs
    {
        unsigned int ndim, prec_nvar;
        double dt;
        std::array<int, 3> local_cells, global_start, global_cells, periodic;
        std::array<int, 6> reflective;
        std::vector<double> rho_ghosted;  // rho with one ghost layer (none along k in 2D), i fastest
        std::vector<double> asqr;
        std::vector<DirectionalArray<double>> laplacian;
    };

    MusicInputs readInputs(const std::string& path)
    {
        MusicInputs m;
        FILE* file = fopen(path.c_str(), "r");
        REQUIRE(file != nullptr);
        REQUIRE(fscanf(file, " ndim_prec_nvar %u %u", &m.ndim, &m.prec_nvar) == 2);
        REQUIRE(m.prec_nvar == 1);
        REQUIRE(fscanf(file, " dt %lf", &m.dt) == 1);
        REQUIRE(fscanf(file, " local_cells %d %d %d", &m.local_cells[0], &m.local_cells[1], &m.local_cells[2]) == 3);
        REQUIRE(fscanf(file, " global_start %d %d %d", &m.global_start[0], &m.global_start[1], &m.global_start[2]) == 3);
        REQUIRE(fscanf(file, " global_cells %d %d %d", &m.global_cells[0], &m.global_cells[1], &m.global_cells[2]) == 3);
        REQUIRE(fscanf(file, " periodic %d %d %d", &m.periodic[0], &m.periodic[1], &m.periodic[2]) == 3);
        REQUIRE(fscanf(file,
                       " reflective %d %d %d %d %d %d",
                       &m.reflective[0],
                       &m.reflective[1],
                       &m.reflective[2],
                       &m.reflective[3],
                       &m.reflective[4],
                       &m.reflective[5]) == 6);

        const int kg             = (m.ndim == 3) ? 1 : 0;
        const size_t num_ghosted = size_t(m.local_cells[0] + 2) * (m.local_cells[1] + 2) * (m.local_cells[2] + 2 * kg);
        const size_t num_cells   = size_t(m.local_cells[0]) * m.local_cells[1] * m.local_cells[2];
        m.rho_ghosted.resize(num_ghosted);
        for (double& v : m.rho_ghosted)
        {
            REQUIRE(fscanf(file, " %lf", &v) == 1);
        }
        m.asqr.resize(num_cells);
        m.laplacian.resize(num_cells);
        for (size_t c = 0; c < num_cells; c++)
        {
            REQUIRE(fscanf(file,
                           " %lf %lf %lf %lf %lf %lf %lf",
                           &m.asqr[c],
                           &m.laplacian[c].get(0, Direction::NEGATIVE),
                           &m.laplacian[c].get(1, Direction::NEGATIVE),
                           &m.laplacian[c].get(2, Direction::NEGATIVE),
                           &m.laplacian[c].get(0, Direction::POSITIVE),
                           &m.laplacian[c].get(1, Direction::POSITIVE),
                           &m.laplacian[c].get(2, Direction::POSITIVE)) == 7);
        }
        fclose(file);
        return m;
    }

    /**
     * @brief Assemble the matrix of one rank from the MUSIC inputs with the given neighbours and boundary conditions,
     * check that the exchanged rho ghosts are the ghost layer MUSIC read, and compare every row, column and value with
     * the MUSIC matrix (before the D A D scaling).
     */
    void checkAgainstMusic(const std::string& label,
                           const int rank,
                           const DirectionalArray<BoundaryConditions>& bcs,
                           const DirectionalArray<int>& neighbours,
                           const MPI_Comm communicator,
                           const std::array<int, 3>& expected_global_start)
    {
        const std::string data_dir = TEST_DATA_DIR;
        const MusicInputs m =
            readInputs(data_dir + "/pressure_matrix_roberts_dynamo_" + label + "_rank" + std::to_string(rank) + "_inputs.dat");
        REQUIRE(m.global_start == expected_global_start);
        REQUIRE(m.periodic == std::array<int, 3> {1, 1, 1});
        REQUIRE(m.reflective == std::array<int, 6> {0, 0, 0, 0, 0, 0});

        const std::array<int, 3> local_end = {
            m.global_start[0] + m.local_cells[0], m.global_start[1] + m.local_cells[1], m.global_start[2] + m.local_cells[2]};
        Problem p(m.ndim, Slab({0, 0, 0}, m.global_cells), Slab(m.global_start, local_end), bcs, neighbours, communicator);
        const int kg = (m.ndim == 3) ? 1 : 0;
        const int gi = m.local_cells[0] + 2, gj = m.local_cells[1] + 2;
        auto ghosted = [&](int i, int j, int k) { return m.rho_ghosted[(i + 1) + gi * ((j + 1) + gj * (k + kg))]; };
        size_t c     = 0;
        for (int k = 0; k < m.local_cells[2]; k++)
        {
            for (int j = 0; j < m.local_cells[1]; j++)
            {
                for (int i = 0; i < m.local_cells[0]; i++, c++)
                {
                    p.rho_cells(i, j, k)           = ghosted(i, j, k);
                    p.squared_sound_speed(i, j, k) = m.asqr[c];
                    p.laplacian(i, j, k)           = m.laplacian[c];
                }
            }
        }

        p.preconditioner->assemblePressureMatrix(m.ndim, m.dt, p.rho, p.squared_sound_speed, p.laplacian);
        const PhysicsBasedPreconditioner& matrix = *p.preconditioner;

        // The face ghosts of rho after the exchange are MUSIC's ghost layer
        const std::array<int, 3> n = m.local_cells;
        for (int k = 0; k < n[2]; k++)
        {
            for (int j = 0; j < n[1]; j++)
            {
                CAPTURE(j, k);
                REQUIRE(p.rho(-1, j, k) == ghosted(-1, j, k));
                REQUIRE(p.rho(n[0], j, k) == ghosted(n[0], j, k));
            }
            for (int i = 0; i < n[0]; i++)
            {
                CAPTURE(i, k);
                REQUIRE(p.rho(i, -1, k) == ghosted(i, -1, k));
                REQUIRE(p.rho(i, n[1], k) == ghosted(i, n[1], k));
            }
        }

        const std::string matrix_path =
            data_dir + "/pressure_solve_roberts_dynamo_" + label + "_rank" + std::to_string(rank) + "_matrix_inputs.dat";
        FILE* file = fopen(matrix_path.c_str(), "r");
        REQUIRE(file != nullptr);
        long long row, column;
        double value;
        size_t entry = 0, local_row = 0;
        while (fscanf(file, "%lld %lld %lf", &row, &column, &value) == 3)
        {
            REQUIRE(entry < matrix.pressureMatrixValues().size());
            while (matrix.pressureMatrixRowOffsets()[local_row + 1] <= entry)
            {
                local_row++;
            }
            REQUIRE(matrix.pressureMatrixRowGlobalIndices()[local_row] == row);
            REQUIRE(matrix.pressureMatrixColumnGlobalIndices()[entry] == column);
            REQUIRE_THAT(matrix.pressureMatrixValues()[entry], Catch::Matchers::WithinRel(value, 1e-14));
            entry++;
        }
        fclose(file);
        REQUIRE(entry == matrix.pressureMatrixValues().size());
    }
}  // namespace

TEST_CASE("assemblePressureMatrix matches the MUSIC matrix of the Roberts dynamo test on 1 rank")
{
    // Periodic in every direction, so the single rank is its own neighbour on every face
    checkAgainstMusic("1rank", 0, ALL_PERIODIC, singleRankNeighbours(ALL_PERIODIC), MPI_COMM_SELF, {0, 0, 0});
}

TEST_CASE("assemblePressureMatrix matches the MUSIC matrix of the Roberts dynamo test on 4 ranks")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 4) { SKIP("The 4 rank MUSIC data needs exactly 4 ranks"); }

    // MUSIC splits the periodic 16^3 domain 2 x 2 x 1 with rank = 2 * (x block) + (y block), which is the order of a
    // cartesian communicator of dims {2, 2, 1} without reordering. Every face has a neighbour rank, so every face is
    // INNER, also across the periodic domain boundary.
    std::array<int, 3> dims    = {2, 2, 1};
    std::array<int, 3> periods = {1, 1, 1};
    MPI_Comm cartesian;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods.data(), false, &cartesian);
    int rank;
    MPI_Comm_rank(cartesian, &rank);
    std::array<int, 3> coords;
    MPI_Cart_coords(cartesian, rank, 3, coords.data());
    DirectionalArray<int> neighbours;
    for (int axis = 0; axis < 3; axis++)
    {
        MPI_Cart_shift(
            cartesian, axis, 1, &neighbours.get(axis, Direction::NEGATIVE), &neighbours.get(axis, Direction::POSITIVE));
    }
    checkAgainstMusic("4ranks",
                      rank,
                      allFaces(BoundaryConditions::INNER),
                      neighbours,
                      cartesian,
                      {8 * coords[0], 8 * coords[1], 16 * coords[2]});
    MPI_Comm_free(&cartesian);
}
