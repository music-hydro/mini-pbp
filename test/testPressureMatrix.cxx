#include "Array3D.h"
#include "PressureMatrix.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
    using GI = PressureMatrix::GlobalIndex;

    /**
     * @brief Storage for the inputs of PressureMatrix::assemble on the cells of one rank.
     *
     * The solution fields have one ghost layer on each side (none along k in 2D), the squared sound speed and the
     * Laplacian coefficients have one value per cell.
     */
    struct CellData
    {
        unsigned int ndim;
        std::array<unsigned int, 3> nc;
        unsigned int kg;
        std::vector<double> rho, pressure, dPde, chi, dTde;
        std::vector<double> squaredSoundSpeed;
        std::array<std::array<std::vector<double>, 2>, 3> laplacian;

        CellData(const unsigned int ndim_, const std::array<unsigned int, 3> nc_) :
            ndim(ndim_),
            nc(nc_),
            kg(ndim_ == 3 ? 1 : 0)
        {
            const size_t numGhosted = size_t(nc[0] + 2) * (nc[1] + 2) * (nc[2] + 2 * kg);
            const size_t numCells   = size_t(nc[0]) * nc[1] * nc[2];
            for (std::vector<double>* field : {&rho, &pressure, &dPde, &chi, &dTde})
            {
                field->assign(numGhosted, 1.0);
            }
            squaredSoundSpeed.assign(numCells, 1.0);
            for (unsigned int axis = 0; axis < 3; axis++)
            {
                for (unsigned int side = 0; side < 2; side++)
                {
                    laplacian[axis][side].assign(numCells, (axis < ndim) ? 1.0 : 0.0);
                }
            }
        }

        /// Position in a ghosted field of local cell (i, j, k), from -1 to nc along each axis with ghosts
        size_t ghosted(const int i, const int j, const int k) const
        { return size_t(i + 1) + size_t(nc[0] + 2) * (size_t(j + 1) + size_t(nc[1] + 2) * size_t(k + int(kg))); }

        void setLaplacian(const double value)
        {
            for (unsigned int axis = 0; axis < ndim; axis++)
            {
                for (unsigned int side = 0; side < 2; side++)
                {
                    laplacian[axis][side].assign(laplacian[axis][side].size(), value);
                }
            }
        }

        /// Assemble `matrix` from this data
        void assemble(PressureMatrix& matrix, const double dt)
        {
            const unsigned int gi = nc[0] + 2, gj = nc[1] + 2, gk = nc[2] + 2 * kg;
            Array3D<double> vrho(gi, gj, gk, rho.data());
            Array3D<double> vpressure(gi, gj, gk, pressure.data());
            Array3D<double> vdPde(gi, gj, gk, dPde.data());
            Array3D<double> vchi(gi, gj, gk, chi.data());
            Array3D<double> vdTde(gi, gj, gk, dTde.data());
            Array3D<double> vasqr(nc[0], nc[1], nc[2], squaredSoundSpeed.data());
            Array3D<double> lapILeft(nc[0], nc[1], nc[2], laplacian[0][0].data());
            Array3D<double> lapIRight(nc[0], nc[1], nc[2], laplacian[0][1].data());
            Array3D<double> lapJLeft(nc[0], nc[1], nc[2], laplacian[1][0].data());
            Array3D<double> lapJRight(nc[0], nc[1], nc[2], laplacian[1][1].data());
            Array3D<double> lapKLeft(nc[0], nc[1], nc[2], laplacian[2][0].data());
            Array3D<double> lapKRight(nc[0], nc[1], nc[2], laplacian[2][1].data());
            PressureMatrix::SolutionFields solution         = {&vrho, &vpressure, &vdPde, &vchi, &vdTde};
            PressureMatrix::LaplacianCoefficients lapCoeffs = {
                {{&lapILeft, &lapIRight}, {&lapJLeft, &lapJRight}, {&lapKLeft, &lapKRight}}
            };
            matrix.assemble(dt, solution, vasqr, lapCoeffs);
        }
    };

    /**
     * @brief Decomposition of a whole domain held by a single rank.
     */
    PressureMatrix::Decomposition
        singleRank(const unsigned int ndim, const std::array<unsigned int, 3> nc, const bool periodic, const bool reflective)
    {
        PressureMatrix::Decomposition d;
        d.ndim        = ndim;
        d.localCells  = nc;
        d.globalStart = {0, 0, 0};
        d.globalCells = {GI(nc[0]), GI(nc[1]), GI(nc[2])};
        d.periodic    = {periodic, periodic, periodic && ndim == 3};
        for (unsigned int axis = 0; axis < 3; axis++)
        {
            d.reflective[axis] = {reflective && axis < ndim, reflective && axis < ndim};
        }
        return d;
    }

    /**
     * @brief Check one row: its global index, columns in order and values.
     */
    void checkRow(const PressureMatrix& matrix,
                  const size_t localRow,
                  const GI expectedRow,
                  const std::vector<GI>& expectedColumns,
                  const std::vector<double>& expectedValues)
    {
        REQUIRE(matrix.rowGlobalIndices().at(localRow) == expectedRow);
        const size_t begin = matrix.rowOffsets().at(localRow);
        const size_t end   = matrix.rowOffsets().at(localRow + 1);
        REQUIRE(end - begin == expectedColumns.size());
        for (size_t n = 0; n < expectedColumns.size(); n++)
        {
            CHECK(matrix.columnGlobalIndices()[begin + n] == expectedColumns[n]);
            CHECK(matrix.values()[begin + n] == expectedValues[n]);
        }
    }

    // dt = 0.5, c^2 = 2 and Laplacian coefficients 4 give J_PP = -1, 1/dt = 2 and off diagonals -4
    const double dt  = 0.5;
    const double cs2 = 2.0;
    const double lap = 4.0;
}  // namespace

TEST_CASE("PressureMatrix rejects invalid input")
{
    PressureMatrix::Decomposition d = singleRank(3, {3, 3, 3}, true, false);
    REQUIRE_THROWS_AS(PressureMatrix(d, 3), std::invalid_argument);

    d.ndim = 1;
    REQUIRE_THROWS_AS(PressureMatrix(d), std::invalid_argument);

    PressureMatrix::Decomposition d2 = singleRank(2, {3, 3, 2}, true, false);
    REQUIRE_THROWS_AS(PressureMatrix(d2), std::invalid_argument);

    PressureMatrix::Decomposition d3 = singleRank(3, {3, 3, 3}, true, false);
    d3.globalStart                   = {1, 0, 0};
    REQUIRE_THROWS_AS(PressureMatrix(d3), std::invalid_argument);
}

TEST_CASE("PressureMatrix numbers unknowns as MUSIC")
{
    // Right half of a 4 x 3 x 2 domain, prec_nvar = 2
    PressureMatrix::Decomposition d = singleRank(3, {2, 3, 2}, false, false);
    d.globalCells                   = {4, 3, 2};
    d.globalStart                   = {2, 0, 0};
    PressureMatrix matrix(d, 2);

    // ivar + prec_nvar * (i + ni * (j + nj * k)) over the whole domain
    REQUIRE(matrix.globalDofIndex(0, 0, 0, 0) == 0);
    REQUIRE(matrix.globalDofIndex(1, 2, 1, 1) == 1 + 2 * (1 + 4 * (2 + 3 * 1)));

    CellData data(3, d.localCells);
    data.assemble(matrix, dt);

    // Rows in the local order (ivar, i, j, k)
    const std::vector<GI> expected = {4, 5, 6, 7, 12, 13};
    for (size_t n = 0; n < expected.size(); n++)
    {
        CHECK(matrix.rowGlobalIndices()[n] == expected[n]);
    }
    REQUIRE(matrix.rowGlobalIndices().size() == 2 * 3 * 2 * 2);
    REQUIRE(matrix.rowOffsets().size() == matrix.rowGlobalIndices().size() + 1);
}

TEST_CASE("PressureMatrix periodic 3D stencil")
{
    PressureMatrix matrix(singleRank(3, {3, 3, 3}, true, false));
    CellData data(3, {3, 3, 3});
    data.setLaplacian(lap);
    data.squaredSoundSpeed.assign(data.squaredSoundSpeed.size(), cs2);
    data.assemble(matrix, dt);

    REQUIRE(matrix.rowGlobalIndices().size() == 27);
    REQUIRE(matrix.values().size() == 27 * 7);

    // Diagonal 1/dt + dt c^2 * 6 lap, neighbours across the periodic boundaries wrap around
    checkRow(matrix, 0, 0, {0, 2, 1, 6, 3, 18, 9}, {26, -4, -4, -4, -4, -4, -4});
    // Centre cell (1, 1, 1)
    checkRow(matrix, 13, 13, {13, 12, 14, 10, 16, 4, 22}, {26, -4, -4, -4, -4, -4, -4});
}

TEST_CASE("PressureMatrix open boundary keeps the face in the diagonal")
{
    PressureMatrix matrix(singleRank(3, {3, 3, 3}, false, false));
    CellData data(3, {3, 3, 3});
    data.setLaplacian(lap);
    data.squaredSoundSpeed.assign(data.squaredSoundSpeed.size(), cs2);
    data.assemble(matrix, dt);

    // Corner cell: no neighbours outside the domain, but the diagonal still has all 6 faces
    checkRow(matrix, 0, 0, {0, 1, 3, 9}, {26, -4, -4, -4});
    checkRow(matrix, 13, 13, {13, 12, 14, 10, 16, 4, 22}, {26, -4, -4, -4, -4, -4, -4});
}

TEST_CASE("PressureMatrix reflective boundary drops the face")
{
    PressureMatrix matrix(singleRank(3, {3, 3, 3}, false, true));
    CellData data(3, {3, 3, 3});
    data.setLaplacian(lap);
    data.squaredSoundSpeed.assign(data.squaredSoundSpeed.size(), cs2);
    data.assemble(matrix, dt);

    // Corner cell: the 3 boundary faces have weight 0, diagonal 2 + 3 * 4
    checkRow(matrix, 0, 0, {0, 1, 3, 9}, {14, -4, -4, -4});
    // Last cell (2, 2, 2)
    checkRow(matrix, 26, 26, {26, 25, 23, 17}, {14, -4, -4, -4});
    checkRow(matrix, 13, 13, {13, 12, 14, 10, 16, 4, 22}, {26, -4, -4, -4, -4, -4, -4});
}

TEST_CASE("PressureMatrix weights the coefficients by rho / rho_face")
{
    PressureMatrix matrix(singleRank(3, {3, 3, 3}, true, false));
    CellData data(3, {3, 3, 3});
    data.setLaplacian(lap);
    data.squaredSoundSpeed.assign(data.squaredSoundSpeed.size(), cs2);
    data.rho[data.ghosted(1, 1, 1)] = 3.0;
    data.assemble(matrix, dt);

    // Cell (1, 1, 1): rho 3, every face rho_f = 2, coef = 4 * 3 / 2 = 6
    checkRow(matrix, 13, 13, {13, 12, 14, 10, 16, 4, 22}, {38, -6, -6, -6, -6, -6, -6});
    // Cell (0, 1, 1): rho 1, its right x face has rho_f = 2, coef = 4 * 1 / 2 = 2
    checkRow(matrix, 12, 12, {12, 14, 13, 9, 15, 3, 21}, {24, -4, -2, -4, -4, -4, -4});
}

TEST_CASE("PressureMatrix 2D stencil")
{
    PressureMatrix matrix(singleRank(2, {3, 3, 1}, true, false));
    CellData data(2, {3, 3, 1});
    data.setLaplacian(lap);
    data.squaredSoundSpeed.assign(data.squaredSoundSpeed.size(), cs2);
    data.assemble(matrix, dt);

    REQUIRE(matrix.values().size() == 9 * 5);
    checkRow(matrix, 0, 0, {0, 2, 1, 6, 3}, {18, -4, -4, -4, -4});
    checkRow(matrix, 4, 4, {4, 3, 5, 1, 7}, {18, -4, -4, -4, -4});
}

TEST_CASE("PressureMatrix neighbours on another rank")
{
    // Right half of a non periodic 4 x 2 x 2 domain
    PressureMatrix::Decomposition d = singleRank(3, {2, 2, 2}, false, false);
    d.globalCells                   = {4, 2, 2};
    d.globalStart                   = {2, 0, 0};
    PressureMatrix matrix(d);
    CellData data(3, d.localCells);
    data.setLaplacian(lap);
    data.squaredSoundSpeed.assign(data.squaredSoundSpeed.size(), cs2);
    data.assemble(matrix, dt);

    // Local cell (0, 0, 0) is global (2, 0, 0): its left x neighbour (1, 0, 0) is on the other rank
    checkRow(matrix, 0, 2, {2, 1, 3, 6, 10}, {26, -4, -4, -4, -4});
    // Local cell (1, 0, 0) is global (3, 0, 0), at the right end of the domain
    checkRow(matrix, 1, 3, {3, 2, 7, 11}, {26, -4, -4, -4});
}

TEST_CASE("PressureMatrix prec_nvar 2 blocks")
{
    PressureMatrix matrix(singleRank(3, {3, 3, 3}, true, false), 2);
    CellData data(3, {3, 3, 3});
    data.setLaplacian(lap);
    data.squaredSoundSpeed.assign(data.squaredSoundSpeed.size(), cs2);
    data.rho.assign(data.rho.size(), 2.0);
    data.pressure.assign(data.pressure.size(), 8.0);
    data.dPde.assign(data.dPde.size(), 1.0);
    data.chi.assign(data.chi.size(), 3.0);
    data.dTde.assign(data.dTde.size(), 0.5);
    data.assemble(matrix, dt);

    // coef = 4, e_coef = 4 * 0.5 * (1.5 + 1.5) = 6
    // J_PP = -1, J_Pe = -dPde / rho = -0.5, J_eP = -dt P / rho^2 = -1, J_ee = -1 / rho = -0.5
    REQUIRE(matrix.rowGlobalIndices().size() == 54);
    REQUIRE(matrix.values().size() == 54 * 14);

    // Pressure row of cell (0, 0, 0): (P, P) then (P, e)
    checkRow(
        matrix, 0, 0, {0, 4, 2, 12, 6, 36, 18, 1, 5, 3, 13, 7, 37, 19}, {26, -4, -4, -4, -4, -4, -4, 18, -3, -3, -3, -3, -3, -3});
    // Energy row of cell (0, 0, 0): (e, P) then (e, e)
    checkRow(
        matrix, 1, 1, {0, 4, 2, 12, 6, 36, 18, 1, 5, 3, 13, 7, 37, 19}, {24, -4, -4, -4, -4, -4, -4, 20, -3, -3, -3, -3, -3, -3});
}

namespace
{
    /**
     * @brief Read the set_matrix inputs written by MUSIC for one rank, and return dt.
     */
    double readMusicInputs(const std::string& path,
                           PressureMatrix::Decomposition& d,
                           std::vector<double>& rho,
                           std::vector<double>& asqr,
                           std::array<std::array<std::vector<double>, 2>, 3>& lapCoeffs)
    {
        FILE* file = fopen(path.c_str(), "r");
        REQUIRE(file != nullptr);
        unsigned int prec_nvar = 0;
        double dt              = 0.0;
        long long gs[3], gc[3];
        int periodic[3], refl[6];
        REQUIRE(fscanf(file, " ndim_prec_nvar %u %u", &d.ndim, &prec_nvar) == 2);
        REQUIRE(prec_nvar == 1);
        REQUIRE(fscanf(file, " dt %lf", &dt) == 1);
        REQUIRE(fscanf(file, " local_cells %u %u %u", &d.localCells[0], &d.localCells[1], &d.localCells[2]) == 3);
        REQUIRE(fscanf(file, " global_start %lld %lld %lld", &gs[0], &gs[1], &gs[2]) == 3);
        REQUIRE(fscanf(file, " global_cells %lld %lld %lld", &gc[0], &gc[1], &gc[2]) == 3);
        REQUIRE(fscanf(file, " periodic %d %d %d", &periodic[0], &periodic[1], &periodic[2]) == 3);
        REQUIRE(fscanf(file, " reflective %d %d %d %d %d %d", &refl[0], &refl[1], &refl[2], &refl[3], &refl[4], &refl[5]) == 6);
        for (unsigned int axis = 0; axis < 3; axis++)
        {
            d.globalStart[axis] = gs[axis];
            d.globalCells[axis] = gc[axis];
            d.periodic[axis]    = periodic[axis] != 0;
            d.reflective[axis]  = {refl[axis] != 0, refl[axis + 3] != 0};
        }

        const unsigned int kg   = (d.ndim == 3) ? 1 : 0;
        const size_t numGhosted = size_t(d.localCells[0] + 2) * (d.localCells[1] + 2) * (d.localCells[2] + 2 * kg);
        const size_t numCells   = size_t(d.localCells[0]) * d.localCells[1] * d.localCells[2];
        rho.resize(numGhosted);
        for (double& v : rho)
        {
            REQUIRE(fscanf(file, " %lf", &v) == 1);
        }
        asqr.resize(numCells);
        for (unsigned int axis = 0; axis < 3; axis++)
        {
            lapCoeffs[axis][0].resize(numCells);
            lapCoeffs[axis][1].resize(numCells);
        }
        for (size_t c = 0; c < numCells; c++)
        {
            REQUIRE(fscanf(file,
                           " %lf %lf %lf %lf %lf %lf %lf",
                           &asqr[c],
                           &lapCoeffs[0][0][c],
                           &lapCoeffs[1][0][c],
                           &lapCoeffs[2][0][c],
                           &lapCoeffs[0][1][c],
                           &lapCoeffs[1][1][c],
                           &lapCoeffs[2][1][c]) == 7);
        }
        fclose(file);
        return dt;
    }
}  // namespace

TEST_CASE("PressureMatrix matches the MUSIC matrix of the Roberts dynamo test")
{
    const std::string dataDir = TEST_DATA_DIR;
    for (const auto& [ranks, label] : std::vector<std::pair<int, std::string>> {
             {1, "1rank" },
             {4, "4ranks"}
    })
    {
        for (int rank = 0; rank < ranks; rank++)
        {
            CAPTURE(label, rank);
            PressureMatrix::Decomposition d;
            CellData data(3, {1, 1, 1});
            const double dt = readMusicInputs(dataDir + "/pressure_matrix_roberts_dynamo_" + label + "_rank" +
                                                  std::to_string(rank) + "_inputs.dat",
                                              d,
                                              data.rho,
                                              data.squaredSoundSpeed,
                                              data.laplacian);
            data.ndim       = d.ndim;
            data.nc         = d.localCells;
            data.kg         = (d.ndim == 3) ? 1 : 0;

            PressureMatrix matrix(d);
            data.assemble(matrix, dt);

            // MUSIC matrix rows of this rank (before the D A D scaling): global row, global column, value
            const std::string matrixPath =
                dataDir + "/pressure_solve_roberts_dynamo_" + label + "_rank" + std::to_string(rank) + "_matrix_inputs.dat";
            FILE* file = fopen(matrixPath.c_str(), "r");
            REQUIRE(file != nullptr);
            long long row, column;
            double value;
            size_t entry = 0, localRow = 0;
            while (fscanf(file, "%lld %lld %lf", &row, &column, &value) == 3)
            {
                REQUIRE(entry < matrix.values().size());
                while (matrix.rowOffsets()[localRow + 1] <= entry)
                {
                    localRow++;
                }
                REQUIRE(matrix.rowGlobalIndices()[localRow] == row);
                REQUIRE(matrix.columnGlobalIndices()[entry] == column);
                REQUIRE_THAT(matrix.values()[entry], Catch::Matchers::WithinRel(value, 1e-14));
                entry++;
            }
            fclose(file);
            REQUIRE(entry == matrix.values().size());
        }
    }
}
