#include "PressureSolver.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdio>
#include <string>
#include <Teuchos_DefaultComm.hpp>
#include <Tpetra_Core.hpp>
#include <vector>

namespace
{
    using GO = PressureSolver::GlobalOrdinal;

    /**
     * @brief The local part of a pressure system written by MUSIC for one rank.
     */
    struct LocalSystem
    {
        std::vector<GO> rowGlobalIndices;
        std::vector<size_t> rowOffsets;
        std::vector<GO> columnGlobalIndices;
        std::vector<double> values;
        std::vector<double> rhs;
        std::vector<double> expectedSolution;
    };

    /**
     * @brief Read the matrix rows and the rhs/solution of one rank from the MUSIC data files.
     *
     * The matrix file has one line per entry (global row, global column, value) with the rows in the local order of the
     * vectors. The rhs/solution file has one line per row (global row, dp_rhs, delta_prec).
     */
    LocalSystem readLocalSystem(const std::string& prefix, int rank)
    {
        LocalSystem system;

        const std::string vectorsPath = prefix + "_rhs_solution_rank" + std::to_string(rank) + ".dat";
        FILE* vectorsFile             = fopen(vectorsPath.c_str(), "r");
        REQUIRE(vectorsFile != nullptr);
        long long row;
        double rhs, solution;
        while (fscanf(vectorsFile, "%lld %lf %lf", &row, &rhs, &solution) == 3)
        {
            system.rowGlobalIndices.push_back(GO(row));
            system.rhs.push_back(rhs);
            system.expectedSolution.push_back(solution);
        }
        fclose(vectorsFile);

        const std::string matrixPath = prefix + "_matrix_rank" + std::to_string(rank) + ".dat";
        FILE* matrixFile             = fopen(matrixPath.c_str(), "r");
        REQUIRE(matrixFile != nullptr);
        long long column;
        double value;
        size_t localRow = 0;
        system.rowOffsets.push_back(0);
        while (fscanf(matrixFile, "%lld %lld %lf", &row, &column, &value) == 3)
        {
            // Entries of a row are contiguous, and rows come in the same order as in the rhs/solution file
            if (GO(row) != system.rowGlobalIndices.at(localRow))
            {
                system.rowOffsets.push_back(system.values.size());
                localRow++;
                REQUIRE(GO(row) == system.rowGlobalIndices.at(localRow));
            }
            system.columnGlobalIndices.push_back(GO(column));
            system.values.push_back(value);
        }
        system.rowOffsets.push_back(system.values.size());
        fclose(matrixFile);
        REQUIRE(system.rowOffsets.size() == system.rowGlobalIndices.size() + 1);

        return system;
    }

    /**
     * @brief Read the number of iterations MUSIC needed for the solve.
     */
    int readMusicIterations(const std::string& prefix)
    {
        FILE* infoFile = fopen((prefix + "_info.dat").c_str(), "r");
        REQUIRE(infoFile != nullptr);
        int iterations = -1;
        REQUIRE(fscanf(infoFile, "niter %d", &iterations) == 1);
        fclose(infoFile);
        return iterations;
    }

    /**
     * @brief Global norm of a set of local values.
     */
    double globalNorm(const Teuchos::Comm<int>& comm, const std::vector<double>& values)
    {
        double localSum = 0.0;
        for (double v : values)
        {
            localSum += v * v;
        }
        double globalSum = 0.0;
        Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, localSum, Teuchos::outArg(globalSum));
        return std::sqrt(globalSum);
    }

    /**
     * @brief Relative residual |b - A x| / |b| of the unscaled system.
     */
    double relativeResidual(const LocalSystem& system, const PressureSolver& solver, const std::vector<double>& solution)
    {
        using Vector = PressureSolver::Vector;

        PressureSolver::CrsMatrix matrix(solver.rowMap(), 7);
        for (size_t row = 0; row < system.rowGlobalIndices.size(); row++)
        {
            const size_t begin = system.rowOffsets[row];
            const size_t count = system.rowOffsets[row + 1] - begin;
            matrix.insertGlobalValues(system.rowGlobalIndices[row],
                                      Teuchos::ArrayView<const GO>(&system.columnGlobalIndices[begin], count),
                                      Teuchos::ArrayView<const double>(&system.values[begin], count));
        }
        matrix.fillComplete();

        Vector x(solver.rowMap());
        Vector ax(solver.rowMap());
        {
            auto view = x.getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (size_t i = 0; i < solution.size(); i++)
            {
                view(i, 0) = solution[i];
            }
        }
        matrix.apply(x, ax);

        std::vector<double> residual(solution.size());
        {
            auto view = ax.getLocalViewHost(Tpetra::Access::ReadOnly);
            for (size_t i = 0; i < solution.size(); i++)
            {
                residual[i] = system.rhs[i] - view(i, 0);
            }
        }
        const auto& comm = *solver.rowMap()->getComm();
        return globalNorm(comm, residual) / globalNorm(comm, system.rhs);
    }
}  // namespace

TEST_CASE("PressureSolver rejects an invalid number of dimensions")
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<GO> rows {GO(comm->getRank())};
    PressureSolver::Settings settings {1, 1.0e-6, 10};
    REQUIRE_THROWS_AS(PressureSolver(settings, rows, comm), std::invalid_argument);
}

TEST_CASE("PressureSolver solves a diagonal system in one iteration")
{
    auto comm            = Tpetra::getDefaultComm();
    const int rank       = comm->getRank();
    const int rowsOnRank = 64;

    // Diagonal matrix with entries 1 + global row, rhs equal to 1
    std::vector<GO> rows(rowsOnRank);
    std::vector<size_t> rowOffsets(rowsOnRank + 1);
    std::vector<double> values(rowsOnRank);
    std::vector<double> rhs(rowsOnRank, 1.0);
    for (int i = 0; i < rowsOnRank; i++)
    {
        rows[i]       = GO(rank * rowsOnRank + i);
        rowOffsets[i] = i;
        values[i]     = 1.0 + double(rows[i]);
    }
    rowOffsets[rowsOnRank] = rowsOnRank;

    PressureSolver::Settings settings {3, 1.0e-10, 15};
    PressureSolver solver(settings, rows, comm);
    solver.setMatrix(rowOffsets, rows, values);
    solver.computePreconditioner();

    std::vector<double> solution(rowsOnRank, 0.0);
    PressureSolver::Result result = solver.solve(rhs, solution);

    REQUIRE(result.converged);
    REQUIRE(result.iterations == 1);
    for (int i = 0; i < rowsOnRank; i++)
    {
        REQUIRE_THAT(solution[i], Catch::Matchers::WithinRel(1.0 / values[i], 1.0e-12));
    }
}

TEST_CASE("PressureSolver reproduces the MUSIC pressure solve of the Roberts dynamo test")
{
    auto comm          = Tpetra::getDefaultComm();
    const int numRanks = comm->getSize();
    if (numRanks != 1 && numRanks != 4) { SKIP("MUSIC data is available for 1 and 4 ranks only"); }

    // 16^3 Roberts dynamo test (tests/3dim/roberts in MUSIC), first pressure solve of the first time step
    const std::string prefix =
        std::string(TEST_DATA_DIR) + "/pressure_solve_roberts_dynamo_" + (numRanks == 1 ? "1rank" : "4ranks");
    const LocalSystem system  = readLocalSystem(prefix, comm->getRank());
    const int musicIterations = readMusicIterations(prefix);

    // Settings of the Roberts test namelist (pbp_residual) and MUSIC defaults
    PressureSolver::Settings settings {3, 2.0e-3, 15, 0, -1};
    PressureSolver solver(settings, system.rowGlobalIndices, comm);
    solver.setMatrix(system.rowOffsets, system.columnGlobalIndices, system.values);
    solver.computePreconditioner();

    std::vector<double> solution(system.rhs.size(), 0.0);
    PressureSolver::Result result = solver.solve(system.rhs, solution);

    REQUIRE(result.converged);
    REQUIRE(result.iterations == musicIterations);
    REQUIRE(relativeResidual(system, solver, solution) < settings.pbp_residual);

    // GMRES stops at a tolerance of 2e-3, so the solution only matches MUSIC to about that level
    std::vector<double> difference(solution.size());
    for (size_t i = 0; i < solution.size(); i++)
    {
        difference[i] = solution[i] - system.expectedSolution[i];
    }
    REQUIRE(globalNorm(*comm, difference) / globalNorm(*comm, system.expectedSolution) < settings.pbp_residual);
}
