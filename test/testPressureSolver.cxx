#include "PressureSolver.h"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
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
        std::vector<GO> row_global_indices;
        std::vector<size_t> row_offsets;
        std::vector<GO> column_global_indices;
        std::vector<double> values;
        std::vector<double> rhs;
        std::vector<double> expected_solution;
    };

    /**
     * @brief Read the matrix rows, the rhs and the expected solution of one rank from the MUSIC data files.
     *
     * @details The files of rank r are `<prefix>_rank<r>_matrix_inputs.dat`, one line per entry (global row, global
     * column, value), `<prefix>_rank<r>_rhs_inputs.dat`, one line per row (global row, dp_rhs), and
     * `<prefix>_rank<r>_expected_outputs.dat`, one line per row (global row, delta_prec). All three list the rows in the
     * local order of the vectors.
     *
     * @param[in] prefix - The path of the data files up to `_rank<r>`.
     * @param[in] rank - The rank to read.
     * @param[in] expected_rows - The number of rows this rank should have.
     *
     * @return The local system of the rank.
     */
    LocalSystem readLocalSystem(const std::string& prefix, const int rank, const size_t expected_rows)
    {
        LocalSystem system;
        const std::string rank_prefix = prefix + "_rank" + std::to_string(rank);
        long long row;

        const std::string rhs_path = rank_prefix + "_rhs_inputs.dat";
        FILE* rhs_file             = fopen(rhs_path.c_str(), "r");
        REQUIRE(rhs_file != nullptr);
        double rhs;
        while (fscanf(rhs_file, "%lld %lf", &row, &rhs) == 2)
        {
            system.row_global_indices.push_back(GO(row));
            system.rhs.push_back(rhs);
        }
        fclose(rhs_file);
        REQUIRE(system.row_global_indices.size() == expected_rows);

        const std::string solution_path = rank_prefix + "_expected_outputs.dat";
        FILE* solution_file             = fopen(solution_path.c_str(), "r");
        REQUIRE(solution_file != nullptr);
        double solution;
        while (fscanf(solution_file, "%lld %lf", &row, &solution) == 2)
        {
            // Same rows in the same order as in the rhs file
            REQUIRE(system.expected_solution.size() < expected_rows);
            REQUIRE(GO(row) == system.row_global_indices[system.expected_solution.size()]);
            system.expected_solution.push_back(solution);
        }
        fclose(solution_file);
        REQUIRE(system.expected_solution.size() == expected_rows);

        const std::string matrix_path = rank_prefix + "_matrix_inputs.dat";
        FILE* matrix_file             = fopen(matrix_path.c_str(), "r");
        REQUIRE(matrix_file != nullptr);
        long long column;
        double value;
        size_t local_row = 0;
        system.row_offsets.push_back(0);
        while (fscanf(matrix_file, "%lld %lld %lf", &row, &column, &value) == 3)
        {
            // Entries of a row are contiguous, and rows come in the same order as in the rhs file
            if (GO(row) != system.row_global_indices.at(local_row))
            {
                system.row_offsets.push_back(system.values.size());
                local_row++;
                REQUIRE(GO(row) == system.row_global_indices.at(local_row));
            }
            system.column_global_indices.push_back(GO(column));
            system.values.push_back(value);
        }
        system.row_offsets.push_back(system.values.size());
        fclose(matrix_file);
        REQUIRE(system.row_offsets.size() == expected_rows + 1);

        return system;
    }

    /**
     * @brief Read the expected number of iterations of the solve (from MUSIC), from `<prefix>_expected_outputs_info.dat`.
     */
    int readExpectedIterations(const std::string& prefix)
    {
        FILE* info_file = fopen((prefix + "_expected_outputs_info.dat").c_str(), "r");
        REQUIRE(info_file != nullptr);
        int iterations = -1;
        REQUIRE(fscanf(info_file, "niter %d", &iterations) == 1);
        fclose(info_file);
        return iterations;
    }

    /**
     * @brief Global norm of a set of local values.
     */
    double globalNorm(const Teuchos::Comm<int>& comm, const std::vector<double>& values)
    {
        double local_sum = 0.0;
        for (double v : values)
        {
            local_sum += v * v;
        }
        double global_sum = 0.0;
        Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, local_sum, Teuchos::outArg(global_sum));
        return std::sqrt(global_sum);
    }

    /**
     * @brief Relative residual |b - A x| / |b| of the system as MUSIC wrote it, for a solution x found by the solver.
     *
     * This is a check of the solver that does not rely on it: the solver only holds the scaled matrix D A D, so the
     * matrix A is built again here from the MUSIC data, then A x is computed with Tpetra and the norms are reduced over
     * all ranks.
     *
     * GMRES stops on the residual of the scaled system, |D b - D A D x'| / |D b| with x = D x'. When every row of A has
     * the same one norm, as in the Roberts test (uniform periodic grid, constant density and sound speed), D is a
     * constant times the identity and the two relative residuals are equal, so this one must also be below the GMRES
     * tolerance. For other systems the two differ.
     *
     * @param[in] system - The matrix rows and rhs of this rank.
     * @param[in] solver - The solver, for its row map (the distribution of the rows over the ranks).
     * @param[in] solution - The solution of this rank, in the local order of the rows.
     *
     * @return The relative residual over all ranks.
     */
    double relativeResidual(const LocalSystem& system, const PressureSolver& solver, const std::vector<double>& solution)
    {
        using Vector = PressureSolver::Vector;

        // Rebuild the unscaled matrix A with the rows of this rank. The column indices are global; fillComplete()
        // works out which columns belong to other ranks and sets up the communication A x needs for them.
        size_t max_entries_per_row = 0;
        for (size_t row = 0; row < system.row_global_indices.size(); row++)
        {
            max_entries_per_row = std::max(max_entries_per_row, system.row_offsets[row + 1] - system.row_offsets[row]);
        }
        PressureSolver::CrsMatrix matrix(solver.rowMap(), max_entries_per_row);
        for (size_t row = 0; row < system.row_global_indices.size(); row++)
        {
            const size_t begin = system.row_offsets[row];
            const size_t count = system.row_offsets[row + 1] - begin;
            matrix.insertGlobalValues(system.row_global_indices[row],
                                      Teuchos::ArrayView<const GO>(&system.column_global_indices[begin], count),
                                      Teuchos::ArrayView<const double>(&system.values[begin], count));
        }
        matrix.fillComplete();

        // Put x in a Tpetra vector with the same row map, so that local entry i is global row row_global_indices[i]
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

        // Local part of r = b - A x; the norms of r and b are then summed over all ranks
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
    PressureSolver::Settings settings {static_cast<PressureSolver::Dimensions>(1), 1.0e-6, 10};
    REQUIRE_THROWS_WITH(PressureSolver(settings, rows, comm),
                        Catch::Matchers::ContainsSubstring("PressureSolver: ndim must be 2 or 3, found 1"));
}

TEST_CASE("PressureSolver solves a diagonal system in one iteration")
{
    auto comm              = Tpetra::getDefaultComm();
    const int rank         = comm->getRank();
    const int rows_on_rank = 64;

    // Diagonal matrix with entries 1 + global row, rhs equal to 1
    std::vector<GO> rows(rows_on_rank);
    std::vector<size_t> row_offsets(rows_on_rank + 1);
    std::vector<double> values(rows_on_rank);
    std::vector<double> rhs(rows_on_rank, 1.0);
    for (int i = 0; i < rows_on_rank; i++)
    {
        rows[i]        = GO(rank * rows_on_rank + i);
        row_offsets[i] = i;
        values[i]      = 1.0 + double(rows[i]);
    }
    row_offsets[rows_on_rank] = rows_on_rank;

    PressureSolver::Settings settings {PressureSolver::THREE_D, 1.0e-10, 15};
    PressureSolver solver(settings, rows, comm);
    solver.setMatrix(row_offsets, rows, values);
    solver.computePreconditioner();

    std::vector<double> solution(rows_on_rank, 0.0);
    PressureSolver::Result result = solver.solve(rhs, solution);

    REQUIRE(result.converged);
    REQUIRE(result.iterations == 1);
    for (int i = 0; i < rows_on_rank; i++)
    {
        REQUIRE_THAT(solution[i], Catch::Matchers::WithinRel(1.0 / values[i], 1.0e-14));
    }
}

TEST_CASE("PressureSolver reproduces the MUSIC pressure solve of the Roberts dynamo test")
{
    auto comm           = Tpetra::getDefaultComm();
    const int num_ranks = comm->getSize();
    if (num_ranks != 1 && num_ranks != 4) { SKIP("MUSIC data is available for 1 and 4 ranks only"); }

    // 16^3 Roberts dynamo test (tests/3dim/roberts in MUSIC), first pressure solve of the first time step
    const std::string prefix =
        std::string(TEST_DATA_DIR) + "/pressure_solve_roberts_dynamo_" + (num_ranks == 1 ? "1rank" : "4ranks");
    const size_t global_rows      = 16 * 16 * 16;  // prec_nvar = 1, one row per cell
    const LocalSystem system      = readLocalSystem(prefix, comm->getRank(), global_rows / size_t(num_ranks));
    const int expected_iterations = readExpectedIterations(prefix);

    // Settings of the Roberts test namelist (pbp_residual) and MUSIC defaults
    PressureSolver::Settings settings {PressureSolver::THREE_D, 2.0e-3, 15, 0, PressureSolver::HEURISTIC};
    PressureSolver solver(settings, system.row_global_indices, comm);
    solver.setMatrix(system.row_offsets, system.column_global_indices, system.values);
    solver.computePreconditioner();

    std::vector<double> solution(system.rhs.size(), 0.0);
    PressureSolver::Result result = solver.solve(system.rhs, solution);

    REQUIRE(result.converged);
    REQUIRE(result.iterations == expected_iterations);
    REQUIRE(relativeResidual(system, solver, solution) < settings.pbp_residual);

    // GMRES stops at a tolerance of 2e-3, so the solution only matches the expected one to about that level
    std::vector<double> difference(solution.size());
    for (size_t i = 0; i < solution.size(); i++)
    {
        difference[i] = solution[i] - system.expected_solution[i];
    }
    REQUIRE(globalNorm(*comm, difference) / globalNorm(*comm, system.expected_solution) < settings.pbp_residual);
}
