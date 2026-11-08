#include "PressureSolver.h"

#include "HostArrayVector.h"
#include <algorithm>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <stdexcept>
#include <string>
#include <Tpetra_computeRowAndColumnOneNorms.hpp>

void PressureSolver::setChebyshevSmoother(Teuchos::ParameterList& muelu_settings, const int degree)
{
    muelu_settings.set("smoother: type", "CHEBYSHEV");
    Teuchos::ParameterList smoother_settings;
    smoother_settings.set("chebyshev: degree", degree);
    muelu_settings.set("smoother: params", smoother_settings);
}

void PressureSolver::enableCoarseSolver(Teuchos::ParameterList& muelu_settings)
{ muelu_settings.set("coarse: type", "KLU2"); }

void PressureSolver::disableCoarseSolver(Teuchos::ParameterList& muelu_settings)
{ muelu_settings.set("coarse: type", "none"); }

double PressureSolver::aggregationRatio(const int ndim, const double factor)
{ return factor * std::pow(3, ndim); }

int PressureSolver::minCoarseSize(const int ndim, const int num_ranks, const int max_levels_override)
{
    const int min_size_absolute = 128;
    const int min_size_per_rank = (ndim == 3 ? 2 : 16);
    if (max_levels_override > 0)
    {
        return 1;  // Let max_levels_override decide the number of levels
    }
    return std::max(min_size_absolute, min_size_per_rank * num_ranks);
}

PressureSolver::PressureSolver(const Settings& settings,
                               Teuchos::ArrayView<const GlobalOrdinal> row_global_indices,
                               const Teuchos::RCP<const Teuchos::Comm<int>>& comm) :
    m_settings(settings)
{
    if (m_settings.ndim != TWO_D && m_settings.ndim != THREE_D)
    {
        throw std::invalid_argument("PressureSolver: ndim must be 2 or 3, found " + std::to_string(m_settings.ndim));
    }

    m_row_map = Teuchos::rcp(
        new Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), row_global_indices, /*indexBase=*/0, comm));
    m_diag     = Teuchos::rcp(new Vector(m_row_map));
    m_y_scaled = Teuchos::rcp(new Vector(m_row_map));

    // GMRES settings
    m_belos_settings.set("Implicit Residual Scaling", "Norm of RHS");
    m_belos_settings.set("Explicit Residual Scaling", "Norm of RHS");
    m_belos_settings.set("Num Blocks", m_settings.gmres_num_blocks);
    m_belos_settings.set("Maximum Restarts", m_settings.gmres_max_restarts);
    m_belos_settings.set("Block Size", 1);
    m_belos_settings.set("Use Single Reduction", true);
    m_belos_settings.set("Maximum Iterations", m_settings.pbp_max_iter);
    m_belos_settings.set("Convergence Tolerance", m_settings.pbp_residual);
    m_belos_settings.set("Output Frequency", 1);
    m_belos_settings.set("Verbosity", Belos::Errors);

    // Multigrid settings, starting from the MueLu defaults for a Poisson problem, since the pressure problem becomes a
    // Poisson problem as dt goes to infinity. 2D problems have less work per rank and are more sensitive to latency, so
    // in 2D the work on the coarse levels is kept small: more smoothing sweeps and, by default, no coarse solver.
    m_muelu_settings.set("cycle type", "V");
    if (m_settings.ndim == TWO_D)
    {
        m_muelu_settings.set("problem: type", "Poisson-2D");
        setChebyshevSmoother(m_muelu_settings, m_settings.chebyshev_degree_2d);
    }
    else
    {
        m_muelu_settings.set("problem: type", "Poisson-3D");
        setChebyshevSmoother(m_muelu_settings, m_settings.chebyshev_degree_3d);
    }

    // The heuristic uses a coarse solver in 3D only; the user can still ask for one in 2D
    const bool use_coarse_solver =
        (m_settings.use_coarse_solver == HEURISTIC) ? (m_settings.ndim == THREE_D) : (m_settings.use_coarse_solver == KLU2);
    if (use_coarse_solver) { enableCoarseSolver(m_muelu_settings); }
    else
    {
        disableCoarseSolver(m_muelu_settings);
    }

    m_muelu_settings.set("number of equations", 1);
    setHierarchySettings();

    // Uncoupled aggregation with serial colouring, and a drop tolerance to keep the coarse operators sparse
    m_muelu_settings.set("aggregation: type", "uncoupled");
    m_muelu_settings.set("aggregation: coloring algorithm", "serial");
    m_muelu_settings.set("aggregation: drop tol", m_settings.aggregation_drop_tol);
    m_muelu_settings.set("verbosity", "none");
}

void PressureSolver::setHierarchySettings()
{
    const int ndim             = m_settings.ndim;
    const int num_ranks        = m_row_map->getComm()->getSize();
    const double num_rows      = double(m_row_map->getGlobalNumElements());
    const int min_size         = minCoarseSize(ndim, num_ranks, m_settings.max_levels_override);
    const double max_ratio     = aggregationRatio(ndim, 1.0);
    const double typical_ratio = aggregationRatio(ndim, 0.5);

    // MueLu builds levels until one has at most "coarse: max size" rows; scaling the minimum size by the largest
    // compression ratio keeps the coarsest level above the minimum size
    m_muelu_settings.set("coarse: max size", int(min_size * max_ratio));

    int max_levels = m_settings.max_levels_override;
    if (max_levels <= 0)
    {
        // Level L has about num_rows / typical_ratio^L rows
        const double ratio = std::max(1.0, num_rows / double(min_size));
        max_levels         = int(std::ceil(std::log(ratio) / std::log(typical_ratio))) + 1;
    }
    m_muelu_settings.set("max levels", max_levels);
}

void PressureSolver::setMatrix(Teuchos::ArrayView<const size_t> row_offsets,
                               Teuchos::ArrayView<const GlobalOrdinal> column_global_indices,
                               Teuchos::ArrayView<const Scalar> values)
{
    const size_t num_local_rows = m_row_map->getLocalNumElements();
    if (size_t(row_offsets.size()) != num_local_rows + 1)
    {
        throw std::invalid_argument("PressureSolver::setMatrix: row_offsets must have one entry per local row plus one");
    }
    if (column_global_indices.size() != values.size() || size_t(values.size()) != row_offsets[num_local_rows])
    {
        throw std::invalid_argument("PressureSolver::setMatrix: column indices and values do not match row_offsets");
    }

    // On the first call, create the matrix with room for the longest row; the entries inserted below then fix the
    // sparsity pattern. Later calls only replace the values, so the matrix is reopened for filling.
    const bool first_call = m_matrix.is_null();
    if (first_call)
    {
        size_t max_entries_per_row = 0;
        for (size_t row = 0; row < num_local_rows; row++)
        {
            max_entries_per_row = std::max(max_entries_per_row, row_offsets[row + 1] - row_offsets[row]);
        }
        m_matrix = Teuchos::rcp(new CrsMatrix(m_row_map, max_entries_per_row));
    }
    else
    {
        m_matrix->resumeFill();
    }

    // Copy each local row into the matrix. Row `row` of the arrays is the row of global index
    // row_global_indices[row] given to the constructor.
    for (size_t row = 0; row < num_local_rows; row++)
    {
        const size_t begin             = row_offsets[row];
        const size_t num_entries       = row_offsets[row + 1] - begin;
        const GlobalOrdinal global_row = m_row_map->getGlobalElement(LocalOrdinal(row));
        if (first_call)
        {
            m_matrix->insertGlobalValues(global_row, column_global_indices(begin, num_entries), values(begin, num_entries));
        }
        else
        {
            m_matrix->replaceGlobalValues(global_row, column_global_indices(begin, num_entries), values(begin, num_entries));
        }
    }

    // Finish the matrix: works out which columns are on other ranks and sets up the communication for the products
    m_matrix->fillComplete();

    // Symmetric diagonal scaling: solve M' X' = Y' with M' = D M D, Y' = D Y and X = D X', where D is the diagonal matrix
    // of the inverse square roots of the row one norms of M. M' keeps the symmetry of M. Here D is computed and M is
    // scaled in place; Y and X are scaled in solve().
    {
        auto equilibration = Tpetra::computeRowOneNorms(*m_matrix);
        auto row_norms     = equilibration.rowNorms;
        auto diag          = m_diag->getLocalViewDevice(Tpetra::Access::OverwriteAll);
        Kokkos::parallel_for(
            "PressureSolver::setDiag", m_diag->getLocalLength(), KOKKOS_LAMBDA(const size_t i) {
                diag(i, 0) = Kokkos::rsqrt(row_norms(i));
            });
    }
    m_matrix->leftScale(*m_diag);
    m_matrix->rightScale(*m_diag);
}

void PressureSolver::computePreconditioner()
{
    if (m_matrix.is_null()) { throw std::logic_error("PressureSolver::computePreconditioner: setMatrix must be called first"); }
    Teuchos::RCP<Operator> op = m_matrix;
    m_preconditioner          = MueLu::CreateTpetraPreconditioner(op, m_muelu_settings);
}

PressureSolver::Result PressureSolver::solve(Teuchos::ArrayView<const Scalar> rhs, Teuchos::ArrayView<Scalar> solution)
{
    if (m_preconditioner.is_null())
    {
        throw std::logic_error("PressureSolver::solve: computePreconditioner must be called first");
    }
    const size_t num_local_rows = m_row_map->getLocalNumElements();
    if (size_t(rhs.size()) != num_local_rows || size_t(solution.size()) != num_local_rows)
    {
        throw std::invalid_argument("PressureSolver::solve: rhs and solution must have one entry per local row");
    }

    // Borrow the caller's arrays rather than copying through them
    HostArrayVector y(m_row_map, rhs.getRawPtr());
    HostArrayVector x(m_row_map, solution.getRawPtr(), HostArrayAccess::WriteOnly);
    Vector& x_vector = x.vec();

    // Y' = D Y
    m_y_scaled->elementWiseMultiply(1.0, *m_diag, y.vec(), 0.0);

    // Solve M' X' = Y' from a zero initial guess. x_vector holds X'; the rescaling from X' to X is done in place at the
    // end to avoid copies.
    x_vector.putScalar(0.0);
    Belos::LinearProblem<Scalar, MultiVector, Operator> problem(
        Teuchos::RCP<Operator>(m_matrix), Teuchos::rcpFromRef(x_vector), m_y_scaled);
    problem.setRightPrec(Teuchos::RCP<Operator>(m_preconditioner));
    problem.setProblem();

    Belos::PseudoBlockGmresSolMgr<Scalar, MultiVector, Operator> solver(Teuchos::rcpFromRef(problem),
                                                                        Teuchos::rcpFromRef(m_belos_settings));
    const Belos::ReturnType ret = solver.solve();

    // X = D X' in place. Aliasing X with an operand is safe: the kernel is elementwise and a zero destination scaling
    // never reads X. The destructor of x brings X back into solution.
    x_vector.elementWiseMultiply(1.0, *m_diag, x_vector, 0.0);

    return Result {solver.getNumIters(), solver.achievedTol(), ret == Belos::Converged};
}
