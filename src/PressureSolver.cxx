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

/*====================================================================================================================*/
namespace
{
    /**
     * @brief Use a Chebyshev smoother on every multigrid level.
     *
     * @param[in,out] mueluSettings - The MueLu settings to change.
     * @param[in] degree - The degree of the Chebyshev polynomial, i.e. the number of smoothing sweeps.
     */
    void setChebyshevSmoother(Teuchos::ParameterList& mueluSettings, const int degree)
    {
        mueluSettings.set("smoother: type", "CHEBYSHEV");
        Teuchos::ParameterList smootherSettings;
        smootherSettings.set("chebyshev: degree", degree);
        mueluSettings.set("smoother: params", smootherSettings);
    }

    /**
     * @brief Solve the coarsest multigrid level directly with KLU2.
     *
     * @param[in,out] mueluSettings - The MueLu settings to change.
     */
    void enableCoarseSolver(Teuchos::ParameterList& mueluSettings)
    { mueluSettings.set("coarse: type", "KLU2"); }

    /**
     * @brief Use no solver on the coarsest multigrid level, only the smoother.
     *
     * @param[in,out] mueluSettings - The MueLu settings to change.
     */
    void disableCoarseSolver(Teuchos::ParameterList& mueluSettings)
    { mueluSettings.set("coarse: type", "none"); }

    /**
     * @brief Rough model of the grid compression ratio of the aggregation, taken from MUSIC.
     *
     * @details Ideally the ratio would be close to 3^ndim; the fudge factor approximates observed ratios in 2D (~3 to
     * ~8) and 3D (~15 to ~25).
     *
     * @param[in] ndim - The number of dimensions of the problem.
     * @param[in] factor - The fudge factor, 1 for the largest ratio, 0.5 for a typical one.
     *
     * @return The number of rows of a level divided by the number of rows of the next coarser level.
     */
    double aggregationRatio(const int ndim, const double factor)
    { return factor * std::pow(3, ndim); }

    /**
     * @brief Minimum number of rows on the coarsest level, taken from MUSIC.
     *
     * @param[in] ndim - The number of dimensions of the problem.
     * @param[in] numRanks - The number of ranks the rows are distributed over.
     * @param[in] maxLevelsOverride - The maximum number of levels set by the user, <= 0 if not set.
     *
     * @return The minimum number of rows, or 1 when maxLevelsOverride sets the number of levels.
     */
    int minCoarseSize(const int ndim, const int numRanks, const int maxLevelsOverride)
    {
        const int minSizeAbsolute = 128;
        const int minSizePerRank  = (ndim == 3 ? 2 : 16);
        if (maxLevelsOverride > 0)
        {
            return 1;  // Let maxLevelsOverride decide the number of levels
        }
        return std::max(minSizeAbsolute, minSizePerRank * numRanks);
    }
}  // namespace

/*====================================================================================================================*/

/*====================================================================================================================*/
PressureSolver::PressureSolver(const Settings& settings,
                               Teuchos::ArrayView<const GlobalOrdinal> rowGlobalIndices,
                               const Teuchos::RCP<const Teuchos::Comm<int>>& comm) :
    m_settings(settings)
{
    if (m_settings.ndim != TWO && m_settings.ndim != THREE)
    {
        throw std::invalid_argument("PressureSolver: ndim must be 2 or 3, found " + std::to_string(m_settings.ndim));
    }

    m_rowMap =
        Teuchos::rcp(new Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), rowGlobalIndices, /*indexBase=*/0, comm));
    m_diag    = Teuchos::rcp(new Vector(m_rowMap));
    m_yScaled = Teuchos::rcp(new Vector(m_rowMap));

    // GMRES settings
    m_belosSettings.set("Implicit Residual Scaling", "Norm of RHS");
    m_belosSettings.set("Explicit Residual Scaling", "Norm of RHS");
    m_belosSettings.set("Num Blocks", 40);
    m_belosSettings.set("Maximum Restarts", 8);
    m_belosSettings.set("Block Size", 1);
    m_belosSettings.set("Use Single Reduction", true);
    m_belosSettings.set("Maximum Iterations", m_settings.pbp_max_iter);
    m_belosSettings.set("Convergence Tolerance", m_settings.pbp_residual);
    m_belosSettings.set("Output Frequency", 1);
    m_belosSettings.set("Verbosity", Belos::Errors);

    // Multigrid settings, starting from the MueLu defaults for a Poisson problem, since the pressure problem becomes a
    // Poisson problem as dt goes to infinity. 2D problems have less work per rank and are more sensitive to latency, so
    // in 2D the work on the coarse levels is kept small: more smoothing sweeps and, by default, no coarse solver.
    const int chebyshevDegree2D = 3;  // More aggressive smoothing in 2D
    const int chebyshevDegree3D = 1;
    m_mueluSettings.set("cycle type", "V");
    if (m_settings.ndim == TWO)
    {
        m_mueluSettings.set("problem: type", "Poisson-2D");
        setChebyshevSmoother(m_mueluSettings, chebyshevDegree2D);
    }
    else
    {
        m_mueluSettings.set("problem: type", "Poisson-3D");
        setChebyshevSmoother(m_mueluSettings, chebyshevDegree3D);
    }

    // The heuristic uses a coarse solver in 3D only; the user can still ask for one in 2D
    const bool useCoarseSolver =
        (m_settings.use_coarse_solver == HEURISTIC) ? (m_settings.ndim == THREE) : (m_settings.use_coarse_solver == KLU2);
    if (useCoarseSolver) { enableCoarseSolver(m_mueluSettings); }
    else
    {
        disableCoarseSolver(m_mueluSettings);
    }

    m_mueluSettings.set("number of equations", 1);
    setHierarchySettings();

    // Uncoupled aggregation with serial colouring, and a drop tolerance to keep the coarse operators sparse
    m_mueluSettings.set("aggregation: type", "uncoupled");
    m_mueluSettings.set("aggregation: coloring algorithm", "serial");
    m_mueluSettings.set("aggregation: drop tol", 0.02);
    m_mueluSettings.set("verbosity", "none");
}

/*====================================================================================================================*/

/*====================================================================================================================*/
void PressureSolver::setHierarchySettings()
{
    const int ndim            = m_settings.ndim;
    const int numRanks        = m_rowMap->getComm()->getSize();
    const double nrows        = double(m_rowMap->getGlobalNumElements());
    const int minSize         = minCoarseSize(ndim, numRanks, m_settings.max_levels_override);
    const double maxRatio     = aggregationRatio(ndim, 1.0);
    const double typicalRatio = aggregationRatio(ndim, 0.5);

    // MueLu builds levels until one has at most "coarse: max size" rows; scaling the minimum size by the largest
    // compression ratio keeps the coarsest level above the minimum size
    m_mueluSettings.set("coarse: max size", int(minSize * maxRatio));

    int maxLevels = m_settings.max_levels_override;
    if (maxLevels <= 0)
    {
        // Level L has about nrows / typicalRatio^L rows
        const double ratio = std::max(1.0, nrows / double(minSize));
        maxLevels          = int(std::ceil(std::log(ratio) / std::log(typicalRatio))) + 1;
    }
    m_mueluSettings.set("max levels", maxLevels);
}

/*====================================================================================================================*/

/*====================================================================================================================*/
void PressureSolver::setMatrix(Teuchos::ArrayView<const size_t> rowOffsets,
                               Teuchos::ArrayView<const GlobalOrdinal> columnGlobalIndices,
                               Teuchos::ArrayView<const Scalar> values)
{
    const size_t numLocalRows = m_rowMap->getLocalNumElements();
    if (size_t(rowOffsets.size()) != numLocalRows + 1)
    {
        throw std::invalid_argument("PressureSolver::setMatrix: rowOffsets must have numLocalRows + 1 entries");
    }
    if (columnGlobalIndices.size() != values.size() || size_t(values.size()) != rowOffsets[numLocalRows])
    {
        throw std::invalid_argument("PressureSolver::setMatrix: column indices and values do not match rowOffsets");
    }

    // On the first call, create the matrix with room for the longest row; the entries inserted below then fix the
    // sparsity pattern. Later calls only replace the values, so the matrix is reopened for filling.
    const bool firstCall = m_matrix.is_null();
    if (firstCall)
    {
        size_t maxEntriesPerRow = 0;
        for (size_t row = 0; row < numLocalRows; row++)
        {
            maxEntriesPerRow = std::max(maxEntriesPerRow, rowOffsets[row + 1] - rowOffsets[row]);
        }
        m_matrix = Teuchos::rcp(new CrsMatrix(m_rowMap, maxEntriesPerRow));
    }
    else
    {
        m_matrix->resumeFill();
    }

    // Copy each local row into the matrix. Row `row` of the arrays is the row of global index
    // rowGlobalIndices[row] given to the constructor.
    for (size_t row = 0; row < numLocalRows; row++)
    {
        const size_t begin            = rowOffsets[row];
        const size_t numEntries       = rowOffsets[row + 1] - begin;
        const GlobalOrdinal globalRow = m_rowMap->getGlobalElement(LocalOrdinal(row));
        if (firstCall)
        {
            m_matrix->insertGlobalValues(globalRow, columnGlobalIndices(begin, numEntries), values(begin, numEntries));
        }
        else
        {
            m_matrix->replaceGlobalValues(globalRow, columnGlobalIndices(begin, numEntries), values(begin, numEntries));
        }
    }

    // Finish the matrix: works out which columns are on other ranks and sets up the communication for the products
    m_matrix->fillComplete();

    // Symmetric diagonal scaling: solve M' X' = Y' with M' = D M D, Y' = D Y and X = D X', where D is the diagonal matrix
    // of the inverse square roots of the row one norms of M. M' keeps the symmetry of M. Here D is computed and M is
    // scaled in place; Y and X are scaled in solve().
    {
        auto equilibration = Tpetra::computeRowOneNorms(*m_matrix);
        auto rowNorms      = equilibration.rowNorms;
        auto diag          = m_diag->getLocalViewDevice(Tpetra::Access::OverwriteAll);
        Kokkos::parallel_for(
            "PressureSolver::setDiag", m_diag->getLocalLength(), KOKKOS_LAMBDA(const size_t i) {
                diag(i, 0) = Kokkos::rsqrt(rowNorms(i));
            });
    }
    m_matrix->leftScale(*m_diag);
    m_matrix->rightScale(*m_diag);
}

/*====================================================================================================================*/

/*====================================================================================================================*/
void PressureSolver::computePreconditioner()
{
    if (m_matrix.is_null()) { throw std::logic_error("PressureSolver::computePreconditioner: setMatrix must be called first"); }
    Teuchos::RCP<Operator> op = m_matrix;
    m_preconditioner          = MueLu::CreateTpetraPreconditioner(op, m_mueluSettings);
}

/*====================================================================================================================*/

/*====================================================================================================================*/
PressureSolver::Result PressureSolver::solve(Teuchos::ArrayView<const Scalar> rhs, Teuchos::ArrayView<Scalar> solution)
{
    if (m_preconditioner.is_null())
    {
        throw std::logic_error("PressureSolver::solve: computePreconditioner must be called first");
    }
    const size_t numLocalRows = m_rowMap->getLocalNumElements();
    if (size_t(rhs.size()) != numLocalRows || size_t(solution.size()) != numLocalRows)
    {
        throw std::invalid_argument("PressureSolver::solve: rhs and solution must have one entry per local row");
    }

    // Borrow the caller's arrays rather than copying through them
    HostArrayVector y(m_rowMap, rhs.getRawPtr());
    HostArrayVector x(m_rowMap, solution.getRawPtr(), HostArrayAccess::WriteOnly);
    Vector& xVector = x.vec();

    // Y' = D Y
    m_yScaled->elementWiseMultiply(1.0, *m_diag, y.vec(), 0.0);

    // Solve M' X' = Y' from a zero initial guess. xVector holds X'; the rescaling from X' to X is done in place at the
    // end to avoid copies.
    xVector.putScalar(0.0);
    Belos::LinearProblem<Scalar, MultiVector, Operator> problem(
        Teuchos::RCP<Operator>(m_matrix), Teuchos::rcpFromRef(xVector), m_yScaled);
    problem.setRightPrec(Teuchos::RCP<Operator>(m_preconditioner));
    problem.setProblem();

    Belos::PseudoBlockGmresSolMgr<Scalar, MultiVector, Operator> solver(Teuchos::rcpFromRef(problem),
                                                                        Teuchos::rcpFromRef(m_belosSettings));
    const Belos::ReturnType ret = solver.solve();

    // X = D X' in place. Aliasing X with an operand is safe: the kernel is elementwise and a zero destination scaling
    // never reads X. The destructor of x brings X back into solution.
    xVector.elementWiseMultiply(1.0, *m_diag, xVector, 0.0);

    return Result {solver.getNumIters(), solver.achievedTol(), ret == Belos::Converged};
}

/*====================================================================================================================*/
