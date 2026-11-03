#include "PressureSolver.h"

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
    void setChebyshevSmoother(Teuchos::ParameterList& mueluSettings, int degree)
    {
        mueluSettings.set("smoother: type", "CHEBYSHEV");
        Teuchos::ParameterList smootherSettings;
        smootherSettings.set("chebyshev: degree", degree);
        mueluSettings.set("smoother: params", smootherSettings);
    }

    void enableCoarseSolver(Teuchos::ParameterList& mueluSettings)
    { mueluSettings.set("coarse: type", "KLU2"); }

    void disableCoarseSolver(Teuchos::ParameterList& mueluSettings)
    { mueluSettings.set("coarse: type", "none"); }

    /**
     * @brief Rough model of the grid compression ratio of the aggregation, taken from MUSIC.
     *
     * Ideally the ratio would be close to 3^ndim; the fudge factor approximates observed ratios in 2D (~3 to ~8) and 3D
     * (~15 to ~25).
     */
    double aggregationRatio(int ndim, double factor)
    { return factor * std::pow(3, ndim); }

    /**
     * @brief Minimum number of rows on the coarsest level, taken from MUSIC.
     */
    int minCoarseSize(int ndim, int numRanks, int maxLevelsOverride)
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
    if (m_settings.ndim != 2 && m_settings.ndim != 3)
    {
        throw std::invalid_argument("PressureSolver: ndim must be 2 or 3, found " + std::to_string(m_settings.ndim));
    }

    m_rowMap =
        Teuchos::rcp(new Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), rowGlobalIndices, /*indexBase=*/0, comm));
    m_diag = Teuchos::rcp(new Vector(m_rowMap));

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
    // Poisson problem as dt goes to infinity
    if (m_settings.ndim == 2)
    {
        m_mueluSettings.set("problem: type", "Poisson-2D");
        setChebyshevSmoother(m_mueluSettings, 3);
    }
    else
    {
        m_mueluSettings.set("problem: type", "Poisson-3D");
        setChebyshevSmoother(m_mueluSettings, 1);
    }
    m_mueluSettings.set("cycle type", "V");

    if (m_settings.use_coarse_solver < 0)
    {
        // Heuristic: no coarse solver in 2D, coarse solver in 3D
        if (m_settings.ndim == 2) { disableCoarseSolver(m_mueluSettings); }
        else
        {
            enableCoarseSolver(m_mueluSettings);
        }
    }
    else if (m_settings.use_coarse_solver == 0) { disableCoarseSolver(m_mueluSettings); }
    else
    {
        enableCoarseSolver(m_mueluSettings);
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

    for (size_t row = 0; row < numLocalRows; row++)
    {
        const size_t begin       = rowOffsets[row];
        const size_t numEntries  = rowOffsets[row + 1] - begin;
        const GlobalOrdinal grow = m_rowMap->getGlobalElement(LocalOrdinal(row));
        if (firstCall) { m_matrix->insertGlobalValues(grow, columnGlobalIndices(begin, numEntries), values(begin, numEntries)); }
        else
        {
            m_matrix->replaceGlobalValues(grow, columnGlobalIndices(begin, numEntries), values(begin, numEntries));
        }
    }
    m_matrix->fillComplete();

    // Symmetric diagonal scaling: solve A' X' = Y' with A' = D A D, Y' = D Y and X = D X', where D is the inverse square
    // root of the row one norms. A' keeps the symmetry of A.
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

    Vector y(m_rowMap);
    {
        auto view = y.getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t i = 0; i < numLocalRows; i++)
        {
            view(i, 0) = rhs[i];
        }
    }

    // Y' = D Y, then solve A' X' = Y' from a zero initial guess
    Vector yScaled(m_rowMap);
    yScaled.elementWiseMultiply(1.0, *m_diag, y, 0.0);
    auto x = Teuchos::rcp(new Vector(m_rowMap));
    x->putScalar(0.0);

    Belos::LinearProblem<Scalar, MultiVector, Operator> problem(
        Teuchos::RCP<Operator>(m_matrix), x, Teuchos::rcpFromRef(yScaled));
    problem.setRightPrec(Teuchos::RCP<Operator>(m_preconditioner));
    problem.setProblem();

    Belos::PseudoBlockGmresSolMgr<Scalar, MultiVector, Operator> solver(Teuchos::rcpFromRef(problem),
                                                                        Teuchos::rcpFromRef(m_belosSettings));
    const Belos::ReturnType ret = solver.solve();

    // X = D X'
    x->elementWiseMultiply(1.0, *m_diag, *x, 0.0);
    {
        auto view = x->getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i < numLocalRows; i++)
        {
            solution[i] = view(i, 0);
        }
    }

    return Result {solver.getNumIters(), solver.achievedTol(), ret == Belos::Converged};
}
/*====================================================================================================================*/