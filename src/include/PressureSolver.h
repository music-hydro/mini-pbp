/**
 * @class PressureSolver
 *
 * @brief Solves the preconditioner pressure system M dP = dp_rhs with Trilinos: symmetric diagonal scaling, Belos
 * pseudo block GMRES and a MueLu multigrid right preconditioner.
 *
 * @details In the physics based preconditioner, the hydro correction is reduced to a parabolic problem for the pressure
 * correction dP. For prec_nvar = 1, M is the 7 point (3D) or 5 point (2D) operator with diagonal
 * 1/dt + dt c^2 sum(coef) and off diagonal -dt c^2 coef for each neighbour, where coef is the pressure coefficients
 * times the neighbour link weights. It acts as a Poisson operator as dt grows, which is why the multigrid starts from
 * the MueLu Poisson defaults. The solution dP is then used to recover the rest of the correction.
 *
 * The matrix is given as CSR arrays with global column indices. It can be set again every Newton iteration, while the
 * multigrid hierarchy is only rebuilt when computePreconditioner() is called (once per time step, on the first Newton
 * iteration).
 *
 * Before the solve the system is scaled: instead of M X = Y, it solves M' X' = Y' with M' = D M D, Y' = D Y and
 * X = D X', where D is the diagonal matrix of the inverse square roots of the row one norms of M. Scaling both sides by
 * the same D keeps the symmetry of M.
 */

#ifndef PressureSolver_H
#define PressureSolver_H

#include <MueLu_TpetraOperator.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

class PressureSolver
{
  public:
    using Scalar        = double;
    using LocalOrdinal  = Tpetra::Map<>::local_ordinal_type;
    using GlobalOrdinal = Tpetra::Map<>::global_ordinal_type;
    using Node          = Tpetra::Map<>::node_type;
    using Map           = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
    using CrsMatrix     = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using Vector        = Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using MultiVector   = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
    using Operator      = Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    /*================================================================================================================*/
    /// Number of dimensions of the problem
    enum NumberOfDimensions
    {
        TWO   = 2,
        THREE = 3
    };

    /**
     * @brief Solver on the coarsest multigrid level. In the MUSIC namelist, use_coarse_solver < 0 is HEURISTIC, 0 is
     * NO_SOLVER and > 0 is KLU2.
     */
    enum CoarseSolver
    {
        HEURISTIC,  // KLU2 in 3D, none in 2D
        NO_SOLVER,  // No coarse solver, only the smoother on the coarsest level
        KLU2        // Direct solve with KLU2
    };

    /**
     * @brief Solver settings, named after the MUSIC namelist entries of `&NewtonKrylov`.
     */
    struct Settings
    {
        NumberOfDimensions ndim;                     // Number of dimensions of the problem
        double pbp_residual;                         // GMRES convergence tolerance, relative to the norm of the RHS
        int pbp_max_iter;                            // Maximum number of GMRES iterations
        int max_levels_override        = 0;          // Maximum number of multigrid levels, <= 0 uses the heuristic
        CoarseSolver use_coarse_solver = HEURISTIC;  // Solver on the coarsest multigrid level
    };

    /**
     * @brief Outcome of a solve.
     */
    struct Result
    {
        int iterations;            // Number of GMRES iterations
        double achievedTolerance;  // Residual reached by GMRES, relative to the norm of the scaled RHS
        bool converged;            // Whether GMRES reached the tolerance
    };

    /**
     * @brief Constructor
     *
     * @param[in] settings - The solver settings.
     * @param[in] rowGlobalIndices - Global indices of the rows owned by this rank, in the local order of the vectors.
     * @param[in] comm - Communicator over which the rows are distributed.
     */
    PressureSolver(const Settings& settings,
                   Teuchos::ArrayView<const GlobalOrdinal> rowGlobalIndices,
                   const Teuchos::RCP<const Teuchos::Comm<int>>& comm);

    /*================================================================================================================*/
    /**
     * @brief Set the matrix values and apply the symmetric diagonal scaling M' = D M D.
     *
     * @details The sparsity pattern is fixed by the first call; later calls must give the same pattern.
     *
     * @param[in] rowOffsets - CSR row offsets for the local rows, of size numLocalRows + 1.
     * @param[in] columnGlobalIndices - Global column index of each entry.
     * @param[in] values - Value of each entry.
     */
    void setMatrix(Teuchos::ArrayView<const size_t> rowOffsets,
                   Teuchos::ArrayView<const GlobalOrdinal> columnGlobalIndices,
                   Teuchos::ArrayView<const Scalar> values);

    /**
     * @brief Build the multigrid hierarchy from the current matrix.
     */
    void computePreconditioner();

    /**
     * @brief Solve the pressure system for the local rows.
     *
     * @param[in]  rhs - Right hand side (dp_rhs) for the local rows.
     * @param[out] solution - Solution (delta_prec) for the local rows.
     *
     * @return The outcome of the solve.
     */
    Result solve(Teuchos::ArrayView<const Scalar> rhs, Teuchos::ArrayView<Scalar> solution);

    /**
     * @brief Getter for the row map.
     */
    Teuchos::RCP<const Map> rowMap() const
    { return m_rowMap; }

    /**
     * @brief Getter for the scaled matrix M' = D M D (see the class description).
     */
    Teuchos::RCP<const CrsMatrix> scaledMatrix() const
    { return m_matrix; }

  private:
    const Settings m_settings;
    Teuchos::RCP<const Map> m_rowMap;
    Teuchos::RCP<CrsMatrix> m_matrix;
    Teuchos::RCP<Vector> m_diag;     // Diagonal of D, the inverse square roots of the row one norms of M
    Teuchos::RCP<Vector> m_yScaled;  // Y' = D Y, the rhs of the scaled system
    Teuchos::ParameterList m_belosSettings;
    Teuchos::ParameterList m_mueluSettings;
    Teuchos::RCP<MueLu::TpetraOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node>> m_preconditioner;

    /**
     * @brief Set the MueLu settings which depend on the size of the matrix.
     */
    void setHierarchySettings();
};

#endif  // PressureSolver_H
