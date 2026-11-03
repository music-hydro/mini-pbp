/**
 * @class PressureSolver
 *
 * @brief Solves the preconditioner pressure system M dP = dp_rhs with Trilinos, using the same method and settings as
 * MUSIC (`src/trilinos/PbpOperator.cpp`): symmetric diagonal scaling, Belos pseudo block GMRES and a MueLu multigrid
 * right preconditioner.
 *
 * In the physics based preconditioner of MUSIC, the hydro correction is reduced to a parabolic problem for the pressure
 * correction dP. For prec_nvar = 1, M is the 7 point (3D) or 5 point (2D) operator built by `set_matrix` in
 * `src/solvers/preconditioner.f90`: diagonal 1/dt + dt c^2 sum(coef), off diagonal -dt c^2 coef for each neighbour,
 * where coeff the pressure coefficients times the neighbour link weights. It acts as a
 * Poisson operator as dt grows, which is why the multigrid starts from the MueLu Poisson defaults. The rhs dp_rhs is
 * built by `set_pressure_rhs` and the solution dP is used by `apply` to recover the rest of the correction.
 *
 * The parts of MUSIC this class follows:
 * - constructor: Belos and MueLu settings and the hierarchy heuristics of the PbpOperator constructor and of
 *   `HierarchyHeuristics` / `GridAggregationModel` (`src/trilinos/PbpOperator.cpp`)
 * - setMatrix(): `PbpOperator::ComputeMatrix`, with the CSR arrays standing in for the `set_matrix` callback and
 *   `music_cpp_pbp_set_matrix_from_arrays` (`src/trilinos/TrilinosInterface.cpp`)
 * - computePreconditioner(): `PbpOperator::ComputePreconditioner`
 * - solve(): `PbpOperator::solve_pressure`, called from `apply` through `music_cpp_pbp_solve_pressure`
 *
 * The matrix is given as CSR arrays with global column indices. As in MUSIC (`PbpPrecond::pre_solve` in
 * `src/trilinos/PbpPrecond.h`), the matrix can be set again every Newton iteration while the multigrid hierarchy is
 * only rebuilt when computePreconditioner() is called (MUSIC does it on the first Newton iteration of a time step).
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
    /**
     * @brief Solver settings, named after the MUSIC namelist entries of `&NewtonKrylov`.
     */
    struct Settings
    {
        int ndim;                      // Number of dimensions of the problem (2 or 3)
        double pbp_residual;           // GMRES convergence tolerance, relative to the norm of the RHS
        int pbp_max_iter;              // Maximum number of GMRES iterations
        int max_levels_override = 0;   // Maximum number of multigrid levels, <= 0 uses the heuristic
        int use_coarse_solver   = -1;  // < 0 heuristic, == 0 no coarse solver, > 0 KLU2 coarse solver
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
     * @brief Set the matrix values and apply the symmetric diagonal scaling.
     *
     * The sparsity pattern is fixed by the first call; later calls must give the same pattern.
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
     * @brief Getter for the scaled matrix D A D.
     */
    Teuchos::RCP<const CrsMatrix> scaledMatrix() const
    { return m_matrix; }

  private:
    const Settings m_settings;
    Teuchos::RCP<const Map> m_rowMap;
    Teuchos::RCP<CrsMatrix> m_matrix;
    Teuchos::RCP<Vector> m_diag;
    Teuchos::ParameterList m_belosSettings;
    Teuchos::ParameterList m_mueluSettings;
    Teuchos::RCP<MueLu::TpetraOperator<Scalar, LocalOrdinal, GlobalOrdinal, Node>> m_preconditioner;

    /**
     * @brief Set the MueLu settings which depend on the size of the matrix.
     */
    void setHierarchySettings();
};

#endif  // PressureSolver_H
