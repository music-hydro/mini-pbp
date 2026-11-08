/**
 * @class PhysicsBasedPreconditioner
 *
 * @brief A class containing the core preconditioning logic.
 */

#ifndef PhysicsBasedPreconditioner_H
#define PhysicsBasedPreconditioner_H

#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "DirectionalArray.h"
#include "PackedVariables.h"
#include "Slab.h"
#include <array>
#include <vector>

class PhysicsBasedPreconditioner
{
  public:
    /**
     * @class Constructor
     *
     * @param[in] solution - The solution containing 3 packed variables (rho, then pressure and then energy)
     * @param[in] solution_dPd - The solution gradients containing 2 packed variables (dPd_rho and then dPd_energy)
     * @param[in] squared_sound_speed - 3D view of the squared sound speed within each each cell.
     * @param[in] prec_nvar - The number of variables to precondition (1: just pressure, 2: pressure and energy)
     */
    PhysicsBasedPreconditioner(PackedVariables& solution,
                               PackedVariables& solution_dPd,
                               Array3D<double>& squared_sound_speed,
                               const unsigned int prec_nvar = 1);

    /**
     * @brief Set the RHS pressure
     *
     * @param[in]  delta_rho - 3D view of the density delta.
     * @param[in]  delta_energy - 3D view of the energy delta.
     * @param[out] rhs_pressure - 3D view of the RHS pressure delta to be set.
     * @param[out] rhs_energy - 3D view of the RHS energy delta to be set. Only set uf prec_nvar is 2.
     */
    void setPressureRHS(const Array3D<double>& delta_rho,
                        const Array3D<double>& delta_energy,
                        Array3D<double>& rhs_pressure,
                        Array3D<double>& rhs_energy);

    /**
     * @brief Reconstruct the energy delta for the next timestep after preconditioning.
     *
     * @param[in]  preconditioned_delta_pressure - 3D view of pressure delta calculated in the preconditioner pressure solve.
     * @param[in]  preconditioned_delta_energy - 3D view of energy delta calculated in the preconditioner pressure solve.
     * @param[in]  dp_rhs_press - 3D view of the pressure array set by setPressureRHS.
     * @param[out] delta_energy - 3D View of the energy delta to be calculated.
     */
    void reconstructEnergyDelta(const Array3D<double>& preconditioned_delta_pressure,
                                const Array3D<double>& preconditioned_delta_energy,
                                const Array3D<double>& dp_rhs_press,
                                Array3D<double>& delta_energy);


    /**
     * @brief Reconstruct the density delta for the next timestep after preconditioning.
     *
     * @param[in]  delta_pressure - 3D View of the pressure delta.
     * @param[in]  delta_energy - 3D View of the energy delta.
     * @param[out] delta_rho - 3D view of the density delta to be calculated.
     */
    void reconstructDensityDelta(const Array3D<double>& delta_pressure,
                                 const Array3D<double>& delta_energy,
                                 Array3D<double>& delta_rho);


    /**
     * @brief Assemble the rows of this rank of the preconditioner pressure matrix, as MUSIC `set_matrix`
     *        (prec_nvar 1).
     *
     * @details For each cell, with coef the grid Laplacian coefficients times rho / rho_face times the neighbour link
     *          weight (0 on a reflective boundary face, else 1), J_PP = -dt c^2:
     *          diagonal 1/dt - J_PP sum(coef), and J_PP coef for each neighbour that exists.
     *
     *          Unknowns are numbered as MUSIC `global_dof_idx0`: the cell position in the global domain, i fastest. A
     *          neighbour across a periodic boundary is the cell at the other end of the domain; a neighbour outside a
     *          non periodic boundary does not exist and gets no entry, but its face still counts in the diagonal.
     *
     *          The rows come in the local order of the cells (i fastest), the entries of a row in the order MUSIC sets
     *          them (diagonal, then NEGATIVE and POSITIVE along each axis), as CSR arrays ready for the solver. Each
     *          call replaces the previous matrix.
     *
     *          The density faces read rho one cell beyond the rank: the ghosts of `rho` are exchanged here, so every
     *          rank of its plan must call this.
     *
     * @param[in] ndim - The number of dimensions (2 or 3).
     * @param[in] dt - The time step.
     * @param[in,out] rho - The density on the cells of this rank, with the plan of this rank (neighbours, boundary
     *                conditions, global and local domains). Its ghosts are exchanged.
     * @param[in] squared_sound_speed - The squared sound speed per cell of this rank.
     * @param[in] laplacian_coefficients - The grid Laplacian coefficients (MUSIC `cell_centered_laplacian_coeffs`) of the
     *            six faces of each cell of this rank.
     */
    void assemblePressureMatrix(const unsigned int ndim,
                                const double dt,
                                Array3DWithGhosts<double>& rho,
                                const Array3D<double>& squared_sound_speed,
                                const Array3D<DirectionalArray<double>>& laplacian_coefficients);

    /**
     * @brief Global indices of the rows of the pressure matrix of this rank, in the local order of the cells.
     */
    const std::vector<long long>& pressureMatrixRowGlobalIndices() const
    { return m_global_pressure_row_indices; }

    /**
     * @brief CSR row offsets of the pressure matrix, of size number of local rows + 1.
     */
    const std::vector<size_t>& pressureMatrixRowOffsets() const
    { return m_pressure_row_offsets; }

    /**
     * @brief Global column index of each entry of the pressure matrix.
     */
    const std::vector<long long>& pressureMatrixColumnGlobalIndices() const
    { return m_global_pressure_column_indices; }

    /**
     * @brief Value of each entry of the pressure matrix.
     */
    const std::vector<double>& pressureMatrixValues() const
    { return m_pressure_matrix_for_solver; }

    /**
     * @brief getter for density index
     */
    unsigned int n_rho() const
    { return m_n_rho; }

    /**
     * @brief getter for pressure index
     */
    unsigned int n_pressure() const
    { return m_n_pressure; }

    /**
     * @brief getter for energy index
     */
    unsigned int n_energy() const
    { return m_n_energy; }

    /**
     * @brief getter for density gradient index
     */
    unsigned int n_dPdrho() const
    { return m_n_dPdrho; }

    /**
     * @brief getter for energy gradient index
     */
    unsigned int n_dPdenergy() const
    { return m_n_dPdenergy; }

    PackedVariables solution() const
    { return m_solution; }

  private:
    /**
     * @brief Compute squared sound speed from the solution variables.
     *
     * @details For now this is set using some data from MUSIC (should be calculated instead)
     */
    void calculateSquaredSoundSpeed();

    /**
     * @brief Calculate the change in energy for the timestep next timestep.
     *
     * @details Energy equation, we do not consider heat conduction here
     *            A_ee = 1/dt and A_pe = -dt*p/rho^2*L(dp)
     *            bis: A_dd = 1/dt and A_pd = -dt*L(dp)
     *
     * @param[in]  preconditioned_delta_pressure - 3D view of pressure delta calculated in the preconditioner pressure solve.
     * @param[in]  dp_rhs_press - 3D view of the pressure array set by setPressureRHS.
     * @param[out] delta_energy - 3D View of the energy delta to be calculated.
     */
    void calculateEnergyDelta(const Array3D<double>& preconditioned_delta_pressure,
                              const Array3D<double>& dp_rhs_press,
                              Array3D<double>& delta_energy);

    /**
     * @brief Grid Laplacian coefficients times rho / rho_face for a given cell, as MUSIC `pressure_coeffs`.
     *
     * @param[in] ndim - The number of dimensions (2 or 3). Along k in 2D, rho_face is rho.
     * @param[in] rho - The density with its ghosts exchanged.
     * @param[in] laplacian_coefficients - The grid Laplacian coefficients of the cell.
     * @param[in] i - The index in the first dimension.
     * @param[in] j - The index in the second dimension.
     * @param[in] k - The index in the third dimension.
     */
    static DirectionalArray<double> pressureCoefficients(const unsigned int ndim,
                                                         const Array3DWithGhosts<double>& rho,
                                                         const DirectionalArray<double>& laplacian_coefficients,
                                                         const int i,
                                                         const int j,
                                                         const int k);

    /**
     * @brief Sum of the coefficients, adding the two directions of an axis first to reduce cancellations between axes,
     *        as MUSIC `coef_sum`.
     */
    static double coefSum(const DirectionalArray<double>& coefficients);

    /**
     * @brief Get the global index of unknown `var_index` at a cell, as MUSIC `global_dof_idx0`: var_index fastest,
     *        then the cell position in the global domain, i fastest.
     *
     * @param[in] position - The cell position, counted from the start of the global domain.
     * @param[in] var_index - The index of the unknown within the cell (0 for the pressure).
     * @param[in] global_slab - Slab representing the global domain.
     *
     * @returns The global index, or -1 if the cell does not exist (a position of -1 along an axis).
     */
    long long globalVarIndex(const std::array<int, 3>& position, const unsigned int var_index, const Slab& global_slab) const;

    /**
     * @brief Append an entry to the current row of the pressure matrix, skipped when the column does not exist, as
     *        MUSIC `set_element`.
     */
    void setElement(const long long column, const double value);

    /// @brief Store of the solution
    PackedVariables m_solution;
    /// @brief Store of the solution gradients
    PackedVariables m_solution_dPd;
    /// @brief Squared sound speed
    Array3D<double> m_squared_sound_speed;
    /// @brief Pressure matrix to be sent to solver
    std::vector<double> m_pressure_matrix_for_solver;
    /// @brief Map of global column indices for m_pressure_matrix_for_solver
    std::vector<long long> m_global_pressure_column_indices;
    /// @brief Map of global row indices for m_pressure_matrix_for_solver, one per row
    std::vector<long long> m_global_pressure_row_indices;
    /// @brief CSR row offsets into m_pressure_matrix_for_solver and m_global_pressure_column_indices
    std::vector<size_t> m_pressure_row_offsets;
    /// @brief Flag to indicate whether to precondition the pressure equation only (`prec_nvar=1`), or both pressure and energy equations (`prec_nvar=2`)
    unsigned int m_prec_nvar;
    /// Indices for each variable
    unsigned int m_n_rho;
    unsigned int m_n_pressure;
    unsigned int m_n_energy;
    unsigned int m_n_dPdrho;
    unsigned int m_n_dPdenergy;
    // This is a temporary solution. The value has been taken from a run of music and should be calculated instead.
    double m_dt = 1.0917241379310345E-004; /** timestep */
};

#endif  // PhysicsBasedPreconditioner_H
