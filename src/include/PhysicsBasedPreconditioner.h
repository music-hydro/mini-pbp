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
#include "GhostExchangePlan.h"
#include "PackedVariables.h"

class PhysicsBasedPreconditioner
{
  public:
    /**
     * @class Constructor
     *
     * @param[in] dt - The timestep for this problem.
     * @param[in] ndim - The number of dimensions (2 or 3).
     * @param[in] nmom - The number of velocity components (2 or 3, at least ndim).
     * @param[in] mhd_enabled - Whether the magnetic field components are present.
     * @param[in] solution - The solution containing 3 packed variables (rho, then pressure and then energy)
     * @param[in] solution_dPd - The solution gradients containing 2 packed variables (dPd_rho and then dPd_energy)
     * @param[in] squared_sound_speed - 3D view of the squared sound speed within each each cell.
     * @param[in] prec_nvar - The number of variables to precondition (1: just pressure, 2: pressure and energy)
     */
    PhysicsBasedPreconditioner(const double dt,
                               const unsigned int ndim,
                               const unsigned int nmom,
                               const bool mhd_enabled,
                               PackedVariablesWithGhosts& solution,
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
     * @brief Assembles the pressure matrix ready for preconditioning
     *
     * @param[in] pressure - The pressure matrix to be assembled.
     * @param[in] laplacian_coefficients
     */
    void assemblePressureMatrix(Array3DWithGhosts<double>& pressure,
                                const Array3D<DirectionalArray<double>>& laplacian_coefficients);

    /**
     * @brief Convert the residual from conserved to primitive variables, the first step of the preconditioner (MUSIC
     * `cons_to_prim` in `src/solvers/preconditioner.f90`).
     *
     * @details rho and e are read from the stored solution. The scalar packed variables must have matching counts.
     *          Empty PackedVariables objects may be passed when there are no active scalars or MHD is disabled.
     *          - delta_rho = delta_conserved_rho
     *          - delta_e = (delta_rho_e - e delta_rho) / rho
     *          - each active scalar: delta_phi = (delta_rho_phi - phi delta_rho) / rho
     *          - each velocity component lives on the faces along its own axis: delta_v = (delta_rho_v - v delta_rho_f) / rho_f,
     *            with rho_f and delta_rho_f the average of the two cells next to the face. In 2D the third component is
     *            invariant along k and uses the cell values.
     *          - each magnetic field component is copied unchanged.
     *
     *          A field staggered along an axis holds at (i, j, k) the face on the negative side of cell (i, j, k). Its cellSlab()
     *          is the faces solved on the rank (MUSIC `solved_local_domain_nodes`), so it may start at 1 or end at n + 1. Every
     *          loop runs over the slab of the primitive delta it writes. The face averages read rho and delta rho one cell
     *          beyond the rank, so both are exchanged on `plan`; this is collective over the neighbours in the plan.
     *
     * @param[in] plan - The ghost exchange plan of the cells of this rank (MUSIC `rho_plans`).
     * @param[in] active_scalars - Packed solution active scalars phi on the cells.
     * @param[in] delta_conserved - Packed conserved residual containing delta rho, delta (rho e) and delta (rho v) components.
     * @param[in] delta_conserved_rho_scalars - Packed delta (rho phi) for each active scalar.
     * @param[in] delta_conserved_magnetic_field - Packed delta B for each component (three when MHD is enabled).
     * @param[out] delta_primitive - Packed primitive residual containing delta rho, delta e and delta v components.
     * @param[out] delta_primitive_scalars - Packed delta phi for each active scalar.
     * @param[out] delta_primitive_magnetic_field - Packed delta B for each component (three when MHD is enabled).
     */
    void conservedToPrimitive(const GhostExchangePlan& plan,
                              const PackedVariables active_scalars,
                              const PackedVariables delta_conserved,
                              const PackedVariables delta_conserved_rho_scalars,
                              const PackedVariables delta_conserved_magnetic_field,
                              PackedVariables delta_primitive,
                              PackedVariables delta_primitive_scalars,
                              PackedVariables delta_primitive_magnetic_field);

    /**
     * @brief Getter for the number of packed variables in the solution
     */
    static unsigned int num_sol_vars()
    { return m_num_sol_vars; }

    /**
     * @brief Getter for the number of packed variables in the solution gradients
     */
    static unsigned int num_sol_dPd_vars()
    { return m_num_sol_dPd_vars; }

    /**
     * @brief getter for density index
     */
    static unsigned int n_rho()
    { return m_n_rho; }

    /**
     * @brief getter for pressure index
     */
    static unsigned int n_pressure()
    { return m_n_pressure; }

    /**
     * @brief getter for energy index
     */
    static unsigned int n_energy()
    { return m_n_energy; }

    /**
     * @brief Getter for the variable index of the velocity in the ith direction
     */
    static unsigned int n_vel_i()
    { return m_n_vel_i; }

    /**
     * @brief Getter for the variable index of the velocity in the jth direction
     */
    static unsigned int n_vel_j()
    { return m_n_vel_j; }

    /**
     * @brief Getter for the variable index of the velocity in the kth direction
     */
    static unsigned int n_vel_k()
    { return m_n_vel_k; }

    /**
     * @brief getter for density gradient index
     */
    static unsigned int n_dPdrho()
    { return m_n_dPdrho; }

    /**
     * @brief getter for energy gradient index
     */
    static unsigned int n_dPdenergy()
    { return m_n_dPdenergy; }

    PackedVariablesWithGhosts solution() const
    { return m_solution; }

    /**
     * @brief Get a read-only version of the pressure matrix to be sent to the solver.
     */
    const std::vector<double>& pressureMatrixForSolver() const
    { return m_pressure_matrix_for_solver; }

    /**
     * @brief Get a read-only version of the pressure column indices to be sent to the solver.
     */
    const std::vector<long long>& globalPressureColumnIndices() const
    { return m_global_pressure_column_indices; }

    /**
     * @brief Get a read-only version of the pressure row indices to be sent to the solver.
     */
    const std::vector<long long>& globalPressureRowIndices() const
    { return m_global_pressure_row_indices; }

    const std::vector<long long>& globalPressureRowOffsets() const
    { return m_global_pressure_row_offsets; }


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
     * @brief Get the pressure coefficients for a given cell
     *
     * @param i - The index in the first dimension.
     * @param j - The index in the second dimension.
     * @param k - The index in the third dimension.
     */
    DirectionalArray<double> pressureCoefficients(const int& i,
                                                  const int& j,
                                                  const int& k,
                                                  const Array3DWithGhosts<double>& solution_rho,
                                                  const DirectionalArray<double>& laplacian_coefficients,
                                                  const DirectionalArray<double>& weights);

    /**
     * @brief Get the index of a variable value at a a specific cell centre.
     *
     * @param[in] i - The global cell index in the first dimension.
     * @param[in] j - The global cell index in the second dimension.
     * @param[in] k - The global cell index in the third dimension.
     * @param[in] var_index - The index of the variable to be accessed.
     * @param[in] global_slab - Slab representing the global domain.
     */
    unsigned long long globalVarIndex(const int& i, const int& j, const int& k, const int& var_index, const Slab& global_slab)
    { return i + global_slab.extent(0) * (j + global_slab.extent(1) * (k + global_slab.extent(2) * var_index)); }

    /// @brief The number of dimensions (2 or 3).
    const unsigned int m_ndim;
    /// @brief The number of velocity components (2 or 3, at least ndim).
    const unsigned int m_nmom;
    /// @brief Whether the magnetic field components are present.
    const bool m_mhd_enabled;
    /// @brief The timestep for this problem.
    const double m_dt;
    /// @brief Store of the solution
    PackedVariablesWithGhosts m_solution;
    /// @brief Store of the solution gradients
    PackedVariables m_solution_dPd;
    /// @brief Squared sound speed
    Array3D<double> m_squared_sound_speed;
    /// @brief Pressure matrix to be sent to solver
    std::vector<double> m_pressure_matrix_for_solver;
    /// @brief Map of global column indices for m_pressure_matrix_for_solver
    std::vector<long long> m_global_pressure_column_indices;
    /// @brief Map of global row indices for m_pressure_matrix_for_solver
    std::vector<long long> m_global_pressure_row_indices;
    std::vector<long long> m_global_pressure_row_offsets;
    /// @brief Flag to indicate whether to precondition the pressure equation only (`prec_nvar=1`), or both pressure and energy equations (`prec_nvar=2`)
    unsigned int m_prec_nvar;
    /// @brief The number of packed variables that must be present in the solution
    static const unsigned int m_num_sol_vars = 6;
    /// @brief The number of packed variables that must be present in the solution gradients
    static const unsigned int m_num_sol_dPd_vars = 2;
    /// Indices for each variable
    static const unsigned int m_n_rho       = 0;
    static const unsigned int m_n_pressure  = 1;
    static const unsigned int m_n_energy    = 2;
    static const unsigned int m_n_vel_i     = 3;
    static const unsigned int m_n_vel_j     = 4;
    static const unsigned int m_n_vel_k     = 5;
    static const unsigned int m_n_dPdrho    = 0;
    static const unsigned int m_n_dPdenergy = 1;
};

#endif  // PhysicsBasedPreconditioner_H
