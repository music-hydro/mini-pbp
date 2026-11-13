/**
 * @class PhysicsBasedPreconditioner
 *
 * @brief A class containing the core preconditioning logic.
 */

#ifndef PhysicsBasedPreconditioner_H
#define PhysicsBasedPreconditioner_H

#include "Array3DWithGhosts.h"
#include "GhostExchangePlan.h"
#include "PackedVariables.h"
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
    PhysicsBasedPreconditioner(PackedVariablesWithGhosts& solution,
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
     * @param[in] delta_pressure - 3D View of the pressure delta.
     * @param[in] delta_energy - 3D View of the energy delta.
     * @param[out] delta_rho - 3D view of the density delta to be calculated.
     */
    void reconstructDensityDelta(const Array3D<double>& delta_pressure,
                                 const Array3D<double>& delta_energy,
                                 Array3D<double>& delta_rho);

    /**
     * @brief Convert the residual from conserved to primitive variables, the first step of the preconditioner (MUSIC
     * `cons_to_prim` in `src/solvers/preconditioner.f90`).
     *
     * @details rho and e are read from the stored solution.
     * - delta_rho = delta_conserved_rho
     * - delta_e = (delta_rho_e - e delta_rho) / rho
     * - each active scalar: delta_phi = (delta_rho_phi - phi delta_rho) / rho
     * - each velocity component lives on the faces along its own axis: delta_v = (delta_rho_v - v delta_rho_f) / rho_f,
     *   with rho_f and delta_rho_f the average of the two cells next to the face. In 2D the third component is
     *   invariant along k and uses the cell values.
     * - each magnetic field component is copied unchanged.
     *
     * A field staggered along an axis holds at (i, j, k) the face on the negative side of cell (i, j, k). Its cellSlab()
     * is the faces solved on the rank (MUSIC `solved_local_domain_nodes`), so it may start at 1 or end at n + 1. Every
     * loop runs over the slab of the primitive delta it writes. The face averages read rho and delta rho one cell
     * beyond the rank, so both are exchanged on `plan`; this is collective over the neighbours in the plan.
     *
     * @param[in] ndim - The number of dimensions (2 or 3).
     * @param[in] nmom - The number of velocity components (2 or 3, at least ndim).
     * @param[in] mhd_enabled - Whether the magnetic field components are present.
     * @param[in] plan - The ghost exchange plan of the cells of this rank (MUSIC `rho_plans`).
     * @param[in] vel - The solution velocity on the same faces as delta_primitive_vel (only the first nmom are read).
     * @param[in] active_scalars - The solution active scalars phi on the cells.
     * @param[in] delta_conserved_rho - delta rho on all the cells of the rank.
     * @param[in] delta_conserved_rho_e - delta (rho e).
     * @param[in] delta_conserved_rho_vel - delta (rho v) of each velocity component (only the first nmom are read).
     * @param[in] delta_conserved_rho_scalars - delta (rho phi) of each active scalar.
     * @param[in] delta_conserved_magnetic_field - delta B of each component (only read with MHD).
     * @param[out] delta_primitive_rho - delta rho.
     * @param[out] delta_primitive_e - delta e.
     * @param[out] delta_primitive_vel - delta v of each velocity component (only the first nmom are written).
     * @param[out] delta_primitive_scalars - delta phi of each active scalar.
     * @param[out] delta_primitive_magnetic_field - delta B of each component (only written with MHD).
     */
    void conservedToPrimitive(const unsigned int ndim,
                              const unsigned int nmom,
                              const bool mhd_enabled,
                              GhostExchangePlan<double>& plan,
                              const std::vector<const Array3D<double>*>& active_scalars,
                              PackedVariables delta_conserved,
                              const std::vector<const Array3D<double>*>& delta_conserved_rho_scalars,
                              const std::array<const Array3D<double>*, 3>& delta_conserved_magnetic_field,
                              PackedVariables delta_primitive,
                              const std::vector<Array3D<double>*>& delta_primitive_scalars,
                              const std::array<Array3D<double>*, 3>& delta_primitive_magnetic_field);

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
     * @brief Getter for the velocity in the ith direction variable index
     */
    unsigned int n_vel_i()
    { return m_n_vel_i; }

    /**
     * @brief Getter for the velocity in the jth direction variable index
     */
    unsigned int n_vel_j()
    { return m_n_vel_j; }

    /**
     * @brief Getter for the velocity in the kth direction variable index
     */
    unsigned int n_vel_k()
    { return m_n_vel_k; }

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

    PackedVariablesWithGhosts solution() const
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

    /// @brief Store of the solution
    PackedVariablesWithGhosts m_solution;
    /// @brief Store of the solution gradients
    PackedVariables m_solution_dPd;
    /// @brief Squared sound speed
    Array3D<double> m_squared_sound_speed;
    /// @brief Flag to indicate whether to precondition the pressure equation only (`prec_nvar=1`), or both pressure and energy equations (`prec_nvar=2`)
    unsigned int m_prec_nvar;
    /// Indices for each variable
    unsigned int m_n_rho;
    unsigned int m_n_pressure;
    unsigned int m_n_energy;
    unsigned int m_n_vel_i;
    unsigned int m_n_vel_j;
    unsigned int m_n_vel_k;
    unsigned int m_n_dPdrho;
    unsigned int m_n_dPdenergy;
    // This is a temporary solution. The value has been taken from a run of music and should be calculated instead.
    double m_dt = 1.0917241379310345E-004; /** timestep */
};

#endif  // PhysicsBasedPreconditioner_H
