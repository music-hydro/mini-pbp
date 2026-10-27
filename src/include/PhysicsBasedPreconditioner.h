/**
 * @class PhysicsBasedPreconditioner
 *
 * @brief A class containing the core preconditioning logic.
 */

#ifndef PhysicsBasedPreconditioner_H
#define PhysicsBasedPreconditioner_H

#include "PackedVariables.h"

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
     * @param[in]  rhs_pressure - 3D view of the pressure array set by setPressureRHS.
     * @param[out] delta_energy - 3D View of the energy delta to be calculated.
     */
    void reconstructEnergyDelta(const Array3D<double>& preconditioned_delta_pressure,
                                const Array3D<double>& preconditioned_delta_energy,
                                const Array3D<double>& rhs_pressure,
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
     * @param[in]  solution_pressure - The solution pressure for the current cell range.
     * @param[in]  solution_rho - The solution density for the current cell range.
     * @param[in]  preconditioned_delta_pressure - Pressure delta for the current cell range calculated in the preconditioner pressure solve.
     * @param[in]  rhs_pressure - The pressure array for the current cell range set by setPressureRHS.
     * @param[in]  squared_sound_speed - The squared sound speeds for the current cell range.
     * @param[in]  total_cells - The total number of cells to be calculated.
     * @param[out] delta_energy - The energy delta to be calculated for the current cell range.
     */
    void calculateEnergyDeltaKernel(const double* solution_pressure,
                                    const double* solution_rho,
                                    const double* preconditioned_delta_pressure,
                                    const double* rhs_pressure,
                                    const double* squared_sound_speed,
                                    const unsigned int& total_cells,
                                    double* delta_energy);

    /// @brief Store of the solution
    PackedVariables m_solution;
    /// @brief Store of the solution gradients
    PackedVariables m_solution_dPd;
    /// Indices for each variable
    unsigned int m_n_rho;
    unsigned int m_n_pressure;
    unsigned int m_n_energy;
    unsigned int m_n_dPdrho;
    unsigned int m_n_dPdenergy;
    /// Flag to indicate whether to precondition the pressure equation only (`prec_nvar=1`), or both pressure and energy equations (`prec_nvar=2`)
    unsigned int m_prec_nvar;
    Array3D<double> m_squared_sound_speed; /** Squared sound speed */
    // This is a temporary solution. The value has been taken from a run of music and should be calculated instead.
    double m_dt = 1.0917241379310345E-004; /** timestep */
};

#endif  // PhysicsBasedPreconditioner_H
