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
     * @param[in] prec_nvar - The number of variables to precondition (1: just pressure, 2: pressure and energy)
     */
    PhysicsBasedPreconditioner(PackedVariables solution, PackedVariables solution_dPd, const int prec_nvar = 1);

    /**
     * @brief Set the RHS pressure
     *
     * @param[in]  delta_rho - 3D view of the density delta.
     * @param[in]  delta_energy - 3D view of the energy delta.
     * @param[out] rhs_pressure - 3D view of the RHS pressure delta to be set.
     * @param[out] rhs_energy - 3D view of the RHS energy delta to be set. Only set uf prec_nvar is 2.
     */
    void setPressureRHS(const Array3D<double> delta_rho,
                        const Array3D<double> delta_energy,
                        Array3D<double> rhs_pressure,
                        Array3D<double> rhs_energy);

    /**
     * @brief getter for density index
     */
    int n_rho() const
    { return m_n_rho; }

    /**
     * @brief getter for pressure index
     */
    int n_pressure() const
    { return m_n_pressure; }

    /**
     * @brief getter for energy index
     */
    int n_energy() const
    { return m_n_energy; }

    /**
     * @brief getter for density gradient index
     */
    int n_dPdrho() const
    { return m_n_dPdrho; }

    /**
     * @brief getter for energy gradient index
     */
    int n_dPdenergy() const
    { return m_n_dPdenergy; }

  private:
    /// @brief Store of the solution
    PackedVariables m_solution;
    /// @brief Store of the solution gradients
    PackedVariables m_solution_dPd;
    /// Indices for each variable
    int m_n_rho;
    int m_n_pressure;
    int m_n_energy;
    int m_n_dPdrho;
    int m_n_dPdenergy;
    /// Flag to indicate whether to precondition the pressure equation only (`prec_nvar=1`), or both pressure and energy equations (`prec_nvar=2`)
    int m_prec_nvar;
};

#endif  // PhysicsBasedPreconditioner_H
