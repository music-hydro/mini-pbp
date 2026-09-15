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
     */
    PhysicsBasedPreconditioner(PackedVariables solution,
                               const int n_rho,
                               const int n_e,
                               const int n_dPdrho,
                               const int n_dPde,
                               const int n_pressure,
                               const int n_energy,
                               const int prec_nvar = 1);

    /**
     * @brief Set the RHS pressure
     */
    void setPressureRHS(const PackedVariables delta_prim_view, PackedVariables dp_rhs_view);

    PackedVariables solution() const
    { return m_solution; }

  private:
    /// Store of the solution
    PackedVariables m_solution;
    /// Indices for each variable
    int m_n_rho;
    int m_n_e;
    int m_n_dPdrho;
    int m_n_dPde;
    int m_n_pressure;
    int m_n_energy;
    /// Flag to indicate whether to precondition the pressure equation only (`prec_nvar=1`), or both pressure and energy equations (`prec_nvar=2`)
    int m_prec_nvar;
};

#endif  // PhysicsBasedPreconditioner_H
