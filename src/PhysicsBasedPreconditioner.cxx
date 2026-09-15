#include "PhysicsBasedPreconditioner.h"

#include "Array3D.h"
#include "Slab.h"

PhysicsBasedPreconditioner::PhysicsBasedPreconditioner(PackedVariables solution,
                                                       const int n_rho,
                                                       const int n_e,
                                                       const int n_dPdrho,
                                                       const int n_dPde,
                                                       const int n_pressure,
                                                       const int n_energy,
                                                       const int prec_nvar) :
    m_solution(solution),
    m_n_rho(n_rho),
    m_n_e(n_e),
    m_n_dPdrho(n_dPdrho),
    m_n_dPde(n_dPde),
    m_n_pressure(n_pressure),
    m_n_energy(n_energy),
    m_prec_nvar(prec_nvar)
{
}

void PhysicsBasedPreconditioner::setPressureRHS(const PackedVariables delta_prim_view, PackedVariables dp_rhs_view)
{
    const Array3D<double> delta_rho = delta_prim_view.singleVariableView(m_n_rho);
    const Array3D<double> delta_e   = delta_prim_view.singleVariableView(m_n_e);

    Array3D<double> dp_rhs_press = dp_rhs_view.singleVariableView(m_n_pressure);

    const Array3D<double> solution_dPdrho = m_solution.singleVariableView(m_n_dPdrho);
    const Array3D<double> solution_dPde   = m_solution.singleVariableView(m_n_dPde);

    Slab rhs_pressure_slab = dp_rhs_press.cellSlab();
    for (int i = rhs_pressure_slab.start(0); i < rhs_pressure_slab.end(0); i++)
    {
        for (int j = rhs_pressure_slab.start(1); j < rhs_pressure_slab.end(1); j++)
        {
            for (int k = rhs_pressure_slab.start(2); k < rhs_pressure_slab.end(2); k++)
            {
                dp_rhs_press(i, j, k) =
                    -(solution_dPdrho(i, j, k) * delta_rho(i, j, k) + solution_dPde(i, j, k) * delta_e(i, j, k));
                if (m_prec_nvar == 2) dp_rhs_view.singleVariableView(m_n_energy)(i, j, k) = -delta_e(i, j, k);
            }
        }
    }
}
