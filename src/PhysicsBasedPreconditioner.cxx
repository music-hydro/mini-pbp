#include "PhysicsBasedPreconditioner.h"

#include "Array3D.h"
#include "Slab.h"
#include <stdexcept>
#include <string>

PhysicsBasedPreconditioner::PhysicsBasedPreconditioner(PackedVariables solution,
                                                       PackedVariables solution_dPd,
                                                       const int prec_nvar) :
    m_solution(solution),
    m_solution_dPd(solution_dPd),
    m_prec_nvar(prec_nvar),
    m_n_rho(0),
    m_n_pressure(1),
    m_n_energy(2),
    m_n_dPdrho(0),
    m_n_dPdenergy(1)
{
    // Verify the size of solutions
    int num_solution_variables = m_solution.numVariables();
    if (num_solution_variables != 3)
    {
        throw std::invalid_argument("Solution should contain exactly 3 variables, found " +
                                    std::to_string(num_solution_variables));
    }
    int num_solution_dPd_variables = m_solution_dPd.numVariables();
    if (num_solution_dPd_variables != 2)
    {
        throw std::invalid_argument("Solution gradients should contain exactly 2 variables, found " +
                                    std::to_string(num_solution_dPd_variables));
    }
}

void PhysicsBasedPreconditioner::setPressureRHS(const Array3D<double> delta_rho,
                                                const Array3D<double> delta_energy,
                                                Array3D<double> rhs_pressure,
                                                Array3D<double> rhs_energy)
{
    const Array3D<double> solution_dPdrho    = m_solution_dPd.singleVariableView(m_n_dPdrho);
    const Array3D<double> solution_dPdenergy = m_solution_dPd.singleVariableView(m_n_dPdenergy);

    // Arrays are column major, so loop with i innermost for unit stride access
    Slab rhs_pressure_slab = rhs_pressure.cellSlab();
    for (int k = rhs_pressure_slab.start(2); k < rhs_pressure_slab.end(2); k++)
    {
        for (int j = rhs_pressure_slab.start(1); j < rhs_pressure_slab.end(1); j++)
        {
            for (int i = rhs_pressure_slab.start(0); i < rhs_pressure_slab.end(0); i++)
            {
                rhs_pressure(i, j, k) =
                    -(solution_dPdrho(i, j, k) * delta_rho(i, j, k) + solution_dPdenergy(i, j, k) * delta_energy(i, j, k));
            }
        }
    }

    if (m_prec_nvar == 2)
    {
        for (int k = rhs_pressure_slab.start(2); k < rhs_pressure_slab.end(2); k++)
        {
            for (int j = rhs_pressure_slab.start(1); j < rhs_pressure_slab.end(1); j++)
            {
                for (int i = rhs_pressure_slab.start(0); i < rhs_pressure_slab.end(0); i++)
                {
                    rhs_energy(i, j, k) = -delta_energy(i, j, k);
                }
            }
        }
    }
}
