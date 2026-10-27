#include "PhysicsBasedPreconditioner.h"

#include "Array3D.h"
#include "Slab.h"
#include <stdexcept>
#include <string>

PhysicsBasedPreconditioner::PhysicsBasedPreconditioner(PackedVariables& solution,
                                                       PackedVariables& solution_dPd,
                                                       Array3D<double>& squared_sound_speed,
                                                       const unsigned int prec_nvar) :
    m_solution(solution),
    m_solution_dPd(solution_dPd),
    m_squared_sound_speed(squared_sound_speed),
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

    unsigned int ni = solution.ni();
    unsigned int nj = solution.nj();
    unsigned int nk = solution.nk();
    if (ni != solution_dPd.ni() || nj != solution_dPd.nj() || nk != solution_dPd.nk())
    {
        throw std::invalid_argument("Solution and solution gradients must have the same domain dimensions");
    }
    calculateSquaredSoundSpeed();
}

void PhysicsBasedPreconditioner::setPressureRHS(const Array3D<double>& delta_rho,
                                                const Array3D<double>& delta_energy,
                                                Array3D<double>& rhs_pressure,
                                                Array3D<double>& rhs_energy)
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

    if (m_prec_nvar == 2) { delta_energy.deepCopy(rhs_energy, -1.0); }
}

void PhysicsBasedPreconditioner::reconstructEnergyDelta(const Array3D<double>& preconditioned_delta_pressure,
                                                        const Array3D<double>& preconditioned_delta_energy,
                                                        const Array3D<double>& rhs_pressure,
                                                        Array3D<double>& delta_energy)
{
    if (m_prec_nvar == 1)
    {
        const Array3D<double> solution_pressure = m_solution.singleVariableView(m_n_pressure);
        const Array3D<double> solution_rho      = m_solution.singleVariableView(m_n_rho);

        Slab delta_energy_slab = delta_energy.cellSlab();
        unsigned int i0        = delta_energy_slab.start(0);
        unsigned int j0        = delta_energy_slab.start(1);
        unsigned int k0        = delta_energy_slab.start(2);
        calculateEnergyDeltaKernel(&solution_pressure(i0, j0, k0),
                                   &solution_rho(i0, j0, k0),
                                   &preconditioned_delta_pressure(i0, j0, k0),
                                   &rhs_pressure(i0, j0, k0),
                                   &m_squared_sound_speed(i0, j0, k0),
                                   delta_energy_slab.volume(),
                                   &delta_energy(i0, j0, k0));
    }
    else
    {
        preconditioned_delta_energy.deepCopy(delta_energy);
    }
}

void PhysicsBasedPreconditioner::calculateEnergyDeltaKernel(const double* solution_pressure,
                                                            const double* solution_rho,
                                                            const double* preconditioned_delta_pressure,
                                                            const double* rhs_pressure,
                                                            const double* squared_sound_speed,
                                                            const unsigned int& total_cells,
                                                            double* delta_energy)
{
    for (unsigned int idx = 0; idx < total_cells; idx++)
    {
        double dt_LdP = (preconditioned_delta_pressure[idx] / m_dt - rhs_pressure[idx]) / squared_sound_speed[idx];
        double rho    = solution_rho[idx];
        delta_energy[idx] += m_dt * (solution_pressure[idx] / (rho * rho)) * dt_LdP;
    }
}

void PhysicsBasedPreconditioner::reconstructDensityDelta(const Array3D<double>& delta_pressure,
                                                         const Array3D<double>& delta_energy,
                                                         Array3D<double>& delta_rho)
{
    const Array3D<double> solution_dPdrho    = m_solution_dPd.singleVariableView(m_n_dPdrho);
    const Array3D<double> solution_dPdenergy = m_solution_dPd.singleVariableView(m_n_dPdenergy);

    Slab delta_rho_slab = delta_rho.cellSlab();
    for (int k = delta_rho_slab.start(2); k < delta_rho_slab.end(2); k++)
    {
        for (int j = delta_rho_slab.start(1); j < delta_rho_slab.end(1); j++)
        {
            for (int i = delta_rho_slab.start(0); i < delta_rho_slab.end(0); i++)
            {
                delta_rho(i, j, k) =
                    (delta_pressure(i, j, k) - solution_dPdenergy(i, j, k) * delta_energy(i, j, k)) / solution_dPdrho(i, j, k);
            }
        }
    }
}

void PhysicsBasedPreconditioner::calculateSquaredSoundSpeed()
{ m_squared_sound_speed.fill(2.5266187266788758E007); }
