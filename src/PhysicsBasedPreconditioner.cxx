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

    Slab rhs_pressure_slab = rhs_pressure.cellSlab();
    setPressureRHSKernel(solution_dPdrho.start(),
                         solution_dPdenergy.start(),
                         delta_rho.start(),
                         delta_energy.start(),
                         rhs_pressure_slab.volume(),
                         rhs_pressure.start());

    if (m_prec_nvar == 2) { delta_energy.deepCopy(rhs_energy, -1.0); }
}

void PhysicsBasedPreconditioner::setPressureRHSKernel(const double* solution_dPdrho,
                                                      const double* solution_dPdenergy,
                                                      const double* delta_rho,
                                                      const double* delta_energy,
                                                      const unsigned int& total_cells,
                                                      double* rhs_pressure)
{
    for (unsigned int idx = 0; idx < total_cells; idx++)
    {
        rhs_pressure[idx] = -(solution_dPdrho[idx] * delta_rho[idx] + solution_dPdenergy[idx] * delta_energy[idx]);
    }
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
        calculateEnergyDeltaKernel(solution_pressure.start(),
                                   solution_rho.start(),
                                   preconditioned_delta_pressure.start(),
                                   rhs_pressure.start(),
                                   m_squared_sound_speed.start(),
                                   delta_energy_slab.volume(),
                                   delta_energy.start());
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
    reconstructDensityDeltaKernel(solution_dPdrho.start(),
                                  solution_dPdenergy.start(),
                                  delta_pressure.start(),
                                  delta_energy.start(),
                                  delta_rho_slab.volume(),
                                  delta_rho.start());
}

void PhysicsBasedPreconditioner::reconstructDensityDeltaKernel(const double* solution_dPdrho,
                                                               const double* solution_dPdenergy,
                                                               const double* delta_pressure,
                                                               const double* delta_energy,
                                                               const unsigned int& total_cells,
                                                               double* delta_rho)
{
    for (unsigned int idx = 0; idx < total_cells; idx++)
    {
        delta_rho[idx] = (delta_pressure[idx] - solution_dPdenergy[idx] * delta_energy[idx]) / solution_dPdrho[idx];
    }
}

void PhysicsBasedPreconditioner::calculateSquaredSoundSpeed()
{ m_squared_sound_speed.fill(2.5266187266788758E007); }
