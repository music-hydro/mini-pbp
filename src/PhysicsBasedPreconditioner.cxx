#include "PhysicsBasedPreconditioner.h"

#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "Slab.h"
#include <stdexcept>
#include <string>

PhysicsBasedPreconditioner::PhysicsBasedPreconditioner(const unsigned int ndim,
                                                       const unsigned int nmom,
                                                       const bool mhd_enabled,
                                                       PackedVariablesWithGhosts& solution,
                                                       PackedVariables& solution_dPd,
                                                       Array3D<double>& squared_sound_speed,
                                                       const unsigned int prec_nvar) :
    m_ndim(ndim),
    m_nmom(nmom),
    m_mhd_enabled(mhd_enabled),
    m_solution(solution),
    m_solution_dPd(solution_dPd),
    m_squared_sound_speed(squared_sound_speed),
    m_prec_nvar(prec_nvar)
{
    if (m_ndim != 2 && m_ndim != 3) { throw std::invalid_argument("ndim must be 2 or 3"); }
    if (m_nmom != 2 && m_nmom != 3) { throw std::invalid_argument("nmom must be 2 or 3"); }
    if (m_nmom < m_ndim) { throw std::invalid_argument("nmom must be at least ndim"); }

    // Verify the size of solutions
    int num_solution_variables = m_solution.numVariables();
    if (num_solution_variables != m_num_sol_vars)
    {
        throw std::invalid_argument("Solution should contain exactly " + std::to_string(m_num_sol_vars) + " variables, found " +
                                    std::to_string(num_solution_variables));
    }
    int num_solution_dPd_variables = m_solution_dPd.numVariables();
    if (num_solution_dPd_variables != m_num_sol_dPd_vars)
    {
        throw std::invalid_argument("Solution gradients should contain exactly " + std::to_string(m_num_sol_dPd_vars) +
                                    " variables, found " + std::to_string(num_solution_dPd_variables));
    }

    if (solution.ni() != solution_dPd.ni() || solution.nj() != solution_dPd.nj() || solution.nk() != solution_dPd.nk())
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
    const Array3D<double> solution_dPdrho    = m_solution_dPd.singleVariableView(n_dPdrho());
    const Array3D<double> solution_dPdenergy = m_solution_dPd.singleVariableView(n_dPdenergy());

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
                                                        const Array3D<double>& dp_rhs_press,
                                                        Array3D<double>& delta_energy)
{
    if (m_prec_nvar == 1) { calculateEnergyDelta(preconditioned_delta_pressure, dp_rhs_press, delta_energy); }
    else
    {
        preconditioned_delta_energy.deepCopy(delta_energy);
    }
}

void PhysicsBasedPreconditioner::calculateEnergyDelta(const Array3D<double>& preconditioned_delta_pressure,
                                                      const Array3D<double>& dp_rhs_press,
                                                      Array3D<double>& delta_energy)
{
    const Array3D<double> solution_pressure = m_solution.singleVariableView(n_pressure());
    const Array3D<double> solution_rho      = m_solution.singleVariableView(n_rho());

    Slab delta_e_slab = delta_energy.cellSlab();
    for (int k = delta_e_slab.start(2); k < delta_e_slab.end(2); k++)
    {
        for (int j = delta_e_slab.start(1); j < delta_e_slab.end(1); j++)
        {
            for (int i = delta_e_slab.start(0); i < delta_e_slab.end(0); i++)
            {
                double dt_LdP =
                    (preconditioned_delta_pressure(i, j, k) / m_dt - dp_rhs_press(i, j, k)) / m_squared_sound_speed(i, j, k);
                double rho = solution_rho(i, j, k);
                delta_energy(i, j, k) += +m_dt * (solution_pressure(i, j, k) / (rho * rho)) * dt_LdP;
            }
        }
    }
}

void PhysicsBasedPreconditioner::reconstructDensityDelta(const Array3D<double>& delta_pressure,
                                                         const Array3D<double>& delta_energy,
                                                         Array3D<double>& delta_rho)
{
    const Array3D<double> solution_dPdrho    = m_solution_dPd.singleVariableView(n_dPdrho());
    const Array3D<double> solution_dPdenergy = m_solution_dPd.singleVariableView(n_dPdenergy());

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

void PhysicsBasedPreconditioner::conservedToPrimitive(const GhostExchangePlan& plan,
                                                      const PackedVariables active_scalars,
                                                      const PackedVariables delta_conserved,
                                                      const PackedVariables delta_conserved_rho_scalars,
                                                      const PackedVariables delta_conserved_magnetic_field,
                                                      PackedVariables delta_primitive,
                                                      PackedVariables delta_primitive_scalars,
                                                      PackedVariables delta_primitive_magnetic_field)
{
    if (active_scalars.numVariables() != delta_conserved_rho_scalars.numVariables() ||
        active_scalars.numVariables() != delta_primitive_scalars.numVariables())
    {
        throw std::invalid_argument("The number of active scalars differs between the inputs");
    }
    if (m_mhd_enabled &&
        (delta_conserved_magnetic_field.numVariables() != 3 || delta_primitive_magnetic_field.numVariables() != 3))
    {
        throw std::invalid_argument("MHD requires three magnetic field components");
    }

    // rho and e from the stored solution, rho with its ghosts
    Array3DWithGhosts<double> sol_rho = m_solution.singleVariableView(n_rho());
    const Array3D<double> sol_e       = m_solution.singleVariableView(n_energy());
    const Array3D<double>& cv_rho     = delta_conserved.singleVariableView(n_rho());
    sol_rho.requireSameSlab(cv_rho, "solution rho and delta_conserved rho");

    // rho one cell beyond the rank, read by the face averages
    sol_rho.exchangeGhosts();

    // Exchange ghost cells for the conserved density
    Array3DWithGhosts<double> delta_rho_g(cv_rho, plan);
    delta_rho_g.exchangeGhosts();

    // Density
    Array3D<double> delta_primitive_rho = delta_primitive.singleVariableView(n_rho());
    delta_primitive_rho.requireSameSlab(cv_rho, "delta_primitive rho and delta_conserved rho");
    cv_rho.deepCopy(delta_primitive_rho);

    // Energy
    Array3D<double> delta_primitive_e           = delta_primitive.singleVariableView(n_energy());
    const Array3D<double> delta_conserved_rho_e = delta_conserved.singleVariableView(n_energy());
    delta_primitive_e.requireSameSlab(cv_rho, "delta_primitive e and delta_conserved rho");
    delta_primitive_e.requireSameSlab(delta_conserved_rho_e, "delta_primitive e and delta_conserved rho e");
    delta_primitive_e.requireSameSlab(sol_e, "delta_primitive e and solution e");
    {
        Array3D<double>& delta_e           = delta_primitive_e;
        const Array3D<double>& delta_rho_e = delta_conserved_rho_e;
        const Slab s                       = delta_e.cellSlab();
        for (int k = s.start(2); k < s.end(2); k++)
        {
            for (int j = s.start(1); j < s.end(1); j++)
            {
                for (int i = s.start(0); i < s.end(0); i++)
                {
                    delta_e(i, j, k) = (-sol_e(i, j, k) * cv_rho(i, j, k) + delta_rho_e(i, j, k)) / sol_rho(i, j, k);
                }
            }
        }
    }

    // Active scalars
    for (unsigned int scalar_index = 0; scalar_index < active_scalars.numVariables(); scalar_index++)
    {
        const Array3D<double> sol_phi       = active_scalars.singleVariableView(scalar_index);
        const Array3D<double> delta_rho_phi = delta_conserved_rho_scalars.singleVariableView(scalar_index);
        Array3D<double> delta_phi           = delta_primitive_scalars.singleVariableView(scalar_index);
        delta_phi.requireSameSlab(cv_rho, "delta_primitive scalar and delta_conserved rho");
        delta_phi.requireSameSlab(delta_rho_phi, "delta_primitive scalar and delta_conserved rho scalar");
        delta_phi.requireSameSlab(sol_phi, "delta_primitive scalar and solution scalar");
        const Slab s = delta_phi.cellSlab();
        for (int k = s.start(2); k < s.end(2); k++)
        {
            for (int j = s.start(1); j < s.end(1); j++)
            {
                for (int i = s.start(0); i < s.end(0); i++)
                {
                    delta_phi(i, j, k) = (delta_rho_phi(i, j, k) - sol_phi(i, j, k) * cv_rho(i, j, k)) / sol_rho(i, j, k);
                }
            }
        }
    }

    // Velocity: each component lives on the faces along its own axis, so convert it over that component's own nodes
    // rather than over the cells slab, which may omit a solved boundary face. Averaging delta rho to the first face, or
    // to a solved positive boundary face, reads the NEGATIVE or POSITIVE ghost of rho and delta_rho_g.
    std::array<unsigned int, 3> velocities_indices = {n_vel_i(), n_vel_j(), n_vel_k()};
    for (unsigned int ax = 0; ax < m_nmom; ax++)
    {
        Array3D<double> delta_v           = delta_primitive.singleVariableView(velocities_indices[ax]);
        const Array3D<double> delta_rho_v = delta_conserved.singleVariableView(velocities_indices[ax]);
        const Array3D<double> sol_v       = m_solution.singleVariableView(velocities_indices[ax]);
        delta_v.requireSameSlab(sol_v, "delta_primitive velocity and solution velocity");
        const Slab s = delta_v.cellSlab();
        // Neighbour cell on the negative side of the face along ax
        const int di = (ax == 0) ? 1 : 0, dj = (ax == 1) ? 1 : 0, dk = (ax == 2) ? 1 : 0;
        for (int k = s.start(2); k < s.end(2); k++)
        {
            for (int j = s.start(1); j < s.end(1); j++)
            {
                for (int i = s.start(0); i < s.end(0); i++)
                {
                    double rho_f, delta_rho_f;
                    if (ax < m_ndim)
                    {
                        rho_f       = 0.5 * (sol_rho(i, j, k) + sol_rho(i - di, j - dj, k - dk));
                        delta_rho_f = 0.5 * (delta_rho_g(i, j, k) + delta_rho_g(i - di, j - dj, k - dk));
                    }
                    else
                    {
                        // Invariant along axis k if ndim < 3
                        rho_f       = sol_rho(i, j, k);
                        delta_rho_f = cv_rho(i, j, k);
                    }
                    delta_v(i, j, k) = (-sol_v(i, j, k) * delta_rho_f + delta_rho_v(i, j, k)) / rho_f;
                }
            }
        }
    }

    // Magnetic field: the conversion is the identity. Copy over each component's own nodes rather than over the cells
    // slab: a component staggered along an axis may own a boundary face that the cells slab does not cover.
    if (m_mhd_enabled)
    {
        for (unsigned int ax = 0; ax < 3; ax++)
        {
            Array3D<double> delta_primitive_b       = delta_primitive_magnetic_field.singleVariableView(ax);
            const Array3D<double> delta_conserved_b = delta_conserved_magnetic_field.singleVariableView(ax);
            delta_primitive_b.requireSameSlab(delta_conserved_b, "delta_primitive B and delta_conserved B");
            delta_conserved_b.deepCopy(delta_primitive_b);
        }
    }
}
