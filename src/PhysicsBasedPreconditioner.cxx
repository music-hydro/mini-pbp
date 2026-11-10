#include "PhysicsBasedPreconditioner.h"

#include "Array3D.h"
#include "Slab.h"
#include <stdexcept>
#include <string>

namespace
{
    /**
     * @brief Throw if two views do not cover the same nodes (MUSIC `ASSERT(a%nodes == b%nodes)`).
     */
    void requireSameSlab(const Array3D<double>& a, const Array3D<double>& b, const std::string& what)
    {
        const Slab sa = a.cellSlab(), sb = b.cellSlab();
        if (sa.start() != sb.start() || sa.end() != sb.end())
        {
            throw std::invalid_argument("cons_to_prim: " + what + " do not cover the same nodes");
        }
    }
}  // namespace

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
    const Array3D<double> solution_pressure = m_solution.singleVariableView(m_n_pressure);
    const Array3D<double> solution_rho      = m_solution.singleVariableView(m_n_rho);

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

void PhysicsBasedPreconditioner::consToPrim(const unsigned int ndim,
                                            const unsigned int nmom,
                                            const bool mhd_enabled,
                                            GhostExchangePlan<double>& plan,

                                            const std::array<const Array3D<double>*, 3>& vel,
                                            const std::vector<const Array3D<double>*>& active_scalars,

                                            const Array3D<double>& delta_cons_rho,
                                            const Array3D<double>& delta_cons_rho_e,
                                            const std::array<const Array3D<double>*, 3>& delta_cons_rho_vel,
                                            const std::vector<const Array3D<double>*>& delta_cons_rho_scalars,
                                            const std::array<const Array3D<double>*, 3>& delta_cons_magnetic_field,

                                            Array3D<double>& delta_prim_rho,
                                            Array3D<double>& delta_prim_e,
                                            const std::array<Array3D<double>*, 3>& delta_prim_vel,
                                            const std::vector<Array3D<double>*>& delta_prim_scalars,
                                            const std::array<Array3D<double>*, 3>& delta_prim_magnetic_field)
{
    if (ndim != 2 && ndim != 3) { throw std::invalid_argument("cons_to_prim: ndim must be 2 or 3"); }
    if (nmom != 2 && nmom != 3) { throw std::invalid_argument("cons_to_prim: nmom must be 2 or 3"); }
    if (nmom < ndim) { throw std::invalid_argument("cons_to_prim: nmom must be at least ndim"); }
    if (active_scalars.size() != delta_cons_rho_scalars.size() || active_scalars.size() != delta_prim_scalars.size())
    {
        throw std::invalid_argument("cons_to_prim: the number of active scalars differs between the inputs");
    }

    // rho and e from the stored solution, rho with its ghosts
    Array3DWithGhosts<double> sol_rho(m_solution.singleVariableView(m_n_rho), plan);
    const Array3D<double> sol_e   = m_solution.singleVariableView(m_n_energy);
    const Array3D<double>& cv_rho = delta_cons_rho;
    requireSameSlab(sol_rho, cv_rho, "solution rho and delta_cons rho");

    // rho one cell beyond the rank, read by the face averages
    sol_rho.exchangeGhosts();

    // delta rho one cell beyond the rank: the halo from the neighbour ranks, or the boundary condition fill
    Array3DWithGhosts<double> delta_rho_g(cv_rho, plan);
    delta_rho_g.exchangeGhosts();

    // Density
    requireSameSlab(delta_prim_rho, cv_rho, "delta_prim rho and delta_cons rho");
    cv_rho.deepCopy(delta_prim_rho);

    // Energy
    requireSameSlab(delta_prim_e, cv_rho, "delta_prim e and delta_cons rho");
    requireSameSlab(delta_prim_e, delta_cons_rho_e, "delta_prim e and delta_cons rho e");
    requireSameSlab(delta_prim_e, sol_e, "delta_prim e and solution e");
    {
        Array3D<double>& delta_e           = delta_prim_e;
        const Array3D<double>& delta_rho_e = delta_cons_rho_e;
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
    for (size_t isc = 0; isc < active_scalars.size(); isc++)
    {
        const Array3D<double>& sol_phi       = *active_scalars[isc];
        const Array3D<double>& delta_rho_phi = *delta_cons_rho_scalars[isc];
        Array3D<double>& delta_phi           = *delta_prim_scalars[isc];
        requireSameSlab(delta_phi, cv_rho, "delta_prim scalar and delta_cons rho");
        requireSameSlab(delta_phi, delta_rho_phi, "delta_prim scalar and delta_cons rho scalar");
        requireSameSlab(delta_phi, sol_phi, "delta_prim scalar and solution scalar");
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
    for (unsigned int ax = 0; ax < nmom; ax++)
    {
        Array3D<double>& delta_v           = *delta_prim_vel[ax];
        const Array3D<double>& delta_rho_v = *delta_cons_rho_vel[ax];
        const Array3D<double>& sol_v       = *vel[ax];
        requireSameSlab(delta_v, sol_v, "delta_prim velocity and solution velocity");
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
                    if (ax < ndim)
                    {
                        rho_f       = 0.5 * (sol_rho(i, j, k) + sol_rho(i - di, j - dj, k - dk));
                        delta_rho_f = 0.5 * (delta_rho_g(i - di, j - dj, k - dk) + delta_rho_g(i, j, k));
                    }
                    else
                    {
                        // Invariant along axis k if ndim < 3
                        rho_f       = sol_rho(i, j, k);
                        delta_rho_f = cv_rho(i, j, k);
                    }
                    const double v   = sol_v(i, j, k);
                    delta_v(i, j, k) = (-v * delta_rho_f + delta_rho_v(i, j, k)) / rho_f;
                }
            }
        }
    }

    // Magnetic field: the conversion is the identity. Copy over each component's own nodes rather than over the cells
    // slab: a component staggered along an axis may own a boundary face that the cells slab does not cover.
    if (mhd_enabled)
    {
        for (unsigned int ax = 0; ax < 3; ax++)
        {
            requireSameSlab(*delta_prim_magnetic_field[ax], *delta_cons_magnetic_field[ax], "delta_prim B and delta_cons B");
            delta_cons_magnetic_field[ax]->deepCopy(*delta_prim_magnetic_field[ax]);
        }
    }
}
