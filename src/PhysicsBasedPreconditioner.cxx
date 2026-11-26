#include "PhysicsBasedPreconditioner.h"

#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "DirectionalArray.h"
#include "Slab.h"
#include <stdexcept>
#include <string>

PhysicsBasedPreconditioner::PhysicsBasedPreconditioner(const double dt,
                                                       const unsigned int ndim,
                                                       const unsigned int nmom,
                                                       const bool mhd_enabled,
                                                       PackedVariablesWithGhosts& solution,
                                                       PackedVariables& solution_dPd,
                                                       Array3D<double>& squared_sound_speed,
                                                       const unsigned int prec_nvar) :
    m_dt(dt),
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

void PhysicsBasedPreconditioner::assemblePressureMatrix(Array3DWithGhosts<double>& solution_rho,
                                                        const Array3D<DirectionalArray<double>>& laplacian_coefficients,
                                                        const Array3D<double>& squared_sound_speed)
{
    const GhostExchangePlan& plan = solution_rho.plan();
    solution_rho.exchangeGhosts();

    m_global_pressure_column_indices.clear();
    m_global_pressure_row_indices.clear();
    m_global_pressure_row_offsets.clear();
    m_pressure_matrix_for_solver.clear();

    m_global_pressure_row_offsets.push_back(0);

    // Fill pressure
    Slab global_slab = plan.globalDomain();
    Slab cell_slab   = solution_rho.cellSlab();
    for (int k = cell_slab.start(2); k < cell_slab.end(2); ++k)
    {
        for (int j = cell_slab.start(1); j < cell_slab.end(1); ++j)
        {
            for (int i = cell_slab.start(0); i < cell_slab.end(0); ++i)
            {

                const DirectionalArray<double> pressure_coefficients = pressureCoefficients(
                    i, j, k, solution_rho, laplacian_coefficients(i, j, k), solution_rho.neighbourLinkWeights(i, j, k));
                const double J_PP = -m_dt * squared_sound_speed(i, j, k);

                double pressure_coefficient_sum = 0.0;
                for (int axis = 0; axis < 3; axis++)
                    pressure_coefficient_sum += pressure_coefficients.get(axis, Direction::NEGATIVE) +
                                                pressure_coefficients.get(axis, Direction::POSITIVE);
                if (pressure_coefficient_sum == 0.0) throw std::runtime_error("pressure_coefficient_sum is zero");

                // Add the current cell's value to the pressure matrix
                const unsigned long long global_index = globalVarIndex(i + plan.localDomain().start(0),
                                                                       j + plan.localDomain().start(1),
                                                                       k + plan.localDomain().start(2),
                                                                       n_pressure(),
                                                                       global_slab);
                m_global_pressure_row_indices.push_back(global_index);
                m_global_pressure_column_indices.push_back(global_index);
                m_pressure_matrix_for_solver.push_back(1.0 / m_dt - J_PP * pressure_coefficient_sum);

                // Add neighbour cell values to the pressure matrix
                for (int axis = 0; axis < 3; ++axis)
                {
                    // Determine the neighbour cell positions
                    std::array<int, 3> negative_neighbour_position =
                        plan.neighbourCellPosition(i, j, k, axis, Direction::NEGATIVE);
                    std::array<int, 3> positive_neighbour_position =
                        plan.neighbourCellPosition(i, j, k, axis, Direction::POSITIVE);

                    // Get the global var indices for each neighbour
                    unsigned long long global_negative_neighbour_var_index = globalVarIndex(negative_neighbour_position[0],
                                                                                            negative_neighbour_position[1],
                                                                                            negative_neighbour_position[2],
                                                                                            n_pressure(),
                                                                                            global_slab);
                    unsigned long long global_positive_neighbour_var_index = globalVarIndex(positive_neighbour_position[0],
                                                                                            positive_neighbour_position[1],
                                                                                            positive_neighbour_position[2],
                                                                                            n_pressure(),
                                                                                            global_slab);

                    // Add the global indices to the column index map
                    m_global_pressure_column_indices.push_back(global_negative_neighbour_var_index);
                    m_global_pressure_column_indices.push_back(global_positive_neighbour_var_index);

                    // Push the value for each neighbour into the pressure matrix
                    m_pressure_matrix_for_solver.push_back(J_PP * pressure_coefficients.get(axis, Direction::NEGATIVE));
                    m_pressure_matrix_for_solver.push_back(J_PP * pressure_coefficients.get(axis, Direction::POSITIVE));
                }
                m_global_pressure_row_offsets.push_back(m_pressure_matrix_for_solver.size());
            }
        }
    }
}

DirectionalArray<double> PhysicsBasedPreconditioner::pressureCoefficients(const int& i,
                                                                          const int& j,
                                                                          const int& k,
                                                                          const Array3DWithGhosts<double>& solution_rho,
                                                                          const DirectionalArray<double>& laplacian_coefficients,
                                                                          const DirectionalArray<double>& weights)
{
    // Coefficients for usual Laplacian operator
    DirectionalArray<double> coefficients = laplacian_coefficients;
    double rho                            = solution_rho(i, j, k);
    DirectionalArray<double> rho_face;

    rho_face.get(0, Direction::NEGATIVE) = (solution_rho(i, j, k) + solution_rho(i - 1, j, k)) * 0.5;
    rho_face.get(0, Direction::POSITIVE) = (solution_rho(i + 1, j, k) + solution_rho(i, j, k)) * 0.5;
    rho_face.get(1, Direction::NEGATIVE) = (solution_rho(i, j, k) + solution_rho(i, j - 1, k)) * 0.5;
    rho_face.get(1, Direction::POSITIVE) = (solution_rho(i, j + 1, k) + solution_rho(i, j, k)) * 0.5;
    // This is only really needed if we are working in 3D but so far we
    // have assumed 3D and handled 2D by setting the length of k to 1.
    rho_face.get(2, Direction::NEGATIVE) = (solution_rho(i, j, k) + solution_rho(i, j, k - 1)) * 0.5;
    rho_face.get(2, Direction::POSITIVE) = (solution_rho(i, j, k + 1) + solution_rho(i, j, k)) * 0.5;

    for (int signed_axis = 0; signed_axis < 6; ++signed_axis)
        coefficients[signed_axis] *= (rho / rho_face[signed_axis]) * weights[signed_axis];

    return coefficients;
}
