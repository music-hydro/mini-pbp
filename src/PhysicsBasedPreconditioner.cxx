#include "PhysicsBasedPreconditioner.h"

#include "Array3D.h"
#include "DirectionalArray.h"
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
    m_prec_nvar(prec_nvar)
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
    const Array3D<double> solution_dPdrho    = m_solution_dPd.singleVariableView(SolutionDpdVariableIndices::DPDRHO);
    const Array3D<double> solution_dPdenergy = m_solution_dPd.singleVariableView(SolutionDpdVariableIndices::DPDENERGY);

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
    const Array3D<double> solution_pressure = m_solution.singleVariableView(SolutionVariableIndices::PRESSURE);
    const Array3D<double> solution_rho      = m_solution.singleVariableView(SolutionVariableIndices::RHO);

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
    const Array3D<double> solution_dPdrho    = m_solution_dPd.singleVariableView(SolutionDpdVariableIndices::DPDRHO);
    const Array3D<double> solution_dPdenergy = m_solution_dPd.singleVariableView(SolutionDpdVariableIndices::DPDENERGY);

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

void PhysicsBasedPreconditioner::assemblePressureMatrix(Array3DWithGhosts<double>& solution_rho,
                                                        const Array3D<DirectionalArray<double>>& laplacian_coefficients)
{
    solution_rho.exchangeGhosts();

    // Compute squared sound speed
    calculateSquaredSoundSpeed();

    // Fill pressure
    Slab global_slab = solution_rho.plan().globalDomain();
    Slab cell_slab   = solution_rho.cellSlab();
    for (int k = cell_slab.start(2); k < cell_slab.end(2); ++k)
    {
        for (int j = cell_slab.start(1); j < cell_slab.end(1); ++j)
        {
            for (int i = cell_slab.start(0); i < cell_slab.end(0); ++i)
            {

                const DirectionalArray<double> pressure_coefficients = pressureCoefficients(
                    i, j, k, solution_rho, laplacian_coefficients(i, j, k), solution_rho.neighbourLinkWeights(i, j, k));
                const double J_PP = m_dt * m_squared_sound_speed(i, j, k);

                double pressure_coefficient_sum = 0.0;
                for (int axis = 0; axis < 3; axis++)
                    pressure_coefficient_sum += pressure_coefficients.get(axis, Direction::NEGATIVE) +
                                                pressure_coefficients.get(axis, Direction::POSITIVE);
                if (pressure_coefficient_sum = 0.0) throw std::runtime_error("pressure_coefficient_sum is zero");

                // Add the current cell's value to the pressure matrix
                const unsigned long long global_index = globalVarIndex(i, j, k, SolutionVariableIndices::PRESSURE, global_slab);
                m_global_pressure_row_indices.push_back(global_index);
                m_global_pressure_column_indices.push_back(global_index);
                m_pressure_matrix_for_solver.push_back(1.0 / m_dt - J_PP * pressure_coefficient_sum);

                // Add neighbour cell values to the pressure matrix
                for (int axis = 0; axis < 3; ++axis)
                {
                    // Determine the neighbour cell positions
                    std::array<int, 3> negative_neighbour_position =
                        solution_rho.plan().neighbourCellPosition(i, j, k, axis, Direction::NEGATIVE);
                    std::array<int, 3> positive_neighbour_position =
                        solution_rho.plan().neighbourCellPosition(i, j, k, axis, Direction::POSITIVE);

                    // Get the global var indices for each neighbour
                    unsigned long long global_negative_neighbour_var_index = globalVarIndex(negative_neighbour_position[0],
                                                                                            negative_neighbour_position[1],
                                                                                            negative_neighbour_position[2],
                                                                                            SolutionVariableIndices::PRESSURE,
                                                                                            global_slab);
                    unsigned long long global_positive_neighbour_var_index = globalVarIndex(positive_neighbour_position[0],
                                                                                            positive_neighbour_position[1],
                                                                                            positive_neighbour_position[2],
                                                                                            SolutionVariableIndices::PRESSURE,
                                                                                            global_slab);

                    // Add the global indices to the column index map
                    m_global_pressure_column_indices.push_back(global_negative_neighbour_var_index);
                    m_global_pressure_column_indices.push_back(global_positive_neighbour_var_index);

                    // Push the value for each neighbour into the pressure matrix
                    m_pressure_matrix_for_solver.push_back(J_PP * pressure_coefficients.get(axis, Direction::NEGATIVE));
                    m_pressure_matrix_for_solver.push_back(J_PP * pressure_coefficients.get(axis, Direction::POSITIVE));
                }
                m_global_pressure_row_indices.push_back(m_pressure_matrix_for_solver.size());
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
        coefficients[signed_axis] *= rho / rho_face[signed_axis] * weights[signed_axis];

    return coefficients;
}
