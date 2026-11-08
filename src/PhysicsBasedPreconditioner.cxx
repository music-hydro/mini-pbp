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

void PhysicsBasedPreconditioner::assemblePressureMatrix(const unsigned int ndim,
                                                        const double dt,
                                                        Array3DWithGhosts<double>& rho,
                                                        const Array3D<double>& squared_sound_speed,
                                                        const Array3D<DirectionalArray<double>>& laplacian_coefficients)
{
    if (m_prec_nvar != 1)
    {
        throw std::invalid_argument("assemblePressureMatrix supports prec_nvar 1 only, found " + std::to_string(m_prec_nvar));
    }
    if (ndim != 2 && ndim != 3) { throw std::invalid_argument("ndim must be 2 or 3, found " + std::to_string(ndim)); }

    // The faces read rho one cell beyond the rank
    rho.exchangeGhosts();

    const GhostExchangePlan<double>& plan = rho.plan();
    const Slab global_slab                = plan.globalDomain();
    const Slab local_slab                 = plan.localDomain();
    const Slab cell_slab                  = rho.cellSlab();
    if (ndim == 2 && global_slab.extent(2) != 1) { throw std::invalid_argument("In 2D there must be a single cell along k"); }
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        if (cell_slab.extent(axis) != local_slab.extent(axis))
        {
            throw std::invalid_argument("The cells of rho must be the local domain of its plan");
        }
    }

    // The pressure is the only unknown with prec_nvar 1
    const unsigned int n_pressure    = 0;
    const size_t num_rows            = size_t(cell_slab.extent(0)) * cell_slab.extent(1) * cell_slab.extent(2);
    const size_t max_entries_per_row = 2 * ndim + 1;

    m_global_pressure_row_indices.clear();
    m_pressure_row_offsets.clear();
    m_global_pressure_column_indices.clear();
    m_pressure_matrix_for_solver.clear();
    m_global_pressure_row_indices.reserve(num_rows);
    m_pressure_row_offsets.reserve(num_rows + 1);
    m_global_pressure_column_indices.reserve(num_rows * max_entries_per_row);
    m_pressure_matrix_for_solver.reserve(num_rows * max_entries_per_row);
    m_pressure_row_offsets.push_back(0);

    for (int k = cell_slab.start(2); k < cell_slab.end(2); ++k)
    {
        for (int j = cell_slab.start(1); j < cell_slab.end(1); ++j)
        {
            for (int i = cell_slab.start(0); i < cell_slab.end(0); ++i)
            {
                // (pressure, pressure) matrix element
                DirectionalArray<double> coefficients = pressureCoefficients(ndim, rho, laplacian_coefficients(i, j, k), i, j, k);
                const DirectionalArray<double> weights = rho.neighbourLinkWeights(i, j, k, ndim);
                for (size_t face = 0; face < coefficients.size(); face++)
                {
                    coefficients[face] = coefficients[face] * weights[face];
                }
                const double J_PP = -dt * squared_sound_speed(i, j, k);

                const std::array<int, 3> position = {local_slab.start(0) - global_slab.start(0) + i - cell_slab.start(0),
                                                     local_slab.start(1) - global_slab.start(1) + j - cell_slab.start(1),
                                                     local_slab.start(2) - global_slab.start(2) + k - cell_slab.start(2)};
                const long long row               = globalVarIndex(position, n_pressure, global_slab);
                m_global_pressure_row_indices.push_back(row);
                setElement(row, 1.0 / dt - J_PP * coefSum(coefficients));
                for (unsigned int axis = 0; axis < ndim; axis++)
                {
                    for (const Direction direction : {Direction::NEGATIVE, Direction::POSITIVE})
                    {
                        const std::array<int, 3> neighbour = plan.neighbourCellPosition(
                            i - cell_slab.start(0), j - cell_slab.start(1), k - cell_slab.start(2), axis, direction);
                        setElement(globalVarIndex(neighbour, n_pressure, global_slab), J_PP * coefficients.get(axis, direction));
                    }
                }
                m_pressure_row_offsets.push_back(m_pressure_matrix_for_solver.size());
            }
        }
    }
}

DirectionalArray<double> PhysicsBasedPreconditioner::pressureCoefficients(const unsigned int ndim,
                                                                          const Array3DWithGhosts<double>& rho,
                                                                          const DirectionalArray<double>& laplacian_coefficients,
                                                                          const int i,
                                                                          const int j,
                                                                          const int k)
{
    // Coefficients for usual Laplacian operator
    DirectionalArray<double> coefficients = laplacian_coefficients;

    // Obtain density at cell center (rho) and faces (rho_face)
    const double rho_centre = rho(i, j, k);
    DirectionalArray<double> rho_face;
    rho_face.fill(rho_centre);
    rho_face.get(0, Direction::NEGATIVE) = (rho(i, j, k) + rho(i - 1, j, k)) * 0.5;
    rho_face.get(0, Direction::POSITIVE) = (rho(i + 1, j, k) + rho(i, j, k)) * 0.5;
    rho_face.get(1, Direction::NEGATIVE) = (rho(i, j, k) + rho(i, j - 1, k)) * 0.5;
    rho_face.get(1, Direction::POSITIVE) = (rho(i, j + 1, k) + rho(i, j, k)) * 0.5;
    if (ndim == 3)
    {
        rho_face.get(2, Direction::NEGATIVE) = (rho(i, j, k) + rho(i, j, k - 1)) * 0.5;
        rho_face.get(2, Direction::POSITIVE) = (rho(i, j, k + 1) + rho(i, j, k)) * 0.5;
    }

    // Correct Laplacian coefficients to obtain: rho * div(1/rho * grad(P))
    for (size_t face = 0; face < coefficients.size(); face++)
    {
        coefficients[face] = coefficients[face] * rho_centre / rho_face[face];
    }
    return coefficients;
}

double PhysicsBasedPreconditioner::coefSum(const DirectionalArray<double>& coefficients)
{
    // Make sure to add the two directions of an axis first to reduce catastrophic cancellations between axes
    double sum = 0.0;
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        sum += coefficients.get(axis, Direction::NEGATIVE) + coefficients.get(axis, Direction::POSITIVE);
    }
    return sum;
}

long long PhysicsBasedPreconditioner::globalVarIndex(const std::array<int, 3>& position,
                                                     const unsigned int var_index,
                                                     const Slab& global_slab) const
{
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        if (position[axis] < 0) { return -1; }
    }
    const long long ni = global_slab.extent(0), nj = global_slab.extent(1);
    return static_cast<long long>(var_index) +
           static_cast<long long>(m_prec_nvar) * (position[0] + ni * (position[1] + nj * static_cast<long long>(position[2])));
}

void PhysicsBasedPreconditioner::setElement(const long long column, const double value)
{
    if (column < 0) { return; }
    m_global_pressure_column_indices.push_back(column);
    m_pressure_matrix_for_solver.push_back(value);
}
