#include "PressureMatrix.h"

#include <stdexcept>
#include <string>

namespace
{
    /**
     * @brief Element of a solution field with one ghost layer at local cell (i, j, k) shifted by (di, dj, dk).
     *
     * @param k_ghosts - Ghost layers along k (1 in 3D, 0 in 2D).
     */
    double ghosted(const Array3D<double>& field,
                   const unsigned int i,
                   const unsigned int j,
                   const unsigned int k,
                   const int di,
                   const int dj,
                   const int dk,
                   const unsigned int k_ghosts)
    { return field(i + 1 + di, j + 1 + dj, k + k_ghosts + dk); }
}  // namespace

PressureMatrix::PressureMatrix(const Decomposition& decomposition, const unsigned int prec_nvar) :
    m_decomposition(decomposition),
    m_prec_nvar(prec_nvar)
{
    if (m_prec_nvar != 1 && m_prec_nvar != 2)
    {
        throw std::invalid_argument("prec_nvar must be 1 or 2, found " + std::to_string(m_prec_nvar));
    }
    if (m_decomposition.ndim != 2 && m_decomposition.ndim != 3)
    {
        throw std::invalid_argument("ndim must be 2 or 3, found " + std::to_string(m_decomposition.ndim));
    }
    if (m_decomposition.ndim == 2 && (m_decomposition.local_cells[2] != 1 || m_decomposition.global_cells[2] != 1))
    {
        throw std::invalid_argument("In 2D there must be a single cell along k");
    }
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        if (m_decomposition.global_start[axis] < 0 ||
            m_decomposition.global_start[axis] + m_decomposition.local_cells[axis] > m_decomposition.global_cells[axis])
        {
            throw std::invalid_argument("The cells of this rank must lie inside the domain along axis " + std::to_string(axis));
        }
    }
}

PressureMatrix::GlobalIndex
    PressureMatrix::globalDofIndex(const GlobalIndex i, const GlobalIndex j, const GlobalIndex k, const unsigned int ivar) const
{
    const std::array<GlobalIndex, 3>& n = m_decomposition.global_cells;
    return GlobalIndex(ivar) + GlobalIndex(m_prec_nvar) * (i + n[0] * (j + n[1] * k));
}

PressureMatrix::GlobalIndex PressureMatrix::neighbourDofIndex(const unsigned int i,
                                                              const unsigned int j,
                                                              const unsigned int k,
                                                              const unsigned int ivar,
                                                              const unsigned int axis,
                                                              const Direction direction) const
{
    std::array<GlobalIndex, 3> position = {
        m_decomposition.global_start[0] + i, m_decomposition.global_start[1] + j, m_decomposition.global_start[2] + k};
    position[axis] += (direction == Direction::NEGATIVE) ? -1 : 1;

    // Along a periodic axis the neighbour is the periodic image, otherwise outside the domain there is none
    const GlobalIndex n = m_decomposition.global_cells[axis];
    if (m_decomposition.periodic[axis]) { position[axis] = ((position[axis] % n) + n) % n; }
    if (position[axis] < 0 || position[axis] >= n) { return -1; }

    return globalDofIndex(position[0], position[1], position[2], ivar);
}

DirectionalArray<double> PressureMatrix::pressureCoefficients(const SolutionFields& solution,
                                                              const LaplacianCoefficients& laplacian,
                                                              const unsigned int i,
                                                              const unsigned int j,
                                                              const unsigned int k) const
{
    const unsigned int kg       = (m_decomposition.ndim == 3) ? 1 : 0;
    const Array3D<double>& vrho = *solution.rho;

    // Coefficients for usual Laplacian operator
    DirectionalArray<double> coef = laplacian(i, j, k);

    // Obtain density at cell center (rho) and faces (rho_f)
    const double rho = ghosted(vrho, i, j, k, 0, 0, 0, kg);
    DirectionalArray<double> rho_f;
    rho_f.fill(rho);
    rho_f.get(0, Direction::NEGATIVE) = (ghosted(vrho, i, j, k, 0, 0, 0, kg) + ghosted(vrho, i, j, k, -1, 0, 0, kg)) * 0.5;
    rho_f.get(0, Direction::POSITIVE) = (ghosted(vrho, i, j, k, 1, 0, 0, kg) + ghosted(vrho, i, j, k, 0, 0, 0, kg)) * 0.5;
    rho_f.get(1, Direction::NEGATIVE) = (ghosted(vrho, i, j, k, 0, 0, 0, kg) + ghosted(vrho, i, j, k, 0, -1, 0, kg)) * 0.5;
    rho_f.get(1, Direction::POSITIVE) = (ghosted(vrho, i, j, k, 0, 1, 0, kg) + ghosted(vrho, i, j, k, 0, 0, 0, kg)) * 0.5;
    if (m_decomposition.ndim >= 3)
    {
        rho_f.get(2, Direction::NEGATIVE) = (ghosted(vrho, i, j, k, 0, 0, 0, kg) + ghosted(vrho, i, j, k, 0, 0, -1, kg)) * 0.5;
        rho_f.get(2, Direction::POSITIVE) = (ghosted(vrho, i, j, k, 0, 0, 1, kg) + ghosted(vrho, i, j, k, 0, 0, 0, kg)) * 0.5;
    }

    // Correct Laplacian coefficients to obtain: rho * div(1/rho * grad(P))
    for (size_t face = 0; face < coef.size(); face++)
    {
        coef[face] = coef[face] * rho / rho_f[face];
    }
    return coef;
}

DirectionalArray<double>
    PressureMatrix::neighbourLinksWeights(const unsigned int i, const unsigned int j, const unsigned int k) const
{
    const std::array<unsigned int, 3> index = {i, j, k};
    DirectionalArray<double> weights;
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        const double used                      = (axis < m_decomposition.ndim) ? 1.0 : 0.0;
        weights.get(axis, Direction::NEGATIVE) = used;
        weights.get(axis, Direction::POSITIVE) = used;
        if (index[axis] == 0 && m_decomposition.reflective.get(axis, Direction::NEGATIVE))
        {
            weights.get(axis, Direction::NEGATIVE) = 0.0;
        }
        if (index[axis] == m_decomposition.local_cells[axis] - 1 && m_decomposition.reflective.get(axis, Direction::POSITIVE))
        {
            weights.get(axis, Direction::POSITIVE) = 0.0;
        }
    }
    return weights;
}

double PressureMatrix::coefSum(const DirectionalArray<double>& coefficients)
{
    // Make sure to add the two directions of an axis first to reduce catastrophic cancellations between axes
    double sum = 0.0;
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        sum += coefficients.get(axis, Direction::NEGATIVE) + coefficients.get(axis, Direction::POSITIVE);
    }
    return sum;
}

void PressureMatrix::setElement(const GlobalIndex column, const double value)
{
    if (column < 0) { return; }
    m_column_global_indices.push_back(column);
    m_values.push_back(value);
}

void PressureMatrix::endRow()
{ m_row_offsets.push_back(m_values.size()); }

void PressureMatrix::assemble(const double dt,
                              const SolutionFields& solution,
                              const Array3D<double>& squared_sound_speed,
                              const LaplacianCoefficients& laplacian)
{
    if (solution.rho == nullptr) { throw std::invalid_argument("The density is required"); }
    if (m_prec_nvar == 2 &&
        (solution.pressure == nullptr || solution.dPde == nullptr || solution.chi == nullptr || solution.dTde == nullptr))
    {
        throw std::invalid_argument("prec_nvar = 2 requires pressure, dPde, chi and dTde");
    }

    const unsigned int ndim                 = m_decomposition.ndim;
    const std::array<unsigned int, 3>& nc   = m_decomposition.local_cells;
    const std::array<GlobalIndex, 3>& start = m_decomposition.global_start;
    const unsigned int kg                   = (ndim == 3) ? 1 : 0;
    const unsigned int n_pressure           = 0;
    const unsigned int n_energy             = 1;
    const size_t num_rows                   = size_t(nc[0]) * nc[1] * nc[2] * m_prec_nvar;
    const size_t max_entries_per_row        = (2 * ndim + 1) * m_prec_nvar;

    m_row_global_indices.clear();
    m_row_offsets.clear();
    m_column_global_indices.clear();
    m_values.clear();
    m_row_global_indices.reserve(num_rows);
    m_row_offsets.reserve(num_rows + 1);
    m_column_global_indices.reserve(num_rows * max_entries_per_row);
    m_values.reserve(num_rows * max_entries_per_row);
    m_row_offsets.push_back(0);

    for (unsigned int k = 0; k < nc[2]; k++)
    {
        for (unsigned int j = 0; j < nc[1]; j++)
        {
            for (unsigned int i = 0; i < nc[0]; i++)
            {
                // (pressure, pressure) matrix element
                DirectionalArray<double> coef          = pressureCoefficients(solution, laplacian, i, j, k);
                const DirectionalArray<double> weights = neighbourLinksWeights(i, j, k);
                for (size_t face = 0; face < coef.size(); face++)
                {
                    coef[face] = coef[face] * weights[face];
                }
                const double J_PP = -dt * squared_sound_speed(i, j, k);

                const GlobalIndex row_p = globalDofIndex(start[0] + i, start[1] + j, start[2] + k, n_pressure);
                m_row_global_indices.push_back(row_p);
                setElement(row_p, 1.0 / dt - J_PP * coefSum(coef));
                for (unsigned int axis = 0; axis < ndim; axis++)
                {
                    for (const Direction direction : {Direction::NEGATIVE, Direction::POSITIVE})
                    {
                        setElement(neighbourDofIndex(i, j, k, n_pressure, axis, direction), J_PP * coef.get(axis, direction));
                    }
                }

                if (m_prec_nvar == 1)
                {
                    endRow();
                    continue;
                }

                // Also set (P,e), (e,P) and (e,e) elements. Geometric coefficients for the thermal part:
                // e_coef = coef * (dT/de * chi)_face
                const Array3D<double>& dTde = *solution.dTde;
                const Array3D<double>& chi  = *solution.chi;
                auto dTde_chi               = [&](const int di, const int dj, const int dk)
                { return ghosted(dTde, i, j, k, di, dj, dk, kg) * ghosted(chi, i, j, k, di, dj, dk, kg); };

                DirectionalArray<double> e_coef = {};
                e_coef.get(0, Direction::NEGATIVE) =
                    coef.get(0, Direction::NEGATIVE) * 0.5 * (dTde_chi(0, 0, 0) + dTde_chi(-1, 0, 0));
                e_coef.get(0, Direction::POSITIVE) =
                    coef.get(0, Direction::POSITIVE) * 0.5 * (dTde_chi(0, 0, 0) + dTde_chi(1, 0, 0));
                e_coef.get(1, Direction::NEGATIVE) =
                    coef.get(1, Direction::NEGATIVE) * 0.5 * (dTde_chi(0, 0, 0) + dTde_chi(0, -1, 0));
                e_coef.get(1, Direction::POSITIVE) =
                    coef.get(1, Direction::POSITIVE) * 0.5 * (dTde_chi(0, 0, 0) + dTde_chi(0, 1, 0));
                if (ndim == 3)
                {
                    e_coef.get(2, Direction::NEGATIVE) =
                        coef.get(2, Direction::NEGATIVE) * 0.5 * (dTde_chi(0, 0, 0) + dTde_chi(0, 0, -1));
                    e_coef.get(2, Direction::POSITIVE) =
                        coef.get(2, Direction::POSITIVE) * 0.5 * (dTde_chi(0, 0, 0) + dTde_chi(0, 0, 1));
                }

                const double rho  = ghosted(*solution.rho, i, j, k, 0, 0, 0, kg);
                const double J_Pe = -ghosted(*solution.dPde, i, j, k, 0, 0, 0, kg) / rho;  // equal to -(Gamma3 - 1)
                const double J_eP = -dt * ghosted(*solution.pressure, i, j, k, 0, 0, 0, kg) / (rho * rho);
                const double J_ee = -1.0 / rho;

                // (pressure, energy)
                setElement(globalDofIndex(start[0] + i, start[1] + j, start[2] + k, n_energy), -J_Pe * coefSum(e_coef));
                for (unsigned int axis = 0; axis < ndim; axis++)
                {
                    for (const Direction direction : {Direction::NEGATIVE, Direction::POSITIVE})
                    {
                        setElement(neighbourDofIndex(i, j, k, n_energy, axis, direction), J_Pe * e_coef.get(axis, direction));
                    }
                }
                endRow();

                // (energy, :) elements
                const GlobalIndex row_e = globalDofIndex(start[0] + i, start[1] + j, start[2] + k, n_energy);
                m_row_global_indices.push_back(row_e);

                setElement(row_p, -J_eP * coefSum(coef));
                for (unsigned int axis = 0; axis < ndim; axis++)
                {
                    for (const Direction direction : {Direction::NEGATIVE, Direction::POSITIVE})
                    {
                        setElement(neighbourDofIndex(i, j, k, n_pressure, axis, direction), J_eP * coef.get(axis, direction));
                    }
                }

                setElement(row_e, 1.0 / dt - J_ee * coefSum(e_coef));
                for (unsigned int axis = 0; axis < ndim; axis++)
                {
                    for (const Direction direction : {Direction::NEGATIVE, Direction::POSITIVE})
                    {
                        setElement(neighbourDofIndex(i, j, k, n_energy, axis, direction), J_ee * e_coef.get(axis, direction));
                    }
                }
                endRow();
            }
        }
    }
}
