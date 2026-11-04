#include "PressureMatrix.h"

#include <stdexcept>
#include <string>

namespace
{
    /**
     * @brief Element of a solution field with one ghost layer at local cell (i, j, k) shifted by (di, dj, dk).
     *
     * @param kGhosts - Ghost layers along k (1 in 3D, 0 in 2D).
     */
    double ghosted(const Array3D<double>& field,
                   const unsigned int i,
                   const unsigned int j,
                   const unsigned int k,
                   const int di,
                   const int dj,
                   const int dk,
                   const unsigned int kGhosts)
    { return field(i + 1 + di, j + 1 + dj, k + kGhosts + dk); }
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
    if (m_decomposition.ndim == 2 && (m_decomposition.localCells[2] != 1 || m_decomposition.globalCells[2] != 1))
    {
        throw std::invalid_argument("In 2D there must be a single cell along k");
    }
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        if (m_decomposition.globalStart[axis] < 0 ||
            m_decomposition.globalStart[axis] + m_decomposition.localCells[axis] > m_decomposition.globalCells[axis])
        {
            throw std::invalid_argument("The cells of this rank must lie inside the domain along axis " + std::to_string(axis));
        }
    }
}

PressureMatrix::GlobalIndex
    PressureMatrix::globalDofIndex(const GlobalIndex i, const GlobalIndex j, const GlobalIndex k, const unsigned int ivar) const
{
    const std::array<GlobalIndex, 3>& n = m_decomposition.globalCells;
    return GlobalIndex(ivar) + GlobalIndex(m_prec_nvar) * (i + n[0] * (j + n[1] * k));
}

PressureMatrix::GlobalIndex PressureMatrix::neighbourDofIndex(const unsigned int i,
                                                              const unsigned int j,
                                                              const unsigned int k,
                                                              const unsigned int ivar,
                                                              const unsigned int axis,
                                                              const unsigned int side) const
{
    std::array<GlobalIndex, 3> position = {
        m_decomposition.globalStart[0] + i, m_decomposition.globalStart[1] + j, m_decomposition.globalStart[2] + k};
    position[axis] += (side == SIDE_L) ? -1 : 1;

    // Along a periodic axis the neighbour is the periodic image, otherwise outside the domain there is none
    const GlobalIndex n = m_decomposition.globalCells[axis];
    if (m_decomposition.periodic[axis]) { position[axis] = ((position[axis] % n) + n) % n; }
    if (position[axis] < 0 || position[axis] >= n) { return -1; }

    return globalDofIndex(position[0], position[1], position[2], ivar);
}

PressureMatrix::Coefficients PressureMatrix::pressureCoefficients(const SolutionFields& solution,
                                                                  const LaplacianCoefficients& laplacian,
                                                                  const unsigned int i,
                                                                  const unsigned int j,
                                                                  const unsigned int k) const
{
    const unsigned int kg       = (m_decomposition.ndim == 3) ? 1 : 0;
    const Array3D<double>& vrho = *solution.rho;

    // Coefficients for usual Laplacian operator
    Coefficients coef;
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        for (unsigned int side = SIDE_L; side <= SIDE_R; side++)
        {
            coef[axis][side] = (*laplacian[axis][side])(i, j, k);
        }
    }

    // Obtain density at cell center (rho) and faces (rho_f)
    const double rho = ghosted(vrho, i, j, k, 0, 0, 0, kg);
    Coefficients rho_f;
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        rho_f[axis] = {rho, rho};
    }
    rho_f[0][SIDE_L] = (ghosted(vrho, i, j, k, 0, 0, 0, kg) + ghosted(vrho, i, j, k, -1, 0, 0, kg)) * 0.5;
    rho_f[0][SIDE_R] = (ghosted(vrho, i, j, k, 1, 0, 0, kg) + ghosted(vrho, i, j, k, 0, 0, 0, kg)) * 0.5;
    rho_f[1][SIDE_L] = (ghosted(vrho, i, j, k, 0, 0, 0, kg) + ghosted(vrho, i, j, k, 0, -1, 0, kg)) * 0.5;
    rho_f[1][SIDE_R] = (ghosted(vrho, i, j, k, 0, 1, 0, kg) + ghosted(vrho, i, j, k, 0, 0, 0, kg)) * 0.5;
    if (m_decomposition.ndim >= 3)
    {
        rho_f[2][SIDE_L] = (ghosted(vrho, i, j, k, 0, 0, 0, kg) + ghosted(vrho, i, j, k, 0, 0, -1, kg)) * 0.5;
        rho_f[2][SIDE_R] = (ghosted(vrho, i, j, k, 0, 0, 1, kg) + ghosted(vrho, i, j, k, 0, 0, 0, kg)) * 0.5;
    }

    // Correct Laplacian coefficients to obtain: rho * div(1/rho * grad(P))
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        for (unsigned int side = SIDE_L; side <= SIDE_R; side++)
        {
            coef[axis][side] = coef[axis][side] * rho / rho_f[axis][side];
        }
    }
    return coef;
}

PressureMatrix::Coefficients
    PressureMatrix::neighbourLinksWeights(const unsigned int i, const unsigned int j, const unsigned int k) const
{
    const std::array<unsigned int, 3> index = {i, j, k};
    Coefficients w;
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        const double used = (axis < m_decomposition.ndim) ? 1.0 : 0.0;
        w[axis]           = {used, used};
        if (index[axis] == 0 && m_decomposition.reflective[axis][SIDE_L]) { w[axis][SIDE_L] = 0.0; }
        if (index[axis] == m_decomposition.localCells[axis] - 1 && m_decomposition.reflective[axis][SIDE_R])
        {
            w[axis][SIDE_R] = 0.0;
        }
    }
    return w;
}

double PressureMatrix::coefSum(const Coefficients& c)
{
    // Make sure to sum along `side` axis first to reduce catastrophic cancellations between directions
    double sum = 0.0;
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        sum += c[axis][SIDE_L] + c[axis][SIDE_R];
    }
    return sum;
}

void PressureMatrix::setElement(const GlobalIndex column, const double value)
{
    if (column < 0) { return; }
    m_columnGlobalIndices.push_back(column);
    m_values.push_back(value);
}

void PressureMatrix::endRow()
{ m_rowOffsets.push_back(m_values.size()); }

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
    const std::array<unsigned int, 3>& nc   = m_decomposition.localCells;
    const std::array<GlobalIndex, 3>& start = m_decomposition.globalStart;
    const unsigned int kg                   = (ndim == 3) ? 1 : 0;
    const unsigned int n_pressure           = 0;
    const unsigned int n_energy             = 1;
    const size_t numRows                    = size_t(nc[0]) * nc[1] * nc[2] * m_prec_nvar;
    const size_t maxEntriesPerRow           = (2 * ndim + 1) * m_prec_nvar;

    m_rowGlobalIndices.clear();
    m_rowOffsets.clear();
    m_columnGlobalIndices.clear();
    m_values.clear();
    m_rowGlobalIndices.reserve(numRows);
    m_rowOffsets.reserve(numRows + 1);
    m_columnGlobalIndices.reserve(numRows * maxEntriesPerRow);
    m_values.reserve(numRows * maxEntriesPerRow);
    m_rowOffsets.push_back(0);

    for (unsigned int k = 0; k < nc[2]; k++)
    {
        for (unsigned int j = 0; j < nc[1]; j++)
        {
            for (unsigned int i = 0; i < nc[0]; i++)
            {
                // (pressure, pressure) matrix element
                Coefficients coef          = pressureCoefficients(solution, laplacian, i, j, k);
                const Coefficients weights = neighbourLinksWeights(i, j, k);
                for (unsigned int axis = 0; axis < 3; axis++)
                {
                    for (unsigned int side = SIDE_L; side <= SIDE_R; side++)
                    {
                        coef[axis][side] = coef[axis][side] * weights[axis][side];
                    }
                }
                const double J_PP = -dt * squared_sound_speed(i, j, k);

                const GlobalIndex rowP = globalDofIndex(start[0] + i, start[1] + j, start[2] + k, n_pressure);
                m_rowGlobalIndices.push_back(rowP);
                setElement(rowP, 1.0 / dt - J_PP * coefSum(coef));
                for (unsigned int axis = 0; axis < ndim; axis++)
                {
                    setElement(neighbourDofIndex(i, j, k, n_pressure, axis, SIDE_L), J_PP * coef[axis][SIDE_L]);
                    setElement(neighbourDofIndex(i, j, k, n_pressure, axis, SIDE_R), J_PP * coef[axis][SIDE_R]);
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
                auto dTdeChi                = [&](const int di, const int dj, const int dk)
                { return ghosted(dTde, i, j, k, di, dj, dk, kg) * ghosted(chi, i, j, k, di, dj, dk, kg); };

                Coefficients e_coef = {};
                e_coef[0][SIDE_L]   = coef[0][SIDE_L] * 0.5 * (dTdeChi(0, 0, 0) + dTdeChi(-1, 0, 0));
                e_coef[0][SIDE_R]   = coef[0][SIDE_R] * 0.5 * (dTdeChi(0, 0, 0) + dTdeChi(1, 0, 0));
                e_coef[1][SIDE_L]   = coef[1][SIDE_L] * 0.5 * (dTdeChi(0, 0, 0) + dTdeChi(0, -1, 0));
                e_coef[1][SIDE_R]   = coef[1][SIDE_R] * 0.5 * (dTdeChi(0, 0, 0) + dTdeChi(0, 1, 0));
                if (ndim == 3)
                {
                    e_coef[2][SIDE_L] = coef[2][SIDE_L] * 0.5 * (dTdeChi(0, 0, 0) + dTdeChi(0, 0, -1));
                    e_coef[2][SIDE_R] = coef[2][SIDE_R] * 0.5 * (dTdeChi(0, 0, 0) + dTdeChi(0, 0, 1));
                }

                const double rho  = ghosted(*solution.rho, i, j, k, 0, 0, 0, kg);
                const double J_Pe = -ghosted(*solution.dPde, i, j, k, 0, 0, 0, kg) / rho;  // equal to -(Gamma3 - 1)
                const double J_eP = -dt * ghosted(*solution.pressure, i, j, k, 0, 0, 0, kg) / (rho * rho);
                const double J_ee = -1.0 / rho;

                // (pressure, energy)
                setElement(globalDofIndex(start[0] + i, start[1] + j, start[2] + k, n_energy), -J_Pe * coefSum(e_coef));
                for (unsigned int axis = 0; axis < ndim; axis++)
                {
                    setElement(neighbourDofIndex(i, j, k, n_energy, axis, SIDE_L), J_Pe * e_coef[axis][SIDE_L]);
                    setElement(neighbourDofIndex(i, j, k, n_energy, axis, SIDE_R), J_Pe * e_coef[axis][SIDE_R]);
                }
                endRow();

                // (energy, :) elements
                const GlobalIndex rowE = globalDofIndex(start[0] + i, start[1] + j, start[2] + k, n_energy);
                m_rowGlobalIndices.push_back(rowE);

                setElement(rowP, -J_eP * coefSum(coef));
                for (unsigned int axis = 0; axis < ndim; axis++)
                {
                    setElement(neighbourDofIndex(i, j, k, n_pressure, axis, SIDE_L), J_eP * coef[axis][SIDE_L]);
                    setElement(neighbourDofIndex(i, j, k, n_pressure, axis, SIDE_R), J_eP * coef[axis][SIDE_R]);
                }

                setElement(rowE, 1.0 / dt - J_ee * coefSum(e_coef));
                for (unsigned int axis = 0; axis < ndim; axis++)
                {
                    setElement(neighbourDofIndex(i, j, k, n_energy, axis, SIDE_L), J_ee * e_coef[axis][SIDE_L]);
                    setElement(neighbourDofIndex(i, j, k, n_energy, axis, SIDE_R), J_ee * e_coef[axis][SIDE_R]);
                }
                endRow();
            }
        }
    }
}
