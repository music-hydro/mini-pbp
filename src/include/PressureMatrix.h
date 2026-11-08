/**
 * @class PressureMatrix
 *
 * @brief Assembles, for the cells of one rank, the matrix of the preconditioner pressure system in CSR form (rows of
 * this rank, global column indices). It is a translation of `set_matrix` in MUSIC `src/solvers/preconditioner.f90`,
 * with `pressure_coeffs`, `coef_sum`, `neighbor_links_weights`, `pbp_dof_t%neighbor` and `global_dof_idx0`.
 *
 * Unknowns are numbered as in MUSIC: order (ivar, i, j, k) over the whole domain, ivar fastest, with the global cell
 * position (i, j, k) counted from the first cell of the domain. A neighbour across a periodic boundary is the cell at
 * the other end of the domain; a neighbour outside a non periodic boundary does not exist and gets no entry.
 *
 * For each cell, with coef the grid Laplacian coefficients times rho / rho_face times the neighbour link weight (0 on a
 * reflective boundary face, else 1):
 * - (P, P): diagonal 1/dt + dt c^2 sum(coef), off diagonal -dt c^2 coef for each neighbour.
 * - for prec_nvar = 2 also (P, e), (e, P) and (e, e), with e_coef = coef times the face average of dT/de chi:
 *   J_Pe = -dP/de / rho, J_eP = -dt P / rho^2, J_ee = -1 / rho.
 *
 * Per cell quantities on the six faces use DirectionalArray, laid out {-x, +x, -y, +y, -z, +z} and indexed by axis
 * (0 x, 1 y, 2 z) and Direction (NEGATIVE, POSITIVE).
 *
 * The rows come in the local order of the vectors (ivar, i, j, k), the entries of a row in the order MUSIC sets them,
 * so the arrays can be given to PressureSolver::setMatrix() as they are.
 */

#ifndef PressureMatrix_H
#define PressureMatrix_H

#include "Array3D.h"
#include "DirectionalArray.h"
#include <array>
#include <cstddef>
#include <vector>

class PressureMatrix
{
  public:
    using GlobalIndex = long long;

    /**
     * @brief Where the cells of this rank sit in the whole domain, and the boundaries they see.
     */
    struct Decomposition
    {
        unsigned int ndim;                        // Number of dimensions (2 or 3)
        std::array<unsigned int, 3> local_cells;  // Cells of this rank along each axis (1 along k in 2D)
        std::array<GlobalIndex, 3> global_start;  // Global position of the first cell of this rank, from 0
        std::array<GlobalIndex, 3> global_cells;  // Cells of the whole domain along each axis
        std::array<bool, 3> periodic;             // Whether the domain is periodic along each axis
        DirectionalArray<bool> reflective;        // Whether this rank has a reflective boundary on that face
    };

    /**
     * @brief The solution fields set_matrix reads, laid out like MUSIC `sol%vars`: one ghost layer on each side of the
     * cells of this rank (none along k in 2D), so cell (i, j, k) of the rank is element (i + 1, j + 1, k + 1) of the
     * view (k + 0 in 2D). Only rho is needed for prec_nvar = 1.
     */
    struct SolutionFields
    {
        const Array3D<double>* rho;       // Density
        const Array3D<double>* pressure;  // Pressure (prec_nvar = 2 only)
        const Array3D<double>* dPde;      // dP/de (prec_nvar = 2 only)
        const Array3D<double>* chi;       // Thermal conductivity chi (prec_nvar = 2 only)
        const Array3D<double>* dTde;      // dT/de (prec_nvar = 2 only)
    };

    /// Grid Laplacian coefficients (MUSIC `cell_centered_laplacian_coeffs`) of the six faces of each cell of the rank
    using LaplacianCoefficients = Array3D<DirectionalArray<double>>;

    /**
     * @brief Constructor
     *
     * @param[in] decomposition - The cells of this rank in the whole domain and their boundaries.
     * @param[in] prec_nvar - The number of variables to precondition (1: just pressure, 2: pressure and energy).
     */
    PressureMatrix(const Decomposition& decomposition, const unsigned int prec_nvar = 1);

    /**
     * @brief Compute the matrix rows of this rank, as MUSIC `set_matrix`.
     *
     * @param[in] dt - The time step.
     * @param[in] solution - The solution fields, with one ghost layer.
     * @param[in] squared_sound_speed - The squared sound speed per cell of the rank (no ghosts).
     * @param[in] laplacian - The grid Laplacian coefficients per cell of the rank (no ghosts).
     */
    void assemble(const double dt,
                  const SolutionFields& solution,
                  const Array3D<double>& squared_sound_speed,
                  const LaplacianCoefficients& laplacian);

    /**
     * @brief Global index of unknown `ivar` (0 pressure, 1 energy) at the cell at global position (i, j, k), as MUSIC
     * `global_dof_idx0`.
     */
    GlobalIndex globalDofIndex(const GlobalIndex i, const GlobalIndex j, const GlobalIndex k, const unsigned int ivar) const;

    /**
     * @brief Global indices of the rows of this rank, in the local order of the vectors.
     */
    const std::vector<GlobalIndex>& rowGlobalIndices() const
    { return m_row_global_indices; }

    /**
     * @brief CSR row offsets, of size number of local rows + 1.
     */
    const std::vector<size_t>& rowOffsets() const
    { return m_row_offsets; }

    /**
     * @brief Global column index of each entry.
     */
    const std::vector<GlobalIndex>& columnGlobalIndices() const
    { return m_column_global_indices; }

    /**
     * @brief Value of each entry.
     */
    const std::vector<double>& values() const
    { return m_values; }

  private:
    /**
     * @brief Grid Laplacian coefficients times rho / rho_face, as MUSIC `pressure_coeffs`.
     */
    DirectionalArray<double> pressureCoefficients(const SolutionFields& solution,
                                                  const LaplacianCoefficients& laplacian,
                                                  const unsigned int i,
                                                  const unsigned int j,
                                                  const unsigned int k) const;

    /**
     * @brief 1 for each neighbour link, 0 on a reflective boundary face and along unused axes, as MUSIC
     * `neighbor_links_weights`.
     */
    DirectionalArray<double> neighbourLinksWeights(const unsigned int i, const unsigned int j, const unsigned int k) const;

    /**
     * @brief Sum of the coefficients, adding the two directions of an axis first to reduce cancellations between axes,
     * as MUSIC `coef_sum`.
     */
    static double coefSum(const DirectionalArray<double>& coefficients);

    /**
     * @brief Global index of unknown `ivar` at the neighbour of local cell (i, j, k) along `axis` in `direction`, or -1
     * if the neighbour does not exist, as MUSIC `pbp_dof_t%neighbor`.
     */
    GlobalIndex neighbourDofIndex(const unsigned int i,
                                  const unsigned int j,
                                  const unsigned int k,
                                  const unsigned int ivar,
                                  const unsigned int axis,
                                  const Direction direction) const;

    /**
     * @brief Append an entry to the current row, skipped when the column does not exist, as MUSIC `set_element`.
     */
    void setElement(const GlobalIndex column, const double value);

    /**
     * @brief Close the current row.
     */
    void endRow();

    const Decomposition m_decomposition;
    const unsigned int m_prec_nvar;
    std::vector<GlobalIndex> m_row_global_indices;
    std::vector<size_t> m_row_offsets;
    std::vector<GlobalIndex> m_column_global_indices;
    std::vector<double> m_values;
};

#endif  // PressureMatrix_H
