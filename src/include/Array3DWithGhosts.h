/**
 * @class Array3DWithGhosts
 *
 * @brief Provides a 3D view on a 1D array with custom bounds and ghost cells for MPI domain decomposition
 */

#ifndef Array3DWithGhosts_H
#define Array3DWithGhosts_H

#include "Array2D.h"
#include "Array3D.h"
#include <array>
#include <stdexcept>

enum Side
{
    LEFT,
    RIGHT
};

template<typename T>
class Array3DWithGhosts : public Array3D<T>
{
    friend class Array3DWithGhostsTest;

  public:
    /**
     * @brief Constructs an Array3DWithGhosts from a parent Array3D.
     *
     * @details Pre-allocates ghost cell arrays bases on the given bounds.
     *
     * @param ni - The number of elements to be extracted in the first direction.
     * @param nj - The number of elements to be extracted in the second direction.
     * @param nk - The number of elements to be extracted in the third direction.
     * @param parent - The parent Array3D without ghosts.
     */
    Array3DWithGhosts(const int& ni, const int& nj, const int& nk, Array3D<T>& parent) :
        Array3D<T>(parent),
        m_ghosts({Array2D<T>(nj, nk),
                  Array2D<T>(nj, nk),
                  Array2D<T>(ni, nk),
                  Array2D<T>(ni, nk),
                  Array2D<T>(ni, nj),
                  Array2D<T>(ni, nj)}),
        m_edges({Array2D<T>(nj, nk),
                 Array2D<T>(nj, nk),
                 Array2D<T>(ni, nk),
                 Array2D<T>(ni, nk),
                 Array2D<T>(ni, nj),
                 Array2D<T>(ni, nj)})
    {
    }

    /**
     * @brief Update the contiguous edge planes to be sent via MPI.
     *
     * @details Values from the non-contiguous edges stored in the bulk inner
     *          array are copied to the contiguous edge arrays ready for
     *          sending.
     */
    void updateEdges()
    {
        int i0 = this->m_slab.start(0), j0 = this->m_slab.start(1), k0 = this->m_slab.start(2);
        int in = this->m_slab.end(0), jn = this->m_slab.end(1), kn = this->m_slab.end(2);
        // Populate the contiguous edge planes within m_edges
        // axis i
        for (int k = k0; k < kn; k++)
        {
            for (int j = j0; j < jn; j++)
            {
                unsigned int jk_idx            = (j * this->m_stride_j) + (k * this->m_stride_k);
                m_edges[0 + Side::LEFT](j, k)  = this->m_data_origin[jk_idx + (i0 * this->m_stride_i)];
                m_edges[0 + Side::RIGHT](j, k) = this->m_data_origin[jk_idx + ((in - 1) * this->m_stride_i)];
            }
        }
        // axis j
        for (int k = k0; k < kn; k++)
        {
            for (int i = i0; i < in; i++)
            {
                unsigned int ik_idx            = (i * this->m_stride_i) + (k * this->m_stride_k);
                m_edges[2 + Side::LEFT](i, k)  = this->m_data_origin[ik_idx + (j0 * this->m_stride_j)];
                m_edges[2 + Side::RIGHT](i, k) = this->m_data_origin[ik_idx + ((jn - 1) * this->m_stride_j)];
            }
        }
        // axis k
        for (int j = j0; j < jn; j++)
        {
            for (int i = i0; i < in; i++)
            {
                unsigned ij_idx                = (i * this->m_stride_i) + (j * this->m_stride_j);
                m_edges[4 + Side::LEFT](i, j)  = this->m_data_origin[ij_idx + (k0 * this->m_stride_k)];
                m_edges[4 + Side::RIGHT](i, j) = this->m_data_origin[ij_idx + ((kn - 1) * this->m_stride_k)];
            }
        }
    }

    /**
     * @brief access operator to return a reference to the given element.
     *
     * @details The values for indices overlapping within ghost are returned
     *          from the respective contiguous slab arrays, rather than the
     *          bulk array.
     *
     * @param i - The index in the first dimension.
     * @param j - The index in the second dimension.
     * @param k - The index in the third dimension.
     *
     * @return Reference to the element at the requested 3D coordinates
     */
    T& operator ()(int i, int j, int k) const override
    {
        std::array<int, 3> indices = {i, j, k};
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            Side side = Side::RIGHT;
            if (indices[axis] == this->m_slab.start(axis) - 1)
                side = Side::LEFT;
            else if (indices[axis] == this->m_slab.end(axis))
                side = Side::RIGHT;
            else
                continue;

            unsigned int axis_side_idx = axis * 2;
            switch (axis)
            {
                case 0: return m_ghosts[axis_side_idx + side](j, k);
                case 1: return m_ghosts[axis_side_idx + side](i, k);
                case 2: return m_ghosts[axis_side_idx + side](i, j);
            }
        }

        const unsigned int offset = (i * this->m_stride_i) + (j * this->m_stride_j) + (k * this->m_stride_k);
        return this->m_data_origin[offset];
    }

    /**
     * @brief Get the contiguous edge plane for a given axis and side.
     *
     * @param[in] axis - The axis who's edge should be returned {i, j, k} => {0, 1, 2}
     * @param[in] side - The side of the requested axis to be returned.
     *
     * @returns A copy of the requested edge plane.
     */
    Array2D<T> edge(const unsigned int& axis, Side side)
    { return m_edges[axis * 2 + side]; }

  private:
    /**
     * @brief Set the ghost cells for a given axis and side.
     *
     * @param[in] axis - The axis that should be set (i, j, k) => (0, 1, 2).
     * @param[in] new_ghost_values - The new ghost cell values.
     */
    void setGhostCells(unsigned int const axis, Side side, Array2D<T> new_ghost_values)
    { new_ghost_values.deepCopy(m_ghosts[axis * 2 + side]); }

    void setEdgeCells(unsigned int const axis, Side side, Array2D<T> new_edge_values)
    { new_edge_values.deepCopy(m_edges[axis * 2 + side]); }

    /// @brief The ghost cells and cells on the edge of the domain owned by this array.
    std::array<Array2D<T>, 6> m_ghosts;
    /// @brief The cells on the edge of the domain owned by this array.
    std::array<Array2D<T>, 6> m_edges;
};

#endif  // Array3DWithGhosts_H
