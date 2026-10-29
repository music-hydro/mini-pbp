/**
 * @class Array3DWithGhosts
 *
 * @brief Provides a 3D view on a 1D array with custom bounds and ghost cells for MPI domain decomposition
 */

#ifndef Array3DWithGhosts_H
#define Array3DWithGhosts_H

#include "Array3D.h"
#include <array>
#include <memory>

enum GhostPlane
{
    LEFT_GHOST,
    LEFT_SLAB,
    RIGHT_GHOST,
    RIGHT_SLAB
};

template<typename T>
class Array3DWithGhosts : public Array3D<T>
{
  public:
    /**
     * @brief Constructs an Array3DWithGhosts from a pointer to a 1D array
     *
     * @details Pre-allocates ghost cell arrays bases on the given bounds.
     *
     * @param data_1D - Pointer to the input 1D array.
     * @param ni - The number of elements to be extracted in the first direction.
     * @param nj - The number of elements to be extracted in the second direction.
     * @param nk - The number of elements to be extracted in the third direction.
     */
    Array3DWithGhosts(const unsigned int ni, const unsigned int nj, const unsigned int nk, T* const& data_1D) :
        Array3D<T>(ni, nj, nk, data_1D)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            switch (axis)
            {
                case 0: m_ghosts[axis] = std::make_unique<Array3D<T>>(nj, nk, 4);
                case 1: m_ghosts[axis] = std::make_unique<Array3D<T>>(ni, nk, 4);
                case 2: m_ghosts[axis] = std::make_unique<Array3D<T>>(ni, nj, 4);
            }
            m_ghosts[axis]->fill(0);
        }
    }

    // See parent class for docstring
    T& operator ()(unsigned int i, unsigned int j, unsigned int k) const override
    {
        Slab slab = cellSlab();

        std::array<unsigned int, 3> indices = {i, j, k};

        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            GhostPlane plane;
            if (indices[axis] == slab.start(axis))
                plane = LEFT_GHOST;
            else if (indices[axis] == slab.start(axis) + 1)
                plane = LEFT_SLAB;
            else if (indices[axis] >= slab.end(axis))
                plane = RIGHT_GHOST;
            else if (indices[axis] >= slab.end(axis) - 1)
                plane = RIGHT_SLAB;
            else
                continue;

            switch (axis)
            {
                case 0: return (*m_ghosts[0])(j, k, plane);
                case 1: return (*m_ghosts[1])(i, k, plane);
                case 2: return (*m_ghosts[2])(i, j, plane);
            }
        }

        const unsigned int offset = (i * this->m_stride_i) + (j * this->m_stride_j) + (k * this->m_stride_k);
        return this->m_data_origin[offset];
    }

    // See parent class for docstring
    Slab cellSlab() const override
    {
        Slab slab({-1, -1, -1}, {(int)(this->m_ni + 1), (int)(this->m_nj + 1), (int)(this->m_nk + 1)});
        return slab;
    }

    /**
     * @brief Set the ghost cells for a given axis and Side.
     *
     * @param[in] axis - The axis that should be set (i, j, k) => (0, 1, 2).
     * @param[in] new_ghost_values - The new ghost cell values.
     */
    void setGhostCells(unsigned int const axis, Array3D<T> new_ghost_values)
    { new_ghost_values.deepCopy((*m_ghosts)[axis]); }

  private:
    /** @brief The ghost cells, including the cells on the edge of the domain owned by this array  */
    std::array<std::unique_ptr<Array3D<T>>, 3> m_ghosts;
};

#endif  // Array3DWithGhosts_H
