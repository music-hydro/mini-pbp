/**
 * @class Array3DWithGhosts
 *
 * @brief Provides a 3D view on a 1D array with custom bounds and ghost cells for MPI domain decomposition
 */

#ifndef Array3DWithGhosts_H
#define Array3DWithGhosts_H

#include "Array2D.h"
#include "Array3D.h"
#include "GhostExchangePlan.h"
#include "MpiType.h"
#include "DirectionalArray.h"
#include "Slab.h"
#include <array>
#include <mpi.h>
#include <stdexcept>

template<typename T>
class Array3DWithGhosts : public Array3D<T>
{
    friend class Array3DWithGhostsTest;

  public:
    /**
     * @brief Constructs an Array3DWithGhosts from an Array3D and a GhostExchangePlan.
     *
     * @details Pre-allocates ghost cell arrays bases on the bounds set within
     *          the plan.
     *
     * @param[in] inner_array - The 3D view of the array without ghost cells.
     * @param[in] plan - The plan defining how ghost cells should be exchanged.
     */
    Array3DWithGhosts(const Array3D<T>& inner_array, GhostExchangePlan<T>& plan) :
        Array3D<T>(inner_array),
        m_plan(plan),
        m_ni(this->m_slab.extent(0)),
        m_nj(this->m_slab.extent(1)),
        m_nk(this->m_slab.extent(2)),
        m_ghosts({Array2D<T>(m_nj, m_nk),
                  Array2D<T>(m_nj, m_nk),
                  Array2D<T>(m_ni, m_nk),
                  Array2D<T>(m_ni, m_nk),
                  Array2D<T>(m_ni, m_nj),
                  Array2D<T>(m_ni, m_nj)}),
        m_boundaries({Array2D<T>(m_nj, m_nk),
                      Array2D<T>(m_nj, m_nk),
                      Array2D<T>(m_ni, m_nk),
                      Array2D<T>(m_ni, m_nk),
                      Array2D<T>(m_ni, m_nj),
                      Array2D<T>(m_ni, m_nj)})
    {
        if (this->m_is_sub_array) throw std::invalid_argument("A sub array cannot be used to create an array with ghosts");
    }

    /**
     * @brief access operator to return a reference to the given element.
     *
     * @details The values for indices overlapping within ghost are returned
     *          from the respective contiguous slab arrays, rather than the
     *          bulk array.
     *
     *          Note that the axes are ordered in priority as i->j->k. For
     *          example, any cell within an ith axis ghost plane will be
     *          returned from that plane. If a cell is not in an ith axis ghost
     *          plane but is in a jth axis ghost plane it will be returned from
     *          sed plane. Finally, if a cell is neither in an ith or jth axis
     *          ghost plane, but is in a kth axis ghost plane, only then will
     *          it be returned from that kth axis ghost plane.
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
            Direction direction = Direction::POSITIVE;
            if (indices[axis] == this->m_slab.start(axis) - 1)
                direction = Direction::NEGATIVE;
            else if (indices[axis] == this->m_slab.end(axis))
                direction = Direction::POSITIVE;
            else
                continue;

            const Array2D<T>& current_ghost = ghost(axis, direction);
            switch (axis)
            {
                case 0: return current_ghost(j, k);
                case 1: return current_ghost(i, k);
                case 2: return current_ghost(i, j);
            }
        }

        const unsigned int offset = (i * this->m_stride_i) + (j * this->m_stride_j) + (k * this->m_stride_k);
        return this->m_data_origin[offset];
    }

    /**
     * @brief Get the contiguous ghost plane for a given axis and direction.
     *
     * @param[in] axis - The axis who's ghost should be returned {i, j, k} => {0, 1, 2}
     * @param[in] direction - The direction of the requested axis to be returned.
     *
     * @returns A copy of the requested ghost plane.
     */
    const Array2D<T>& ghost(const unsigned int& axis, const Direction& direction) const
    { return m_ghosts.get(axis, direction); }

    /**
     * @brief Get the contiguous boundary plane for a given axis and direction.
     *
     * @param[in] axis - The axis who's boundary should be returned {i, j, k} => {0, 1, 2}
     * @param[in] direction - The direction of the requested axis to be returned.
     *
     * @returns A copy of the requested boundary plane.
     */
    const Array2D<T>& boundary(const unsigned int& axis, const Direction& direction) const
    { return m_boundaries.get(axis, direction); }

    /**
     * @brief Exchange ghost cells with neighbour ranks according to the plan.
     */
    void exchangeGhosts()
    {
        int my_rank;
        MPI_Comm communicator = m_plan.communicator();
        MPI_Comm_rank(communicator, &my_rank);
        // Fill contiguous boundary slabs
        updateBoundaries();
        // Exchange ghosts
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            // Store requests in a vector to be waited for later
            std::vector<MPI_Request> requests;

            // Set the negative direction boundary ghosts
            int positive_neighbour_rank = m_plan.neighbours().get(axis, Direction::POSITIVE);
            if (positive_neighbour_rank != MPI_PROC_NULL)
            {
                MPI_Request send_request;
                const Array2D<T>& positive_boundary_plane = boundary(axis, Direction::POSITIVE);
                int buffer_size                           = positive_boundary_plane.cellSlab().volume();
                MPI_Isend(positive_boundary_plane.start(),
                          buffer_size,
                          MpiType<T>::value,
                          positive_neighbour_rank,
                          Direction::POSITIVE,
                          communicator,
                          &send_request);
                requests.push_back(send_request);

                MPI_Request recv_request;
                const Array2D<T>& positive_ghost_plane = ghost(axis, Direction::POSITIVE);
                MPI_Irecv(positive_ghost_plane.start(),
                          buffer_size,
                          MpiType<T>::value,
                          positive_neighbour_rank,
                          Direction::NEGATIVE,
                          communicator,
                          &recv_request);
                requests.push_back(recv_request);
            }
            else
                applyBoundaryConditions(axis, Direction::POSITIVE);

            int negative_neighbour_rank = m_plan.neighbours().get(axis, Direction::NEGATIVE);
            if (negative_neighbour_rank != MPI_PROC_NULL)
            {
                MPI_Request send_request;
                const Array2D<T>& negative_boundary_plane = boundary(axis, Direction::NEGATIVE);
                int buffer_size                           = negative_boundary_plane.cellSlab().volume();
                MPI_Isend(negative_boundary_plane.start(),
                          buffer_size,
                          MpiType<T>::value,
                          negative_neighbour_rank,
                          Direction::NEGATIVE,
                          communicator,
                          &send_request);
                requests.push_back(send_request);

                MPI_Request recv_request;
                const Array2D<T>& negative_ghost_plane = ghost(axis, Direction::NEGATIVE);
                MPI_Irecv(negative_ghost_plane.start(),
                          buffer_size,
                          MpiType<T>::value,
                          negative_neighbour_rank,
                          Direction::POSITIVE,
                          communicator,
                          &recv_request);
                requests.push_back(recv_request);
            }
            else
                applyBoundaryConditions(axis, Direction::NEGATIVE);

            for (int ridx = 0; ridx < requests.size(); ridx++)
            {
                int result = MPI_Wait(&requests[ridx], MPI_STATUS_IGNORE);
                if (result != MPI_SUCCESS) throw std::runtime_error("Ghost exchange failed for rank " + std::to_string(my_rank));
            }
        }
    }

  private:
    /**
     * @brief Update the contiguous boundary planes to be sent via MPI.
     *
     * @details Values from the non-contiguous boundaries stored in the bulk inner
     *          array are copied to the contiguous boundary arrays ready for
     *          sending.
     */
    void updateBoundaries()
    {
        int i0 = this->m_slab.start(0), j0 = this->m_slab.start(1), k0 = this->m_slab.start(2);
        int in = this->m_slab.end(0), jn = this->m_slab.end(1), kn = this->m_slab.end(2);
        // Populate the contiguous boundary planes within m_boundaries
        // axis i
        const Array2D<T>& i_negative_boundary = boundary(0U, Direction::NEGATIVE);
        const Array2D<T>& i_positive_boundary = boundary(0U, Direction::POSITIVE);
        for (int k = k0; k < kn; k++)
        {
            for (int j = j0; j < jn; j++)
            {
                unsigned int jk_idx       = (j * this->m_stride_j) + (k * this->m_stride_k);
                i_negative_boundary(j, k) = this->m_data_origin[jk_idx + (i0 * this->m_stride_i)];
                i_positive_boundary(j, k) = this->m_data_origin[jk_idx + ((in - 1) * this->m_stride_i)];
            }
        }
        // axis j
        const Array2D<T>& j_negative_boundary = boundary(1U, Direction::NEGATIVE);
        const Array2D<T>& j_positive_boundary = boundary(1U, Direction::POSITIVE);
        for (int k = k0; k < kn; k++)
        {
            for (int i = i0; i < in; i++)
            {
                unsigned int ik_idx       = (i * this->m_stride_i) + (k * this->m_stride_k);
                j_negative_boundary(i, k) = this->m_data_origin[ik_idx + (j0 * this->m_stride_j)];
                j_positive_boundary(i, k) = this->m_data_origin[ik_idx + ((jn - 1) * this->m_stride_j)];
            }
        }
        // axis k
        const Array2D<T>& k_negative_boundary = boundary(2U, Direction::NEGATIVE);
        const Array2D<T>& k_positive_boundary = boundary(2U, Direction::POSITIVE);
        for (int j = j0; j < jn; j++)
        {
            for (int i = i0; i < in; i++)
            {
                unsigned ij_idx           = (i * this->m_stride_i) + (j * this->m_stride_j);
                k_negative_boundary(i, j) = this->m_data_origin[ij_idx + (k0 * this->m_stride_k)];
                k_positive_boundary(i, j) = this->m_data_origin[ij_idx + ((kn - 1) * this->m_stride_k)];
            }
        }
    }

    /**
     * @brief Apply the boundary conditions to a ghost plane.
     *
     * @param[in] axis - The axis whose ghost should be filled.
     * @param[in] direction - The direction on which the ghost plane should be filled.
     */
    void applyBoundaryConditions(unsigned int axis, Direction direction)
    {
        Array2D<T> ghost_plane    = m_ghosts.get(axis, direction);
        Array2D<T> boundary_plane = m_boundaries.get(axis, direction);
        BoundaryConditions bc     = m_plan.boundaryConditions().get(axis, direction);

        T parity;
        if (bc == BoundaryConditions::ZERO_VALUE)
            parity = -1.0;
        else if (bc == BoundaryConditions::ZERO_DERIVATIVE)
            parity = 1.0;
        else
            throw std::runtime_error("Invalid boundary condition: " + std::to_string((int)bc));

        boundary_plane.deepCopy(ghost_plane, parity);
    }

    const GhostExchangePlan<T> m_plan;
    const unsigned int m_ni;
    const unsigned int m_nj;
    unsigned int m_nk;
    /// @brief The ghost cells and cells on the boundary of the domain owned by this array.
    DirectionalArray<Array2D<T>> m_ghosts;
    /// @brief The cells on the boundary of the domain owned by this array.
    DirectionalArray<Array2D<T>> m_boundaries;
};

#endif  // Array3DWithGhosts_H
