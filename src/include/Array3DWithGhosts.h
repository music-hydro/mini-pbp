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
#include "Slab.h"
#include <array>
#include <iostream>
#include <mpi.h>

enum Direction
{
    NEGATIVE,
    POSITIVE
};

template<typename T>
class Array3DWithGhosts : public Array3D<T>
{
    friend class Array3DWithGhostsTest;

  public:
    /**
     * @brief Constructs an Array3DWithGhosts from a GhostExchangePlan.
     *
     * @details Pre-allocates ghost cell arrays bases on the bounds set within
     *          the plan.
     *
     * @param plan - The plan defining how ghost cells should be exchanged.
     */
    Array3DWithGhosts(GhostExchangePlan<T>& plan) :
        Array3D<T>(plan.array()),
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

            // unsigned int axis_side_idx = get_plane_index(axis, direction);
            size_t idx = planeIndex(axis, direction);
            switch (axis)
            {
                case 0: return m_ghosts[idx](j, k);
                case 1: return m_ghosts[idx](i, k);
                case 2: return m_ghosts[idx](i, j);
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
    Array2D<T> ghost(const unsigned int& axis, Direction direction)
    { return m_ghosts[axis * 2U + direction]; }

    /**
     * @brief Get the contiguous boundary plane for a given axis and direction.
     *
     * @param[in] axis - The axis who's boundary should be returned {i, j, k} => {0, 1, 2}
     * @param[in] direction - The direction of the requested axis to be returned.
     *
     * @returns A copy of the requested boundary plane.
     */
    Array2D<T> boundary(const unsigned int& axis, Direction direction)
    { return m_boundaries[axis * 2U + direction]; }

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
            std::vector<MPI_Request> requests;
            size_t negative_plane_idx          = planeIndex(axis, Direction::NEGATIVE);
            size_t positive_plane_idx          = planeIndex(axis, Direction::POSITIVE);
            Array2D<T> negative_boundary_plane = m_boundaries[negative_plane_idx];
            Array2D<T> positive_boundary_plane = m_boundaries[positive_plane_idx];
            Array2D<T> negative_ghost_plane    = m_ghosts[negative_plane_idx];
            Array2D<T> positive_ghost_plane    = m_ghosts[positive_plane_idx];

            // Get the neighbours
            int negative_neighbour_rank = m_plan.neighbours()[negative_plane_idx];
            int positive_neighbour_rank = m_plan.neighbours()[positive_plane_idx];

            // Set the negative direction boundary ghosts
            if (positive_neighbour_rank != MPI_PROC_NULL)
            {
                MPI_Request send_request;
                MPI_Request recv_request;
                int buffer_size = positive_boundary_plane.cellSlab().volume();
                MPI_Isend(positive_boundary_plane.start(),
                          buffer_size,
                          MpiType<T>::value,
                          positive_neighbour_rank,
                          Direction::POSITIVE,
                          communicator,
                          &send_request);
                MPI_Irecv(positive_ghost_plane.start(),
                          buffer_size,
                          MpiType<T>::value,
                          positive_neighbour_rank,
                          Direction::NEGATIVE,
                          communicator,
                          &recv_request);
                requests.push_back(send_request);
                requests.push_back(recv_request);
            }
            else
                applyBoundaryConditions(axis, Direction::POSITIVE);
            if (negative_neighbour_rank != MPI_PROC_NULL)
            {
                int buffer_size = negative_boundary_plane.cellSlab().volume();
                MPI_Request send_request;
                MPI_Request recv_request;
                MPI_Isend(negative_boundary_plane.start(),
                          buffer_size,
                          MpiType<T>::value,
                          negative_neighbour_rank,
                          Direction::NEGATIVE,
                          communicator,
                          &send_request);
                MPI_Irecv(negative_ghost_plane.start(),
                          buffer_size,
                          MpiType<T>::value,
                          negative_neighbour_rank,
                          Direction::POSITIVE,
                          communicator,
                          &recv_request);
                requests.push_back(send_request);
                requests.push_back(recv_request);
            }
            else
                applyBoundaryConditions(axis, Direction::NEGATIVE);

            for (int ridx = 0; ridx < requests.size(); ridx++)
            {
                std::cout << "Rank " << my_rank << ", for axis " << axis << ", waiting for request " << ridx << std::endl;
                int result = MPI_Wait(&requests[ridx], MPI_STATUS_IGNORE);
                if (result != MPI_SUCCESS)
                {
                    throw std::runtime_error("Ghost exchange failed for rank " + std::to_string(my_rank));
                    MPI_Finalize();
                }
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
        size_t negative_plane_idx = planeIndex(0U, Direction::NEGATIVE);
        size_t positive_plane_idx = planeIndex(0U, Direction::POSITIVE);
        for (int k = k0; k < kn; k++)
        {
            for (int j = j0; j < jn; j++)
            {
                unsigned int jk_idx                    = (j * this->m_stride_j) + (k * this->m_stride_k);
                m_boundaries[negative_plane_idx](j, k) = this->m_data_origin[jk_idx + (i0 * this->m_stride_i)];
                m_boundaries[positive_plane_idx](j, k) = this->m_data_origin[jk_idx + ((in - 1) * this->m_stride_i)];
            }
        }
        // axis j
        negative_plane_idx = planeIndex(1U, Direction::NEGATIVE);
        positive_plane_idx = planeIndex(1U, Direction::POSITIVE);
        for (int k = k0; k < kn; k++)
        {
            for (int i = i0; i < in; i++)
            {
                unsigned int ik_idx                    = (i * this->m_stride_i) + (k * this->m_stride_k);
                m_boundaries[negative_plane_idx](i, k) = this->m_data_origin[ik_idx + (j0 * this->m_stride_j)];
                m_boundaries[positive_plane_idx](i, k) = this->m_data_origin[ik_idx + ((jn - 1) * this->m_stride_j)];
            }
        }
        // axis k
        negative_plane_idx = planeIndex(2U, Direction::NEGATIVE);
        positive_plane_idx = planeIndex(2U, Direction::POSITIVE);
        for (int j = j0; j < jn; j++)
        {
            for (int i = i0; i < in; i++)
            {
                unsigned ij_idx                        = (i * this->m_stride_i) + (j * this->m_stride_j);
                m_boundaries[negative_plane_idx](i, j) = this->m_data_origin[ij_idx + (k0 * this->m_stride_k)];
                m_boundaries[positive_plane_idx](i, j) = this->m_data_origin[ij_idx + ((kn - 1) * this->m_stride_k)];
            }
        }
    }

    /**
     * @brief Consistent definition of how to index m_boundaries and m_ghosts
     *
     * @param[in] axis - The axis whose plane should be indexed.
     * @param[in] direction - The direction of the axis to be index.
     *
     * @returns The plane index for accessing m_boundaries and m_ghosts.
     */
    size_t planeIndex(const unsigned int& axis, const Direction& direction) const
    { return static_cast<size_t>(axis * 2U + direction); }

    /**
     * @brief Apply the boundary conditions to a ghost plane.
     *
     * @param[in] axis - The axis whose ghost should be filled.
     * @param[in] direction - The direction on which the ghost plane should be filled.
     */
    void applyBoundaryConditions(unsigned int axis, Direction direction)
    {
        size_t plane_idx          = planeIndex(axis, direction);
        Array2D<T> ghost_plane    = m_ghosts[plane_idx];
        Array2D<T> boundary_plane = m_boundaries[plane_idx];
        BoundaryConditions bc     = m_plan.boundary_conditions()[plane_idx];

        T parity;
        if (bc == BoundaryConditions::ZERO_VALUE)
            parity = -1.0;
        else if (bc == BoundaryConditions::ZERO_DERIVATIVE)
            parity = 1.0;
        else
            throw std::runtime_error("Invalid boundary condition: " + std::to_string((int)bc));

        boundary_plane.deepCopy(ghost_plane, parity);
    }

    GhostExchangePlan<T> m_plan;
    unsigned int m_ni;
    unsigned int m_nj;
    unsigned int m_nk;
    /// @brief The ghost cells and cells on the boundary of the domain owned by this array.
    std::array<Array2D<T>, 6> m_ghosts;
    /// @brief The cells on the boundary of the domain owned by this array.
    std::array<Array2D<T>, 6> m_boundaries;
};

#endif  // Array3DWithGhosts_H
