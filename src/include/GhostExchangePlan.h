/**
 * @class GhostExchangePlan
 *
 * @brief Defines the plan for exchanging ghost cells between MPI ranks.
 */

#ifndef GhostExchangePlan_H
#define GhostExchangePlan_H

#include "DirectionalArray.h"
#include "Slab.h"
#include <array>
#include <mpi.h>
#include <stdexcept>

enum BoundaryConditions
{
    ZERO_VALUE,
    ZERO_DERIVATIVE,
    PERIODIC,
    REFLECTIVE,
    INNER
};

template<typename T>
class GhostExchangePlan
{
    friend class Array3DWithGhostsTest;

  public:
    /**
     * @brief Copy constructor
     */
    GhostExchangePlan(const GhostExchangePlan<T>& parent) :
        m_global_domain(parent.globalDomain()),
        m_local_domain(parent.localDomain()),
        m_boundary_conditions(parent.boundaryConditions()),
        m_neighbours(parent.neighbours()),
        m_communicator(parent.communicator()),
        m_my_rank(-1)
    { MPI_Comm_rank(m_communicator, &m_my_rank); }

    /**
     * @brief Constructor
     *
     * @details `boundary_conditions` and `neighbours` should have the
     *          following structure:
     *              {-x, +x, -y, +y, -z, +z}
     *
     * @param[in] global_domain - The slab which defines the global domain (i.e. including all cells).
     * @param[in] local_domain - The cells of this rank, in the global indices of `global_domain`.
     * @param[in] boundary_conditions - The boundary conditions for this ranks block of cells.
     * @param[in] neighbours - The neighbouring MPI ranks for the current process.
     * @param[in] communicator - The MPI communicator for this plan.
     */
    GhostExchangePlan(const Slab& global_domain,
                      const Slab& local_domain,
                      const DirectionalArray<BoundaryConditions>& boundary_conditions,
                      const DirectionalArray<int>& neighbours,
                      const MPI_Comm& communicator) :
        m_global_domain(global_domain),
        m_local_domain(local_domain),
        m_boundary_conditions(boundary_conditions),
        m_neighbours(neighbours),
        m_communicator(communicator),
        m_my_rank(-1)
    {
        MPI_Comm_rank(m_communicator, &m_my_rank);

        // Validate BCs and neighbours. PERIODIC BC should result in neighbours at boundaries.
        for (unsigned int axis_direction = 0; axis_direction < 6; axis_direction++)
            if (m_neighbours[axis_direction] == MPI_PROC_NULL)
            {
                if (m_boundary_conditions[axis_direction] == BoundaryConditions::PERIODIC)
                    throw std::invalid_argument("No neighbour found at periodic boundary");

                if (m_boundary_conditions[axis_direction] == BoundaryConditions::INNER)
                    throw std::invalid_argument("No neighbour found at inner boundary");
            }

        for (unsigned int axis = 0; axis < 3; axis++)
            if (m_local_domain.start(axis) < m_global_domain.start(axis) || m_local_domain.end(axis) > m_global_domain.end(axis))
                throw std::invalid_argument("The local domain must lie inside the global domain");
    }

    /**
     * @brief Constructor for a rank which holds the whole global domain.
     *
     * @copydetails GhostExchangePlan(const Slab&, const Slab&, const DirectionalArray<BoundaryConditions>&, const DirectionalArray<int>&, const MPI_Comm&)
     */
    GhostExchangePlan(const Slab& global_domain,
                      const DirectionalArray<BoundaryConditions>& boundary_conditions,
                      const DirectionalArray<int>& neighbours,
                      const MPI_Comm& communicator) :
        GhostExchangePlan(global_domain, global_domain, boundary_conditions, neighbours, communicator)
    {
    }

    /**
     * @brief Get the global indices of a neighbouring cell in a given direction along an axis, as MUSIC
     *        `pbp_dof_t%neighbor`.
     *
     * @details The position is counted from the start of the global domain. A neighbour outside the global domain is
     *          the periodic image at the other end when the domain is periodic there: the face has a PERIODIC boundary
     *          condition, or it is an INNER face with a neighbour rank across the domain boundary (a periodic cartesian
     *          communicator). Otherwise there is no neighbour and the position along `axis` is -1.
     *
     * @param[in] i - The local cell index of this rank in the first dimension.
     * @param[in] j - The local cell index of this rank in the second dimension.
     * @param[in] k - The local cell index of this rank in the third dimension.
     * @param[in] axis - The axis along which to look for neighbours.
     * @param[in] direction - The direction in which along the axis to look for neighbours.
     */
    std::array<int, 3> neighbourCellPosition(const int& i,
                                             const int& j,
                                             const int& k,
                                             const unsigned int& axis,
                                             const Direction& direction) const
    {
        const std::array<int, 3> global_start = m_global_domain.start();
        std::array<int, 3> position           = {m_local_domain.start(0) - global_start[0] + i,
                                                 m_local_domain.start(1) - global_start[1] + j,
                                                 m_local_domain.start(2) - global_start[2] + k};
        position[axis]                        = direction == Direction::NEGATIVE ? position[axis] - 1 : position[axis] + 1;
        const int n                           = static_cast<int>(m_global_domain.extent(axis));
        if (position[axis] < 0 || position[axis] >= n)
        {
            const BoundaryConditions bc = m_boundary_conditions.get(axis, direction);
            const bool periodic         = bc == BoundaryConditions::PERIODIC ||
                                          (bc == BoundaryConditions::INNER && m_neighbours.get(axis, direction) != MPI_PROC_NULL);
            position[axis]              = periodic ? ((position[axis] % n) + n) % n : -1;
        }
        return position;
    }

    /**
     * @brief Get the global boundary conditions for this plan.
     *
     * @return An array of boundary conditions with the following structure:
     *              {-x, +x, -y, +y, -z, +z}
     */
    DirectionalArray<BoundaryConditions> boundaryConditions() const
    { return m_boundary_conditions; }

    /**
     * @brief Get the neighbours for this plan.
     *
     * @details `neighbours` should have the following structure:
     *              {-x, +x, -y, +y, -z, +z}
     */
    DirectionalArray<int> neighbours() const
    { return m_neighbours; }

    /**
     * @brief Get the MPI communicator for this plan.
     */
    MPI_Comm communicator() const
    { return m_communicator; }

    /**
     * @brief Get the slab which defines the global domain.
     */
    Slab globalDomain() const
    { return m_global_domain; }

    /**
     * @brief Get the slab of the cells of this rank, in the global indices of globalDomain().
     */
    Slab localDomain() const
    { return m_local_domain; }

  private:
    const Slab m_global_domain;
    const Slab m_local_domain;
    const DirectionalArray<BoundaryConditions> m_boundary_conditions;
    const DirectionalArray<int> m_neighbours;
    const MPI_Comm m_communicator;
    int m_my_rank;
};

#endif  // GhostExchangePlan_H
