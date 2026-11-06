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
     * @param[in] boundary_conditions - The boundary conditions for this ranks block of cells.
     * @param[in] neighbours - The neighbouring MPI ranks for the current process.
     * @param[in] communicator - The MPI communicator for this plan.
     */
    GhostExchangePlan(const Slab& global_domain,
                      const DirectionalArray<BoundaryConditions>& boundary_conditions,
                      const DirectionalArray<int>& neighbours,
                      const MPI_Comm& communicator) :
        m_global_domain(global_domain),
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
    }

    /**
     * @brief Get the global indices of a neighbouring cell in a given direction along an axis.
     *
     * @param[in] i - The cell index in the first dimension.
     * @param[in] j - The cell index in the second dimension.
     * @param[in] k - The cell index in the third dimension.
     * @param[in] axis - The axis along which to look for neighbours.
     * @param[in] direction - The direction in which along the axis to look for neighbours.
     */
    std::array<int, 3>
        neighbourCellPosition(const int& i, const int& j, const int& k, const unsigned int& axis, const Direction& direction)
    {
        const Slab& global_domain   = globalDomain();
        std::array<int, 3> position = {global_domain.start(0) + i, global_domain.start(1) + j, global_domain.start(2) + k};
        position[axis]              = direction == Direction::NEGATIVE ? position[axis] - 1 : position[axis] + 1;
        int n                       = global_domain.end(axis);
        if (boundaryConditions().get(axis, direction) == BoundaryConditions::PERIODIC)
            position[axis] = ((position[axis] % n) + n) % n;
        else if (position[axis] < global_domain.start(axis) || position[axis] >= n)
            position[axis] = -1;
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

  private:
    const Slab m_global_domain;
    const DirectionalArray<BoundaryConditions> m_boundary_conditions;
    const DirectionalArray<int> m_neighbours;
    const MPI_Comm m_communicator;
    int m_my_rank;
};

#endif  // GhostExchangePlan_H
