/**
 * @class GhostExchangePlan
 *
 * @brief Defines the plan for exchanging ghost cells between MPI ranks.
 */

#ifndef GhostExchangePlan_H
#define GhostExchangePlan_H

#include "DirectionalArray.h"
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
     * @param[in] boundary_conditions - The boundary conditions for this ranks block of cells.
     * @param[in] neighbours - The neighbouring MPI ranks for the current process.
     * @param[in] communicator - The MPI communicator for this plan.
     */
    GhostExchangePlan(const DirectionalArray<BoundaryConditions>& boundary_conditions,
                      const DirectionalArray<int>& neighbours,
                      const MPI_Comm& communicator) :
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

  private:
    const DirectionalArray<BoundaryConditions> m_boundary_conditions;
    const DirectionalArray<int> m_neighbours;
    const MPI_Comm m_communicator;
    int m_my_rank;
};

#endif  // GhostExchangePlan_H
