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
    INNER
};

class GhostExchangePlan
{
    friend class Array3DWithGhostsTest;
    friend class MpiTest;

  public:
    /**
     * @brief Copy constructor
     */
    GhostExchangePlan(const GhostExchangePlan& parent) :
        m_communicator(parent.communicator()),
        m_communicator_dimensions(parent.communicator_dimensions()),
        m_neighbours(parent.neighbours()),
        m_global_domain(parent.globalDomain()),
        m_local_domain(parent.localDomain()),
        m_global_boundary_conditions(parent.globalBoundaryConditions()),
        m_local_boundary_conditions(parent.localBoundaryConditions())
    {
    }

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
    GhostExchangePlan(const Slab& global_domain,
                      const DirectionalArray<BoundaryConditions>& global_boundary_conditions,
                      const DirectionalArray<int>& neighbours,
                      const std::array<int, 3> communicator_dimensions,
                      const MPI_Comm& communicator) :
        m_communicator(communicator),
        m_communicator_dimensions(communicator_dimensions),
        m_neighbours(neighbours),
        m_global_domain(global_domain),
        m_local_domain(determineLocalDomain()),
        m_global_boundary_conditions(global_boundary_conditions),
        m_local_boundary_conditions(determineLocalBCs())
    {
    }

    /**
     * @brief Get the global boundary conditions for this plan.
     *
     * @return An array of boundary conditions with the following structure:
     *              {-x, +x, -y, +y, -z, +z}
     */
    DirectionalArray<BoundaryConditions> globalBoundaryConditions() const
    { return m_global_boundary_conditions; }

    /**
     * @brief Get the local boundary conditions for this plan.
     *
     * @return An array of boundary conditions with the following structure:
     *              {-x, +x, -y, +y, -z, +z}
     */
    DirectionalArray<BoundaryConditions> localBoundaryConditions() const
    { return m_local_boundary_conditions; }

    /**
     * @brief Get the neighbours for this plan.
     *
     * @details `neighbours` should have the following structure:
     *              {-x, +x, -y, +y, -z, +z}
     */
    DirectionalArray<int> neighbours() const
    { return m_neighbours; }

    /**
     * @brief Get the dimensions of the MPI communicator for this plan.
     */
    std::array<int, 3> communicator_dimensions() const
    { return m_communicator_dimensions; }

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
     * @brief Get the slab which defines the local domain (i.e. the section of the global domain owned by this rank).
     */
    Slab localDomain() const
    { return m_local_domain; }

  private:
    /// @brief The MPI communicator for this plan.
    const MPI_Comm m_communicator;
    /// @brief The cartesian dimensions for this plan.
    const std::array<int, 3> m_communicator_dimensions;
    /// @brief The IDs of the current ranks neighbours.
    const DirectionalArray<int> m_neighbours;
    /// @brief The global domain defining the cells for the entire problem.
    const Slab m_global_domain;
    /// @brief The local domain defining the cells for the current rank.
    const Slab m_local_domain;
    /// @brief The boundary conditions for the global domain for the entire problem.
    const DirectionalArray<BoundaryConditions> m_global_boundary_conditions;
    /// @brief The boundary conditions for the local domain for the current rank.
    const DirectionalArray<BoundaryConditions> m_local_boundary_conditions;

    /**
     * @brief Determine the cartesian coordinates of the current MPI rank.
     *
     * @returns The cartesian coordinates of this MPI rank.
     */
    std::array<int, 3> determineCartesianCoordinates()
    {
        int my_cart_rank;
        MPI_Comm_rank(m_communicator, &my_cart_rank);
        std::array<int, 3> coordinates;
        MPI_Cart_coords(m_communicator, my_cart_rank, 3, coordinates.data());
        return coordinates;
    }

    /**
     * @brief Determine the local domain for this plan.
     */
    Slab determineLocalDomain()
    {
        std::array<int, 3> local_start, local_end;
        std::array<int, 3> coordinates = determineCartesianCoordinates();
        for (unsigned int axis = 0; axis < 3; axis++)
        {
            int global_start  = m_global_domain.start(axis);
            int global_length = m_global_domain.extent(axis);
            // Start is global start + local length times coordinate
            unsigned int local_n = global_length / m_communicator_dimensions[axis];
            local_start[axis]    = global_start + local_n * coordinates[axis];

            // End is local_start + local length
            local_end[axis] = local_start[axis] + local_n;

            // Add remainder if rank is at a positive global boundary
            if (coordinates[axis] == m_communicator_dimensions[axis] - 1)
                local_end[axis] += global_length % m_communicator_dimensions[axis];
        }
        return Slab(local_start, local_end);
    }

    /**
     * @brief Determine this ranks local boundary conditions based on the global ones and this ranks neighbours.
     *
     * @details This is defined as a static function to ensure the correct state is used to determine the local boundary conditions.
     *
     * @param[in] communicator - The MPI communicator for the current problem.
     * @param[in] communicator_dimensions - The cartesian dimensions of the current MPI communicator topology.
     * @param[in] global_bcs - The boundary conditions for the entire problem.
     * @param[in] neighbours - The neighbouring ranks for this MPI rank.
     *
     * @returns The boundary conditions for the local domain for the current rank.
     */
    DirectionalArray<BoundaryConditions> determineLocalBCs()
    {
        // Get the coordinates of this rank in the cartesian MPI topology
        std::array<int, 3> coordinates = determineCartesianCoordinates();

        DirectionalArray<BoundaryConditions> local_bcs;
        local_bcs.fill(BoundaryConditions::INNER);
        for (unsigned int axis = 0; axis < 3U; axis++)
        {
            // Do not link these two if statements as, in a single rank case, these -ve and +ve bounds can be at the same coordinate
            if (coordinates[axis] == 0)
                local_bcs.get(axis, Direction::NEGATIVE) = m_global_boundary_conditions.get(axis, Direction::NEGATIVE);
            if (coordinates[axis] == m_communicator_dimensions[axis] - 1)
                local_bcs.get(axis, Direction::POSITIVE) = m_global_boundary_conditions.get(axis, Direction::POSITIVE);
        }

        // Validate BCs and neighbours. PERIODIC BC should result in neighbours at boundaries.
        for (unsigned int axis_direction = 0; axis_direction < 6; axis_direction++)
        {
            if (m_neighbours[axis_direction] == MPI_PROC_NULL)
            {
                if (local_bcs[axis_direction] == BoundaryConditions::PERIODIC)
                    throw std::invalid_argument("No neighbour found at periodic boundary");

                if (local_bcs[axis_direction] == BoundaryConditions::INNER)
                    throw std::invalid_argument("No neighbour found at inner boundary");
            }
        }
        return local_bcs;
    }
};

#endif  // GhostExchangePlan_H
