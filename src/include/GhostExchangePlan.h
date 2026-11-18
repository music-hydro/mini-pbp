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

class GhostExchangePlan
{
    friend class Array3DWithGhostsTest;

  public:
    /**
     * @brief Copy constructor
     */
    GhostExchangePlan(const GhostExchangePlan& parent) :
        m_global_domain(parent.globalDomain()),
        m_local_domain(parent.localDomain()),
        m_global_boundary_conditions(parent.globalBoundaryConditions()),
        m_local_boundary_conditions(parent.localBoundaryConditions()),
        m_neighbours(parent.neighbours()),
        m_communicator_dimensions(parent.communicator_dimensions()),
        m_communicator(parent.communicator())
    {
    }

    /**
     * @brief Constructor
     *
     * @details `boundary_conditions` and `neighbours` should have the
     *          following structure:
     *              {-x, +x, -y, +y, -z, +z}
     *
     * @param[in] global_domain - The slab which defines the global domain (i.e. including all cells).
     * @param[in] global_boundary_conditions - The global boundary conditions.
     * @param[in] neighbours - The neighbouring MPI ranks for the current process.
     * @param[in] communicator - The MPI communicator for this plan.
     */
    GhostExchangePlan(const Slab& global_domain,
                      const DirectionalArray<BoundaryConditions>& global_boundary_conditions,
                      const DirectionalArray<int>& neighbours,
                      const std::array<int, 3> communicator_dimensions,
                      const MPI_Comm& communicator) :
        m_global_domain(global_domain),
        m_local_domain(determineLocalDomain(global_domain, communicator_dimensions, communicator)),
        m_global_boundary_conditions(global_boundary_conditions),
        m_local_boundary_conditions(
            determineLocalBCs(communicator, communicator_dimensions, global_boundary_conditions, neighbours)),
        m_neighbours(neighbours),
        m_communicator_dimensions(communicator_dimensions),
        m_communicator(communicator)
    {
    }

    static std::array<int, 3> determineCartesianCoordinates(const MPI_Comm& communicator)
    {
        int my_cart_rank;
        MPI_Comm_rank(communicator, &my_cart_rank);
        std::array<int, 3> coordinates;
        MPI_Cart_coords(communicator, my_cart_rank, 3, coordinates.data());
        return coordinates;
    }

    static Slab determineLocalDomain(const Slab& global_domain,
                                     const std::array<int, 3> communicator_dimensions,
                                     const MPI_Comm& communicator)
    {
        std::array<int, 3> local_start, local_end;
        std::array<int, 3> coordinates = determineCartesianCoordinates(communicator);
        for (unsigned int axis = 0; axis < 3; axis++)
        {
            int global_start  = global_domain.start(axis);
            int global_length = global_domain.extent(axis);
            // Start is global start + local length times coordinate
            unsigned int local_n = global_length / communicator_dimensions[axis];
            local_start[axis]    = global_start + local_n * coordinates[axis];

            // End is local_start + local length
            local_end[axis] = local_start[axis] + local_n;
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
     */
    static DirectionalArray<BoundaryConditions> determineLocalBCs(const MPI_Comm& communicator,
                                                                  const std::array<int, 3>& communicator_dimensions,
                                                                  const DirectionalArray<BoundaryConditions>& global_bcs,
                                                                  const DirectionalArray<int>& neighbours)
    {
        // Get the coordinates of this rank in the cartesian MPI topology
        std::array<int, 3> coordinates = determineCartesianCoordinates(communicator);

        DirectionalArray<BoundaryConditions> local_bcs;
        local_bcs.fill(BoundaryConditions::INNER);
        for (unsigned int axis = 0; axis < 3U; axis++)
        {
            // Do not link these two if statements as, in a single rank case, these -ve and +ve bounds can be at the same coordinate
            if (coordinates[axis] == 0) local_bcs.get(axis, Direction::NEGATIVE) = global_bcs.get(axis, Direction::NEGATIVE);
            if (coordinates[axis] == communicator_dimensions[axis] - 1)
                local_bcs.get(axis, Direction::POSITIVE) = global_bcs.get(axis, Direction::POSITIVE);
        }

        // Validate BCs and neighbours. PERIODIC BC should result in neighbours at boundaries.
        for (unsigned int axis_direction = 0; axis_direction < 6; axis_direction++)
        {
            if (neighbours[axis_direction] == MPI_PROC_NULL)
            {
                if (local_bcs[axis_direction] == BoundaryConditions::PERIODIC)
                    throw std::invalid_argument("No neighbour found at periodic boundary");

                if (local_bcs[axis_direction] == BoundaryConditions::INNER)
                    throw std::invalid_argument("No neighbour found at inner boundary");
            }
        }
        return local_bcs;
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
    std::array<int, 3> neighbourCellPosition(const int& i,
                                             const int& j,
                                             const int& k,
                                             const unsigned int& axis,
                                             const Direction& direction) const
    {
        const Slab& global_domain   = globalDomain();
        std::array<int, 3> position = {global_domain.start(0) + i, global_domain.start(1) + j, global_domain.start(2) + k};
        position[axis]              = direction == Direction::NEGATIVE ? position[axis] - 1 : position[axis] + 1;
        int n                       = global_domain.end(axis);
        if (globalBoundaryConditions().get(axis, direction) == BoundaryConditions::PERIODIC)
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
    const Slab m_global_domain;
    const Slab m_local_domain;
    const DirectionalArray<BoundaryConditions> m_global_boundary_conditions;
    const DirectionalArray<BoundaryConditions> m_local_boundary_conditions;
    const DirectionalArray<int> m_neighbours;
    const std::array<int, 3> m_communicator_dimensions;
    const MPI_Comm m_communicator;
};

#endif  // GhostExchangePlan_H
