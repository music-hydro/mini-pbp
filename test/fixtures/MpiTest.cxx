/**
 * @class MpiTest
 *
 * @brief Static methods for constructing the MPI domain for this problem.
 *
 * @details A 3D cartesian topology is used.
 */

#ifndef MpiTest_H
#define MpiTest_H

#include "GhostExchangePlan.h"
#include <array>
#include <mpi.h>
#include <stdexcept>

enum Axis
{
    X,
    Y,
    Z
};

class MpiTest
{
  public:
    /**
     * @brief Default constructor
     *
     * @details Creates a cartesian communicator and stores the coordinates of
     *          this rank and the ranks of its neighbours.
     */
    MpiTest() :
        m_global_bcs({BoundaryConditions::ZERO_VALUE,
                      BoundaryConditions::ZERO_VALUE,
                      BoundaryConditions::PERIODIC,
                      BoundaryConditions::PERIODIC,
                      BoundaryConditions::ZERO_DERIVATIVE,
                      BoundaryConditions::ZERO_DERIVATIVE}),
        m_cartesian_communicator(createCartesianTopology(m_global_bcs)),
        m_cartesian_coordinates(cartesianCoordinates(m_cartesian_communicator)),
        m_neighbours(neighbours(m_cartesian_communicator)),
        m_local_boundary_conditions(localBoundaryConditions(m_global_bcs, m_neighbours))
    {
    }

    /**
     * @brief Create a cartesian topology from all ranks within MPI_COMM_WORLD
     */
    static MPI_Comm createCartesianTopology(const std::array<BoundaryConditions, 6>& global_bcs)
    {
        int num_ranks;
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        for (int axis = 0; axis < 6; axis += 2)
        {
            if (global_bcs[axis] != global_bcs[axis + 1] and
                (global_bcs[axis] == BoundaryConditions::PERIODIC || global_bcs[axis + 1] == BoundaryConditions::PERIODIC))
                throw std::invalid_argument("Both sides of a periodic boundary must be periodic");
        }

        std::array<int, 3> periods;
        for (int axis = 0; axis < 3; axis++)
            periods[axis] = (int)(global_bcs[axis * 2U] == BoundaryConditions::PERIODIC);

        std::array<int, 3> dimensions {};
        MPI_Dims_create(num_ranks, 3, dimensions.data());

        MPI_Comm cartesian_communicator;
        MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions.data(), periods.data(), true, &cartesian_communicator);
        return cartesian_communicator;
    }

    /**
     * @brief Get the coordinates of this rank in the cartesian communicator.
     *
     * @param[in] cartesian_communicator - The cartesian communicator to
     *                                     extract this ranks coordinates from.
     */
    static std::array<int, 3> cartesianCoordinates(const MPI_Comm& cartesian_communicator)
    {
        int my_cart_rank = myRank(cartesian_communicator);

        std::array<int, 3> coordinates;
        MPI_Cart_coords(cartesian_communicator, my_cart_rank, 3, coordinates.data());
        return coordinates;
    }

    /**
     * @brief Get this ranks neighbours in the cartesian communicator.
     *
     * @param[in] cartesian_communicator - The cartesian communicator to get
     *                                     this ranks neighbours conditions
     *                                     from.
     */
    static std::array<int, 6> neighbours(const MPI_Comm& cartesian_communicator)
    {
        std::array<int, 6> neighbours;
        MPI_Cart_shift(cartesian_communicator, Axis::X, 1, &neighbours[0], &neighbours[1]);
        MPI_Cart_shift(cartesian_communicator, Axis::Y, 1, &neighbours[2], &neighbours[3]);
        MPI_Cart_shift(cartesian_communicator, Axis::Z, 1, &neighbours[4], &neighbours[5]);
        return neighbours;
    }

    /**
     * @brief Get the boundary conditions for this ranks cells.
     *
     * @param[in] cartesian_communicator - The cartesian communicator to get
     *                                     this ranks boundary conditions from.
     * @param[in] global_bcs - The global boundary conditions for this problem.
     * @param[in] neighbours - This ranks neighbours.
     *
     * @returns The boundary conditions for this ranks block of cells.
     */
    static std::array<BoundaryConditions, 6> localBoundaryConditions(const std::array<BoundaryConditions, 6>& global_bcs,
                                                                     const std::array<int, 6>& neighbours)
    {
        std::array<BoundaryConditions, 6> bcs {};
        for (int axis_direction = 0; axis_direction < 6; axis_direction++)
        {
            if (global_bcs[axis_direction] == BoundaryConditions::NONE) throw std::runtime_error("Found NONE");
            bcs[axis_direction] =
                neighbours[axis_direction] == MPI_PROC_NULL ? global_bcs[axis_direction] : BoundaryConditions::NONE;
        }
        return bcs;
    }

    /**
     * @brief Get the cartesian communicator.
     */
    MPI_Comm cartesian_communicator()
    { return m_cartesian_communicator; }

  protected:
    /// @brief Global boundary conditions.
    const std::array<BoundaryConditions, 6> m_global_bcs;
    /// @brief The cartesian communicator.
    const MPI_Comm m_cartesian_communicator;
    /// @brief The coordinates of this rank in the cartesian communicator.
    const std::array<int, 3> m_cartesian_coordinates;
    /// @brief The ranks of this ranks neighbours: {-x, +x, -y, +y, -z, +z}.
    const std::array<int, 6> m_neighbours;
    /// @brief Boundary conditions for this ranks cells.
    const std::array<BoundaryConditions, 6> m_local_boundary_conditions;

    /**
     * @brief Get my rank in the given communicator.
     *
     * @param[in] communicator - The communicator to which this rank belongs.
     *
     * @returns The rank of this process in the given `communicator`.
     */
    static int myRank(MPI_Comm communicator)
    {
        int my_rank;
        MPI_Comm_rank(communicator, &my_rank);
        return my_rank;
    }
};

#endif  // MpiTest_H
