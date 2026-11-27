#include "DirectionalArray.h"
#include "fixtures/MpiTest.cxx"
#include "GhostExchangePlan.h"
#include "Slab.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>
#include <string>

TEST_CASE_METHOD(MpiTest, "Array3DWithGhosts::neighbourCellPosition returns the correct positions")
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks != 4) { SKIP("Test requires exactly 4 ranks"); }

    std::array<int, 3> global_end;
    for (unsigned int axis = 0; axis < 3; ++axis)
        global_end[axis] = 2 * m_dimensions[axis];
    Slab global_slab({0, 0, 0}, global_end);
    DirectionalArray<BoundaryConditions> global_bcs {
        {BoundaryConditions::ZERO_DERIVATIVE,
         BoundaryConditions::ZERO_DERIVATIVE,
         BoundaryConditions::PERIODIC,
         BoundaryConditions::PERIODIC,
         BoundaryConditions::ZERO_VALUE,
         BoundaryConditions::ZERO_VALUE}
    };
    GhostExchangePlan my_plan(global_slab, global_bcs, m_neighbours, m_dimensions, m_cartesian_communicator);
    Slab local_domain = my_plan.localDomain();

    // Test one axis at a time
    for (unsigned int axis = 0; axis < 3; ++axis)
    {
        // Populate a cell position in the centre of the local domain
        std::array<int, 3> local_cell;
        for (unsigned int other_axis = 0; other_axis < 3; ++other_axis)
            local_cell[other_axis] = local_domain.extent(other_axis) / 2;

        // Test one direction at a time
        for (Direction direction : {Direction::NEGATIVE, Direction::POSITIVE})
        {
            // Set the current axis index to the local boundary
            local_cell[axis] = direction == Direction::NEGATIVE ? 0 : local_domain.extent(axis) - 1;

            // initialise the expected position to the global original local cell position in global coordinates
            std::array<int, 3> expected;
            for (unsigned int other_axis = 0; other_axis < 3; ++other_axis)
                expected[other_axis] = local_domain.start(other_axis) + local_cell[other_axis];

            // Check the current rank is at a global boundary in the current direction
            const bool rank_at_global_boundary = direction == Direction::NEGATIVE ?
                                                     m_cartesian_coordinates[axis] == 0 :
                                                     m_cartesian_coordinates[axis] == m_dimensions[axis] - 1;

            if (!rank_at_global_boundary)
            {
                // If we're at an inner boundary, set the expected index one step in the current direction
                expected[axis] = direction == Direction::NEGATIVE ? local_domain.start(axis) - 1 : local_domain.end(axis);
            }
            else if (global_bcs.get(axis, direction) == BoundaryConditions::PERIODIC)
            {
                // If we're at a periodic boundary, set the expected index to the index at the opposite global boundary
                expected[axis] = direction == Direction::NEGATIVE ? global_slab.end(axis) - 1 : global_slab.start(axis);
            }
            else
            {
                // If we're at a global boundary that is not periodic we should have no neighbour (i.e. -1)
                expected[axis] = -1;
            }

            const std::array<int, 3> actual =
                my_plan.neighbourCellPosition(local_cell[0], local_cell[1], local_cell[2], axis, direction);
            INFO("axis=" + std::to_string(axis) + ", direction=" + std::to_string(static_cast<int>(direction)) +
                 ", coordinates={" + std::to_string(m_cartesian_coordinates[0]) + ", " +
                 std::to_string(m_cartesian_coordinates[1]) + ", " + std::to_string(m_cartesian_coordinates[2]) + "}");
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE_METHOD(MpiTest, "GhostExchangePlan::determineLocalBCs throws for invalid neighbours")
{
    DirectionalArray<int> invalid_neighbours;
    invalid_neighbours.fill(MPI_PROC_NULL);
    DirectionalArray<BoundaryConditions> global_bcs;
    std::string error_msg;

    SECTION("No neighbour for PERIODIC boundary")
    {
        global_bcs.fill(BoundaryConditions::PERIODIC);
        error_msg = "No neighbour found at periodic boundary";
        // The first error will come from the -ve ith direction. Therefore, the error message depends on
        // the valid neighbours in that direction. I.e. if the valid neighbours exists, the local boundary
        // will have been set to INNER.
        if (m_neighbours.get(0U, Direction::NEGATIVE) != MPI_PROC_NULL) error_msg = "No neighbour found at inner boundary";
    }

    SECTION("No neighbour for INNER boundary")
    {
        global_bcs.fill(BoundaryConditions::INNER);
        error_msg = "No neighbour found at inner boundary";
    }

    REQUIRE_THROWS_WITH(
        GhostExchangePlan(m_global_domain_slab, global_bcs, invalid_neighbours, m_dimensions, m_cartesian_communicator),
        Catch::Matchers::ContainsSubstring(error_msg));
}

TEST_CASE_METHOD(MpiTest, "GhostExchangePlan::determineLocalDomain returns the correct local domain")
{
    Slab local_domain = GhostExchangePlan::determineLocalDomain(m_global_domain_slab, m_dimensions, m_cartesian_communicator);
    // Expected result will differ for each rank depending on the total number of ranks

    int num_ranks;
    MPI_Comm_size(m_cartesian_communicator, &num_ranks);
    int rank;
    MPI_Comm_rank(m_cartesian_communicator, &rank);

    std::array<int, 3> expected_start;
    std::array<int, 3> expected_end;
    if (num_ranks == 1)
    {
        expected_start = m_global_domain_slab.start();
        expected_end   = m_global_domain_slab.end();
    }

    if (num_ranks == 4)
    {
        if (rank == 0)
        {
            expected_start = {0, 0, 0};
            expected_end   = {15, 20, 30};
        }
        else if (rank == 1)
        {
            expected_start = {0, 20, 0};
            expected_end   = {15, 40, 30};
        }
        else if (rank == 2)
        {
            expected_start = {15, 0, 0};
            expected_end   = {30, 20, 30};
        }
        else if (rank == 3)
        {
            expected_start = {15, 20, 0};
            expected_end   = {30, 40, 30};
        }
    }

    if (num_ranks == 8)
    {
        if (rank == 0)
        {
            expected_start = {0, 0, 0};
            expected_end   = {15, 20, 15};
        }
        else if (rank == 1)
        {
            expected_start = {0, 0, 15};
            expected_end   = {15, 20, 30};
        }
        else if (rank == 2)
        {
            expected_start = {0, 20, 0};
            expected_end   = {15, 40, 15};
        }
        else if (rank == 3)
        {
            expected_start = {0, 20, 15};
            expected_end   = {15, 40, 30};
        }
        else if (rank == 4)
        {
            expected_start = {15, 0, 0};
            expected_end   = {30, 20, 15};
        }
        else if (rank == 5)
        {
            expected_start = {15, 0, 15};
            expected_end   = {30, 20, 30};
        }
        else if (rank == 6)
        {
            expected_start = {15, 20, 0};
            expected_end   = {30, 40, 15};
        }
        else if (rank == 7)
        {
            expected_start = {15, 20, 15};
            expected_end   = {30, 40, 30};
        }
    }

    for (unsigned int axis = 0; axis < 3; axis++)
    {
        REQUIRE(expected_start[axis] == local_domain.start(axis));
        REQUIRE(expected_end[axis] == local_domain.end(axis));
    }
}
