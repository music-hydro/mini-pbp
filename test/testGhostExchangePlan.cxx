#include "DirectionalArray.h"
#include "fixtures/MpiTest.cxx"
#include "GhostExchangePlan.h"
#include "Slab.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>

TEST_CASE_METHOD(MpiTest, "Array3DWithGhosts::neighbourCellPosition returns the correct positions")
{
    int n = 3;
    // The neighbouring ranks shouldn't matter
    DirectionalArray<int> neighbours({0, 0, 0, 0, 0, 0});
    DirectionalArray<BoundaryConditions> bcs({BoundaryConditions::INNER,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::PERIODIC,
                                              BoundaryConditions::PERIODIC,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::INNER});
    // Choose global slab so indices are on global boundaries
    Slab global_slab({0, 0, 0}, {n, n, n});

    GhostExchangePlan my_plan(global_slab, bcs, neighbours, m_dimensions, m_cartesian_communicator);

    // i axis
    int axis                                       = 0;
    std::array<int, 3> negative_neighbour_position = my_plan.neighbourCellPosition(0, 1, 1, axis, Direction::NEGATIVE);
    REQUIRE(negative_neighbour_position[0] == -1);
    REQUIRE(negative_neighbour_position[1] == 1);
    REQUIRE(negative_neighbour_position[2] == 1);
    std::array<int, 3> positive_neighbour_position = my_plan.neighbourCellPosition(2, 1, 1, axis, Direction::POSITIVE);
    REQUIRE(positive_neighbour_position[0] == -1);
    REQUIRE(positive_neighbour_position[1] == 1);
    REQUIRE(positive_neighbour_position[2] == 1);

    // j axis
    axis                        = 1;
    negative_neighbour_position = my_plan.neighbourCellPosition(1, 0, 1, axis, Direction::NEGATIVE);
    REQUIRE(negative_neighbour_position[0] == 1);
    REQUIRE(negative_neighbour_position[1] == 2);  // PERIODIC
    REQUIRE(negative_neighbour_position[2] == 1);
    positive_neighbour_position = my_plan.neighbourCellPosition(1, 2, 1, axis, Direction::POSITIVE);
    REQUIRE(positive_neighbour_position[0] == 1);
    REQUIRE(positive_neighbour_position[1] == 0);  // PERIODIC
    REQUIRE(positive_neighbour_position[2] == 1);

    // K axis
    axis                        = 2;
    negative_neighbour_position = my_plan.neighbourCellPosition(1, 1, 0, axis, Direction::NEGATIVE);
    REQUIRE(negative_neighbour_position[0] == 1);
    REQUIRE(negative_neighbour_position[1] == 1);
    REQUIRE(negative_neighbour_position[2] == -1);
    positive_neighbour_position = my_plan.neighbourCellPosition(1, 1, 2, axis, Direction::POSITIVE);
    REQUIRE(positive_neighbour_position[0] == 1);
    REQUIRE(positive_neighbour_position[1] == 1);
    REQUIRE(positive_neighbour_position[2] == -1);
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
        global_bcs.fill(BoundaryConditions::ZERO_DERIVATIVE);
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

    for (unsigned int axis = 0; axis < 3; axis++)
    {
        REQUIRE(expected_start[axis] == local_domain.start(axis));
        REQUIRE(expected_end[axis] == local_domain.end(axis));
    }
}
