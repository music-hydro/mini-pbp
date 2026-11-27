#include "DirectionalArray.h"
#include "fixtures/MpiTest.cxx"
#include "GhostExchangePlan.h"
#include "Slab.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>
#include <string>

TEST_CASE_METHOD(MpiTest, "GhostExchangePlan Constructor throws for boundary without a required neighbour")
{
    DirectionalArray<int> invalid_neighbours;
    invalid_neighbours.fill(MPI_PROC_NULL);
    DirectionalArray<BoundaryConditions> global_bcs;
    std::string expected_error_message;

    SECTION("PERIODIC boundary without neighbour")
    {
        global_bcs.fill(BoundaryConditions::PERIODIC);
        expected_error_message = "No neighbour found at periodic boundary";
        // The first error will come from the -ve ith direction. Therefore, the error message depends on
        // the valid neighbours in that direction. I.e. if the valid neighbours exists, the local boundary
        // will have been set to INNER.
        if (m_neighbours.get(0U, Direction::NEGATIVE) != MPI_PROC_NULL)
            expected_error_message = "No neighbour found at inner boundary";
    }

    SECTION("INNER boundary without neighbour")
    {
        global_bcs.fill(BoundaryConditions::INNER);
        expected_error_message = "No neighbour found at inner boundary";
    }

    REQUIRE_THROWS_WITH(
        GhostExchangePlan(m_global_domain_slab, global_bcs, invalid_neighbours, m_dimensions, m_cartesian_communicator),
        Catch::Matchers::ContainsSubstring(expected_error_message));
}

TEST_CASE_METHOD(MpiTest, "GhostExchangePlan Verify local domain and bcs are correct")
{
    DirectionalArray<BoundaryConditions> global_bcs;
    global_bcs.fill(BoundaryConditions::ZERO_DERIVATIVE);

    // No divisible by 8 to allow testing where the remainder is placed.
    Slab global_domain({0, 0, 0}, {31, 41, 31});
    GhostExchangePlan plan(global_domain, global_bcs, m_neighbours, m_dimensions, m_cartesian_communicator);

    // Expected result will differ for each rank depending on the total number of ranks
    int num_ranks;
    MPI_Comm_size(m_cartesian_communicator, &num_ranks);
    int rank;
    MPI_Comm_rank(m_cartesian_communicator, &rank);

    std::array<int, 3> expected_start;
    std::array<int, 3> expected_end;
    DirectionalArray<BoundaryConditions> expected_local_bcs = global_bcs;
    if (num_ranks == 1)
    {
        expected_start = global_domain.start();
        expected_end   = global_domain.end();
    }


    // Domain is broken down in only the first 2 dimensions (i and j)
    if (num_ranks == 4)
    {
        if (rank == 0)
        {
            expected_start                                 = {0, 0, 0};
            expected_end                                   = {15, 20, 31};
            expected_local_bcs.get(0, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::POSITIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 1)
        {
            expected_start                                 = {0, 20, 0};
            expected_end                                   = {15, 41, 31};
            expected_local_bcs.get(0, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::NEGATIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 2)
        {
            expected_start                                 = {15, 0, 0};
            expected_end                                   = {31, 20, 31};
            expected_local_bcs.get(0, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::POSITIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 3)
        {
            expected_start                                 = {15, 20, 0};
            expected_end                                   = {31, 41, 31};
            expected_local_bcs.get(0, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::NEGATIVE) = BoundaryConditions::INNER;
        }
    }

    // Domain is broken down in all 3 dimensions (i, then k, then j)
    if (num_ranks == 8)
    {
        if (rank == 0)
        {
            expected_start                                 = {0, 0, 0};
            expected_end                                   = {15, 20, 15};
            expected_local_bcs.get(0, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(2, Direction::POSITIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 1)
        {
            expected_start                                 = {0, 0, 15};
            expected_end                                   = {15, 20, 31};
            expected_local_bcs.get(0, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(2, Direction::NEGATIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 2)
        {
            expected_start                                 = {0, 20, 0};
            expected_end                                   = {15, 41, 15};
            expected_local_bcs.get(0, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(2, Direction::POSITIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 3)
        {
            expected_start                                 = {0, 20, 15};
            expected_end                                   = {15, 41, 31};
            expected_local_bcs.get(0, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(2, Direction::NEGATIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 4)
        {
            expected_start                                 = {15, 0, 0};
            expected_end                                   = {31, 20, 15};
            expected_local_bcs.get(0, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(2, Direction::POSITIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 5)
        {
            expected_start                                 = {15, 0, 15};
            expected_end                                   = {31, 20, 31};
            expected_local_bcs.get(0, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::POSITIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(2, Direction::NEGATIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 6)
        {
            expected_start                                 = {15, 20, 0};
            expected_end                                   = {31, 41, 15};
            expected_local_bcs.get(0, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(2, Direction::POSITIVE) = BoundaryConditions::INNER;
        }
        else if (rank == 7)
        {
            expected_start                                 = {15, 20, 15};
            expected_end                                   = {31, 41, 31};
            expected_local_bcs.get(0, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(1, Direction::NEGATIVE) = BoundaryConditions::INNER;
            expected_local_bcs.get(2, Direction::NEGATIVE) = BoundaryConditions::INNER;
        }
    }

    CAPTURE(rank, num_ranks);
    for (unsigned int axis = 0; axis < 3; axis++)
    {
        // Local domain
        REQUIRE(expected_start[axis] == plan.localDomain().start(axis));
        REQUIRE(expected_end[axis] == plan.localDomain().end(axis));

        // Local boundary conditions
        REQUIRE(expected_local_bcs == plan.localBoundaryConditions());
    }
}
