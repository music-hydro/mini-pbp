#include "DirectionalArray.h"
#include "fixtures/MpiTest.cxx"
#include "GhostExchangePlan.h"
#include "Slab.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>

TEST_CASE_METHOD(MpiTest, "Array3DWithGhosts::neighbourCellPosition returns the correct weights")
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
    DirectionalArray<int> invalid_neighbours(
        {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL});
    DirectionalArray<BoundaryConditions> global_bcs;
    std::string error_msg;

    SECTION("No neighbour for PERIODIC boundary")
    {
        global_bcs = {BoundaryConditions::PERIODIC,
                      BoundaryConditions::PERIODIC,
                      BoundaryConditions::ZERO_VALUE,
                      BoundaryConditions::ZERO_VALUE,
                      BoundaryConditions::ZERO_DERIVATIVE,
                      BoundaryConditions::ZERO_DERIVATIVE};
        error_msg  = "No neighbour found at periodic boundary";
    }

    SECTION("No neighbour for INNER boundary")
    {
        global_bcs = {BoundaryConditions::INNER,
                      BoundaryConditions::INNER,
                      BoundaryConditions::ZERO_VALUE,
                      BoundaryConditions::ZERO_VALUE,
                      BoundaryConditions::ZERO_DERIVATIVE,
                      BoundaryConditions::ZERO_DERIVATIVE};
        error_msg  = "No neighbour found at inner boundary";
    }

    REQUIRE_THROWS_WITH(
        GhostExchangePlan::determineLocalBCs(m_cartesian_communicator, m_dimensions, global_bcs, invalid_neighbours),
        Catch::Matchers::ContainsSubstring(error_msg));
}
