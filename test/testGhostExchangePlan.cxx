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
    // No neighbour across the i and k faces of the domain, periodic along j. An INNER face with a neighbour rank at the
    // domain boundary would be periodic too (see the next test).
    DirectionalArray<int> neighbours({MPI_PROC_NULL, MPI_PROC_NULL, 0, 0, MPI_PROC_NULL, MPI_PROC_NULL});
    DirectionalArray<BoundaryConditions> bcs({BoundaryConditions::ZERO_VALUE,
                                              BoundaryConditions::ZERO_VALUE,
                                              BoundaryConditions::PERIODIC,
                                              BoundaryConditions::PERIODIC,
                                              BoundaryConditions::ZERO_DERIVATIVE,
                                              BoundaryConditions::ZERO_DERIVATIVE});
    // Choose global slab so indices are on global boundaries
    Slab global_slab({0, 0, 0}, {n, n, n});

    GhostExchangePlan<double> my_plan(global_slab, bcs, neighbours, m_cartesian_communicator);

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

TEST_CASE_METHOD(MpiTest, "GhostExchangePlan::neighbourCellPosition counts from the global domain and wraps across INNER faces")
{
    // Cells 2 to 3 along i of a 4 x 3 x 3 domain, with a neighbour rank on every face (a periodic cartesian
    // communicator), so every face is INNER
    DirectionalArray<int> neighbours({0, 0, 0, 0, 0, 0});
    DirectionalArray<BoundaryConditions> bcs({BoundaryConditions::INNER,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::INNER});
    GhostExchangePlan<double> my_plan(
        Slab({0, 0, 0}, {4, 3, 3}), Slab({2, 0, 0}, {4, 3, 3}), bcs, neighbours, m_cartesian_communicator);

    // Local cell 0 along i is global cell 2: its negative neighbour is global cell 1, on the other rank
    REQUIRE(my_plan.neighbourCellPosition(0, 1, 1, 0, Direction::NEGATIVE) == std::array<int, 3> {1, 1, 1});
    // Local cell 1 along i is global cell 3: its positive neighbour is across the domain boundary, global cell 0
    REQUIRE(my_plan.neighbourCellPosition(1, 1, 1, 0, Direction::POSITIVE) == std::array<int, 3> {0, 1, 1});
    // Along j the rank holds the whole domain: cell 0 wraps to cell 2
    REQUIRE(my_plan.neighbourCellPosition(1, 0, 1, 1, Direction::NEGATIVE) == std::array<int, 3> {3, 2, 1});
}

TEST_CASE("GhostExchangePlan Constructor throws for boundary without a required neighbour",
          "[boundary_conditions][expected_error_message]")
{
    DirectionalArray<int> neighbours = {MPI_PROC_NULL, 1, 2, 3, 4, 5};
    Slab global_domain({0, 0, 0}, {3, 3, 3});
    DirectionalArray<BoundaryConditions> global_bcs;
    std::string expected_error_message;

    SECTION("PERIODIC boundary without neighbour")
    {
        expected_error_message = "No neighbour found at periodic boundary";
        global_bcs             = {BoundaryConditions::PERIODIC,
                                  BoundaryConditions::PERIODIC,
                                  BoundaryConditions::ZERO_DERIVATIVE,
                                  BoundaryConditions::ZERO_DERIVATIVE,
                                  BoundaryConditions::ZERO_VALUE,
                                  BoundaryConditions::ZERO_VALUE};
    }

    SECTION("INNER boundary without neighbour")
    {
        expected_error_message = "No neighbour found at inner boundary";
        global_bcs             = {BoundaryConditions::INNER,
                                  BoundaryConditions::ZERO_VALUE,
                                  BoundaryConditions::ZERO_DERIVATIVE,
                                  BoundaryConditions::ZERO_DERIVATIVE,
                                  BoundaryConditions::ZERO_VALUE,
                                  BoundaryConditions::ZERO_VALUE};
    }

    REQUIRE_THROWS_WITH(GhostExchangePlan<double>(global_domain, global_bcs, neighbours, MPI_COMM_WORLD),
                        Catch::Matchers::ContainsSubstring(expected_error_message));
}
