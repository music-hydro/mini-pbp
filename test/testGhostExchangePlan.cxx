#include "DirectionalArray.h"
#include "GhostExchangePlan.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>

TEST_CASE("GhostExchangePlan Constructor throws for boundary without a required neighbour",
          "[boundary_conditions][expected_error_message]")
{
    DirectionalArray<int> neighbours = {MPI_PROC_NULL, 1, 2, 3, 4, 5};
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

    REQUIRE_THROWS_WITH(GhostExchangePlan<double>(global_bcs, neighbours, MPI_COMM_WORLD),
                        Catch::Matchers::ContainsSubstring(expected_error_message));
}
