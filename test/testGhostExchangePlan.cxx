#include "GhostExchangePlan.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>

TEST_CASE("GhostExchangePlan Constructor throws for PERIODIC boundary without neighbour")
{
    std::array<int, 6> neighbours                = {MPI_PROC_NULL, 1, 2, 3, 4, 5};
    std::array<BoundaryConditions, 6> global_bcs = {BoundaryConditions::PERIODIC,
                                                    BoundaryConditions::PERIODIC,
                                                    BoundaryConditions::ZERO_DERIVATIVE,
                                                    BoundaryConditions::ZERO_DERIVATIVE,
                                                    BoundaryConditions::ZERO_VALUE,
                                                    BoundaryConditions::ZERO_VALUE};

    REQUIRE_THROWS_WITH(GhostExchangePlan<double>(global_bcs, neighbours, MPI_COMM_WORLD),
                        Catch::Matchers::ContainsSubstring("No neighbour found at periodic boundary"));
}

TEST_CASE("GhostExchangePlan Constructor throws for INNER boundary without neighbour")
{
    std::array<int, 6> neighbours                = {MPI_PROC_NULL, 1, 2, 3, 4, 5};
    std::array<BoundaryConditions, 6> global_bcs = {BoundaryConditions::INNER,
                                                    BoundaryConditions::INNER,
                                                    BoundaryConditions::ZERO_DERIVATIVE,
                                                    BoundaryConditions::ZERO_DERIVATIVE,
                                                    BoundaryConditions::ZERO_VALUE,
                                                    BoundaryConditions::ZERO_VALUE};

    REQUIRE_THROWS_WITH(GhostExchangePlan<double>(global_bcs, neighbours, MPI_COMM_WORLD),
                        Catch::Matchers::ContainsSubstring("No neighbour found at inner boundary"));
}
