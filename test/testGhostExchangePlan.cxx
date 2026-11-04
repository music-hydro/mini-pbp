#include "GhostExchangePlan.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>

TEST_CASE("GhostExchangePlan Constructor does not make copies of data")
{
    unsigned int n = 3;
    size_t size    = n * n * n;
    std::vector<double> x_raw(size);
    Array3D<double> x(n, n, n, x_raw);

    std::array<BoundaryConditions, 6> bcs = {BoundaryConditions::NONE,
                                             BoundaryConditions::NONE,
                                             BoundaryConditions::ZERO_DERIVATIVE,
                                             BoundaryConditions::ZERO_DERIVATIVE,
                                             BoundaryConditions::ZERO_VALUE,
                                             BoundaryConditions::ZERO_VALUE};
    std::array<int, 6> neighbours         = {0, 0, 0, 0, 0, 0};

    GhostExchangePlan<double> plan(x, bcs, neighbours, MPI_COMM_WORLD);

    Array3D<double> array_from_plan = plan.array();

    Slab slab           = x.cellSlab();
    double* current_ptr = x_raw.data();
    for (int k = slab.start(2); k < slab.end(2); k++)
    {
        for (int j = slab.start(1); j < slab.end(1); j++)
        {
            for (int i = slab.start(0); i < slab.end(0); i++)
            {
                double* x_array_ptr = &x(i, j, k);
                REQUIRE(array_from_plan(i, j, k) == *x_array_ptr);
                REQUIRE(&array_from_plan(i, j, k) == x_array_ptr);
                REQUIRE(x_array_ptr == current_ptr);
                current_ptr++;
            }
        }
    }
}

TEST_CASE("GhostExchangePlan Constructor throws for periodic boundary without neighbour")
{
    unsigned int n = 4;
    size_t size    = n * n * n;
    std::vector<double> x_raw(size);
    Array3D<double> x(n, n, n, x_raw);

    std::array<int, 6> neighbours                = {MPI_PROC_NULL, 1, 2, 3, 4, 5};
    std::array<BoundaryConditions, 6> global_bcs = {BoundaryConditions::PERIODIC,
                                                    BoundaryConditions::PERIODIC,
                                                    BoundaryConditions::ZERO_DERIVATIVE,
                                                    BoundaryConditions::ZERO_DERIVATIVE,
                                                    BoundaryConditions::ZERO_VALUE,
                                                    BoundaryConditions::ZERO_VALUE};

    REQUIRE_THROWS_WITH(GhostExchangePlan<double>(x, global_bcs, neighbours, MPI_COMM_WORLD),
                        Catch::Matchers::ContainsSubstring("No neighbour found at periodic boundary"));
}
