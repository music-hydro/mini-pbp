#include "GhostExchangePlan.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("GhostExchangePlan Constructor does not make copies of data")
{
    unsigned int n = 3;
    size_t size    = n * n * n;
    std::vector<double> x_raw(size);
    Array3D<double> x(n, n, n, x_raw);

    std::array<BoundaryConditions, 6> bcs  = {BoundaryConditions::NONE,
                                              BoundaryConditions::NONE,
                                              BoundaryConditions::ZERO_DERIVATIVE,
                                              BoundaryConditions::ZERO_DERIVATIVE,
                                              BoundaryConditions::ZERO_VALUE,
                                              BoundaryConditions::ZERO_VALUE};
    std::array<unsigned int, 6> neighbours = {0, 0, 0, 0, 0, 0};

    GhostExchangePlan<double> plan(x, bcs, neighbours);

    Array3D<double> array_from_plan = plan.array();

    Slab slab = x.cellSlab();
    for (int k = slab.start(2); k < slab.end(2); k++)
    {
        for (int j = slab.start(1); j < slab.end(1); j++)
        {
            for (int i = slab.start(0); i < slab.end(0); i++)
            {
                REQUIRE(array_from_plan(i, j, k) == x(i, j, k));
                REQUIRE(&array_from_plan(i, j, k) == &x(i, j, k));
            }
        }
    }
}
