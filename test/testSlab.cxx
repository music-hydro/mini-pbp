#include "Slab.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Slab is read only after construction")
{
    Slab slab({1, 2, 3}, {2, 4, 6});

    for (int i = 0; i < 3; i++)
    {
        int expected = i + 1;
        REQUIRE(slab.start(i) == expected);
        REQUIRE(slab.end(i) == expected * 2);
    }
}

TEST_CASE("Slab::volume returns the correct number of cells")
{
    SECTION("All 0->3")
    {
        unsigned int n = 3;
        Slab slab({0, 0, 0}, {n, n, n});
        REQUIRE(slab.volume() == 27);
    }

    SECTION("Non-zero start with different lengths")
    {
        unsigned int i0 = 1;
        unsigned int j0 = 3;
        unsigned int k0 = 2;
        unsigned int ni = 5;
        unsigned int nj = 10;
        unsigned int nk = 7;
        Slab slab({i0, j0, k0}, {ni, nj, nk});
        REQUIRE(slab.volume() == 140);
    }
}
