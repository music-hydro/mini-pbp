#include "Slab.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

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

TEST_CASE("Slab::extent returns the domain size for a given axis")
{
    Slab slab({-1, 4, 2}, {3, 6, 5});
    REQUIRE(slab.extent(0) == 4);
    REQUIRE(slab.extent(1) == 2);
    REQUIRE(slab.extent(2) == 3);
}

TEST_CASE("Slab::extent Throws for an invalid axis")
{
    Slab slab({0, 0, 0}, {3, 3, 3});
    REQUIRE_THROWS_WITH(slab.extent(4), Catch::Matchers::ContainsSubstring("Invalid axis: 4"));
}
