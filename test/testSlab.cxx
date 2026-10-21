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
