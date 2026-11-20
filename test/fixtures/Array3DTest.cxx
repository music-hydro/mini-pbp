/**
 * @file Array3DTest
 *
 * @brief Helpers for tests of Array3D
 */

#ifndef Array3DTest_H
#define Array3DTest_H

#include "Array3D.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

struct Array3DTest
{
    /// @brief Every value of `a` equals `expected` (or is NaN when `expected` is NaN, i.e. not written)
    static void requireAll(const Array3D<double>& a, const double expected)
    {
        const Slab s = a.cellSlab();
        for (int k = s.start(2); k < s.end(2); k++)
        {
            for (int j = s.start(1); j < s.end(1); j++)
            {
                for (int i = s.start(0); i < s.end(0); i++)
                {
                    if (std::isnan(expected)) { REQUIRE(std::isnan(a(i, j, k))); }
                    else
                    {
                        REQUIRE_THAT(a(i, j, k), Catch::Matchers::WithinRel(expected, 1e-14));
                    }
                }
            }
        }
    }
};

#endif  // Array3DTest_H
