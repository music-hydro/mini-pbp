#include "Array2D.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Array2D copy constructor preserves memory locations")
{
    const int n = 3;
    Array2D<double> x(n, n);
    Array2D<double> y(x);

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            REQUIRE(y(i, j) == x(i, j));
            REQUIRE(&y(i, j) == &x(i, j));
        }
    }
}

TEST_CASE("Array2D copy remains valid after original is destroyed")
{
    Array2D<double>* copy = nullptr;
    {
        Array2D<double> original(4, 5);
        original(1, 2) = 42.0;
        copy           = new Array2D<double>(original);

        REQUIRE((*copy)(1, 2) == 42.0);
        REQUIRE(&(*copy)(1, 2) == &original(1, 2));
    }

    REQUIRE((*copy)(1, 2) == 42.0);
    (*copy)(1, 2) = 17.0;
    REQUIRE((*copy)(1, 2) == 17.0);
    delete copy;
}

TEST_CASE("Array2D(i,j) accesses contiguously")
{
    const int ni = 30;
    const int nj = 40;

    // Create an Array2D instance wrapping the 1D array
    Array2D<double> my_array(ni, nj);

    // Verify that looping through row-major i,j remains contiguous.
    double* current_ptr = &my_array(0, 0);
    for (int j = 0; j < nj; ++j)
    {
        for (int i = 0; i < ni; ++i)
        {
            REQUIRE(&my_array(i, j) == current_ptr);
            current_ptr++;
        }
    }
}
