#include "Array2D.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("Array2D copy constructor preserves memory locations")
{
    unsigned int n = 3;
    Array2D<double> x(n, n);
    Array2D<double> y(x);

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            REQUIRE(y(i, j) == x(i, j));
            REQUIRE(&y(i, j) == &x(i, j));
        }
    }
}

TEST_CASE("Array2D(i,j) accesses contiguously")
{

    // Define the problem starting from an arbitrary point along the array
    int ni = 30;
    int nj = 40;

    // Create an Array3D instance wrapping the 1D array
    Array2D<double> my_array(ni, nj);

    // Verify that looping through row-major i,j,k remains contiguous.
    double* current_ptr = &my_array(0, 0);
    for (int j = 0; j < nj; j++)
    {
        for (int i = 0; i < ni; i++)
        {
            REQUIRE(&my_array(i, j) == current_ptr);
            current_ptr++;
        }
    }
}
