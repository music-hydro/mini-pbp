#include "Array3D.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("Array3D constructor errors for incorrectly sized inputs")
{
    unsigned int n = 3;
    std::vector<double> x(3);
    REQUIRE_THROWS_WITH(Array3D(n, n, n, x),
                        Catch::Matchers::ContainsSubstring("Size of data_1D is not equal to (ni * nj * nk)"));
}

TEST_CASE("Array3D(i,j,k) accesses contiguously")
{

    // Define the problem starting from an arbitrary point along the array
    int ni_full = 3, ni_view = 2;
    int nj_full = 4, nj_view = 2;
    int nk_full = 3, nk_view = 3;
    int starting_index = 2;
    int size           = ni_full * nj_full * nk_full;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d_pointer(size);

    // Populate the 1D array with an incrementing contiguous counter
    for (int i = 0; i < size; i++)
        input_1d_pointer[i] = (double)i;

    // Create an Array3D instance wrapping the 1D array
    Array3D<double> my_array(ni_view, nj_view, nk_view, &input_1d_pointer[0] + starting_index);

    // Verify that looping through row-major i,j,k remains contiguous.
    for (int k = 0; k < nk_view; k++)
    {
        for (int j = 0; j < nj_view; j++)
        {
            for (int i = 0; i < ni_view; i++)
            {
                REQUIRE(my_array(i, j, k) == input_1d_pointer[starting_index]);
                REQUIRE(&my_array(i, j, k) == &input_1d_pointer[starting_index]);
                starting_index++;
            }
        }
    }
}

TEST_CASE("Array3D.cellSlab() returns correct range")
{
    // Define the problem
    int n_full[3] = {3, 4, 3};
    int n_view[3] = {2, 2, 3};
    int size      = n_full[0] * n_full[1] * n_full[2];

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d_pointer(size);

    // Populate the 1D array with an incrementing contiguous counter
    for (int i = 0; i < size; i++)
        input_1d_pointer[i] = (double)i;

    // Create an Array3D instance wrapping the 1D array
    int starting_index = 2;
    Array3D<double> my_array(n_view[0], n_view[1], n_view[2], &input_1d_pointer[0] + starting_index);

    Slab my_slab = my_array.cellSlab();

    for (int i = 0; i < 3; i++)
    {
        REQUIRE(my_slab.start(i) == 0);
        REQUIRE(my_slab.end(i) == n_view[i]);
    }
}

TEST_CASE("Array3D.deepCopy() Makes a copy without reallocating")
{
    unsigned int n          = 3;
    unsigned long long size = n * n * n;

    std::vector<double> x_raw(size);
    std::vector<double> y_raw(size);

    Array3D<double> x(n, n, n, &x_raw[0]);
    Array3D<double> y(n, n, n, &y_raw[0]);

    for (int i = 0; i < size; i++)
    {
        x_raw[i] = 1;
        y_raw[i] = 2;
    }

    x.deepCopy(y);

    int counter = 0;
    for (int k = 0; k < n; k++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                REQUIRE(y(i, j, k) == x(i, j, k));
                REQUIRE(&y(i, j, k) == &y_raw[counter]);
                REQUIRE(&y(i, j, k) != &x_raw[counter]);
                counter++;
            }
        }
    }
}
