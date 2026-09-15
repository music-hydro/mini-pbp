#include "Array3D.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Array3D(i,j,k) accesses contiguously")
{

    // Define the problem
    int ni_full = 3, ni_view = 2;
    int nj_full = 4, nj_view = 2;
    int nk_full = 3, nk_view = 3;
    int size = ni_full * nj_full * nk_full;

    // Allocate a 1D array equal to the size of a 3D array
    double* input_1d_pointer = new double[size];

    // Populate the 1D array with an incrementing contiguous counter
    for (int i = 0; i < size; i++)
        input_1d_pointer[i] = (double)i;

    // Create an Array3D instance wrapping the 1D array
    int starting_index = 2;
    Array3D<double> my_array(input_1d_pointer + starting_index, ni_view, nj_view, nk_view);

    // Verify that looping through row-major i,j,k remains contiguous.
    for (int i = 0; i < ni_view; i++)
    {
        for (int j = 0; j < nj_view; j++)
        {
            for (int k = 0; k < nk_view; k++)
            {
                CHECK(my_array(i, j, k) == input_1d_pointer[starting_index]);
                REQUIRE(&my_array(i, j, k) == &input_1d_pointer[starting_index]);
                starting_index++;
            }
        }
    }
}
