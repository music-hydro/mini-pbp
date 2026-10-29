#include "Array3DWithGhosts.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Array3DWithGhosts(i,j,k) accesses contiguously")
{
    // Define the problem starting from an arbitrary point along the array
    unsigned int ni_full = 3, ni_view = 2;
    unsigned int nj_full = 4, nj_view = 2;
    unsigned int nk_full = 3, nk_view = 3;
    unsigned int starting_index = 2;
    size_t size                 = ni_full * nj_full * nk_full;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d_pointer(size);

    // Populate the 1D array with an incrementing contiguous counter
    for (int i = 0; i < size; i++)
        input_1d_pointer[i] = (double)i;

    // Create an Array3D instance wrapping the 1D array
    Array3DWithGhosts<double> my_array(ni_view, nj_view, nk_view, &input_1d_pointer[0] + starting_index);

    // Verify that looping through row-major i,j,k remains contiguous for inner cells
    Slab slab = my_array.cellSlab();
    int i0    = slab.start(0);
    int j0    = slab.start(1);
    int k0    = slab.start(2);
    int in    = slab.end(0);
    int jn    = slab.end(1);
    int kn    = slab.end(2);
    for (unsigned int k = k0 + 1; k < kn - 1; k++)
    {
        for (unsigned int j = j0 + 1; j < jn - 1; j++)
        {
            for (unsigned int i = i0 + 1; i < in - 1; i++)
            {
                REQUIRE(my_array(i, j, k) == input_1d_pointer[starting_index]);
                REQUIRE(&my_array(i, j, k) == &input_1d_pointer[starting_index]);
                starting_index++;
            }
        }
    }

    // Verify that ghost cells are contiguous across each plane
    // axis i
    double* current_left  = &my_array(i0, j0 + 1, k0 + 1);
    double* current_right = &my_array(in, j0 + 1, k0 + 1);
    for (unsigned int k = k0 + 1; k < kn - 1; k++)
    {
        for (unsigned int j = j0 + 1; j < jn - 1; j++)
        {
            REQUIRE(&my_array(i0, j, k) == current_left);
            REQUIRE(&my_array(in, j, k) == current_right);
            current_left++;
            current_right++;
        }
    }
    // axis j
    current_left  = &my_array(i0 + 1, j0, k0 + 1);
    current_right = &my_array(i0 + 1, jn, k0 + 1);
    for (unsigned int k = k0 + 1; k < kn - 1; k++)
    {
        for (unsigned int i = i0 + 1; i < in - 1; i++)
        {
            REQUIRE(&my_array(i, j0, k) == current_left);
            REQUIRE(&my_array(i, jn, k) == current_right);
            current_left++;
            current_right++;
        }
    }
    // axis k
    current_left  = &my_array(i0 + 1, j0 + 1, k0);
    current_right = &my_array(i0 + 1, j0 + 1, kn);
    for (unsigned int j = j0 + 1; j < jn - 1; j++)
    {
        for (unsigned int i = i0 + 1; i < in - 1; i++)
        {
            REQUIRE(&my_array(i, j, k0) == current_left);
            REQUIRE(&my_array(i, j, kn) == current_right);
            current_left++;
            current_right++;
        }
    }
}
