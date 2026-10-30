#include "Array3DWithGhosts.h"
#include "fixtures/Array3DWithGhostsTest.cxx"
#include <catch2/catch_test_macros.hpp>

TEST_CASE_METHOD(Array3DWithGhostsTest, "Array3DWithGhosts(i,j,k) accesses contiguously")
{
    // Verify that the inner cells match the values from the input Array3D
    Slab slab = this->my_array_with_ghosts->cellSlab();
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
                REQUIRE((*my_array_with_ghosts)(i, j, k) == (*my_array)(i, j, k));
                REQUIRE(&(*my_array_with_ghosts)(i, j, k) == &(*my_array)(i, j, k));
            }
        }
    }

    // Verify that ghost and edge cells are contiguous across each plane
    // axis i
    double* current_ghost_left  = &(*my_array_with_ghosts)(i0 - 1, j0, k0);
    double* current_ghost_right = &(*my_array_with_ghosts)(in, j0, k0);
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE(&(*my_array_with_ghosts)(i0 - 1, j, k) == current_ghost_left);
            REQUIRE(&(*my_array_with_ghosts)(in, j, k) == current_ghost_right);
            current_ghost_left++;
            current_ghost_right++;
        }
    }
    // axis j
    for (unsigned int k = k0; k < kn; k++)
    {
        current_ghost_left  = &(*my_array_with_ghosts)(i0 + 1, j0 - 1, k);
        current_ghost_right = &(*my_array_with_ghosts)(i0 + 1, jn, k);
        for (unsigned int i = i0 + 1; i < in; i++)
        {
            REQUIRE(&(*my_array_with_ghosts)(i, j0 - 1, k) == current_ghost_left);
            REQUIRE(&(*my_array_with_ghosts)(i, jn, k) == current_ghost_right);
            current_ghost_left++;
            current_ghost_right++;
        }
    }
    // axis k
    current_ghost_left  = &(*my_array_with_ghosts)(i0, j0, k0 - 1);
    current_ghost_right = &(*my_array_with_ghosts)(i0, j0, kn);
    for (unsigned int j = j0 + 1; j < jn; j++)
    {
        current_ghost_left  = &(*my_array_with_ghosts)(i0 + 1, j, k0 - 1);
        current_ghost_right = &(*my_array_with_ghosts)(i0 + 1, j, kn);
        for (unsigned int i = i0 + 1; i < in; i++)
        {
            REQUIRE(&(*my_array_with_ghosts)(i, j, k0 - 1) == current_ghost_left);
            REQUIRE(&(*my_array_with_ghosts)(i, j, kn) == current_ghost_right);
            current_ghost_left++;
            current_ghost_right++;
        }
    }
}

TEST_CASE_METHOD(Array3DWithGhostsTest, "Array3DWithGhosts::updateEdges populates edges correctly")
{
    // Verify edges do not contain the expected values
    Slab slab = my_array_with_ghosts->cellSlab();
    int i0 = slab.start(0), j0 = slab.start(1), k0 = slab.start(2);
    int in = slab.end(0), jn = slab.end(1), kn = slab.end(2);
    Array2D<double> i_left_edge  = my_array_with_ghosts->edge(0, Side::LEFT);
    Array2D<double> i_right_edge = my_array_with_ghosts->edge(0, Side::RIGHT);
    Array2D<double> j_left_edge  = my_array_with_ghosts->edge(1, Side::LEFT);
    Array2D<double> j_right_edge = my_array_with_ghosts->edge(1, Side::RIGHT);
    Array2D<double> k_left_edge  = my_array_with_ghosts->edge(2, Side::LEFT);
    Array2D<double> k_right_edge = my_array_with_ghosts->edge(2, Side::RIGHT);

    // axis i
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE(i_left_edge(j, k) != (*my_array_with_ghosts)(i0, j, k));
            REQUIRE(i_right_edge(j, k) != (*my_array_with_ghosts)(in - 1, j, k));
        }
    }
    // axis j
    for (int k = k0; k < kn; k++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(j_left_edge(i, k) != (*my_array_with_ghosts)(i, j0, k));
            REQUIRE(j_right_edge(i, k) != (*my_array_with_ghosts)(i, jn - 1, k));
        }
    }
    // axis k
    for (int j = j0; j < jn; j++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(k_left_edge(i, j) != (*my_array_with_ghosts)(i, j, k0));
            REQUIRE(k_right_edge(i, j) != (*my_array_with_ghosts)(i, j, kn - 1));
        }
    }

    // Update edges
    my_array_with_ghosts->updateEdges();

    // Verify edges contain the expected values
    // axis i
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE(i_left_edge(j, k) == (*my_array_with_ghosts)(i0, j, k));
            REQUIRE(i_right_edge(j, k) == (*my_array_with_ghosts)(in - 1, j, k));
        }
    }
    // axis j
    for (int k = k0; k < kn; k++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(j_left_edge(i, k) == (*my_array_with_ghosts)(i, j0, k));
            REQUIRE(j_right_edge(i, k) == (*my_array_with_ghosts)(i, jn - 1, k));
        }
    }
    // axis k
    for (int j = j0; j < jn; j++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(k_left_edge(i, j) == (*my_array_with_ghosts)(i, j, k0));
            REQUIRE(k_right_edge(i, j) == (*my_array_with_ghosts)(i, j, kn - 1));
        }
    }
}
