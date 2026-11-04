#include "Array3DWithGhosts.h"
#include "fixtures/Array3DWithGhostsTest.cxx"
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

TEST_CASE_METHOD(Array3DWithGhostsTest, "Array3DWithGhosts(i,j,k) accesses contiguously")
{
    Slab slab = this->my_array_with_ghosts->cellSlab();
    int i0    = slab.start(0);
    int j0    = slab.start(1);
    int k0    = slab.start(2);
    int in    = slab.end(0);
    int jn    = slab.end(1);
    int kn    = slab.end(2);

    // Verify that the inner cells match the values from the input Array3D and are contiguous
    double* current_ptr = &(*my_array)(i0, j0, k0);
    for (int k = k0; k < kn; ++k)
    {
        for (int j = j0; j < jn; ++j)
        {
            for (int i = i0; i < in; ++i)
            {
                double* const my_array_ptr = &(*my_array)(i, j, k);
                REQUIRE((*my_array_with_ghosts)(i, j, k) == *my_array_ptr);
                REQUIRE(&(*my_array_with_ghosts)(i, j, k) == my_array_ptr);
                REQUIRE(my_array_ptr == current_ptr);
                current_ptr++;
            }
        }
    }

    // Verify that ghost and boundary cells are contiguous across each plane
    // axis i
    double* current_ghost_negative = &(*my_array_with_ghosts)(i0 - 1, j0, k0);
    double* current_ghost_positive = &(*my_array_with_ghosts)(in, j0, k0);
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE(&(*my_array_with_ghosts)(i0 - 1, j, k) == current_ghost_negative);
            REQUIRE(&(*my_array_with_ghosts)(in, j, k) == current_ghost_positive);
            current_ghost_negative++;
            current_ghost_positive++;
        }
    }
    // axis j
    current_ghost_negative = &(*my_array_with_ghosts)(i0, j0 - 1, k0);
    current_ghost_positive = &(*my_array_with_ghosts)(i0, jn, k0);
    for (int k = k0; k < kn; k++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(&(*my_array_with_ghosts)(i, j0 - 1, k) == current_ghost_negative);
            REQUIRE(&(*my_array_with_ghosts)(i, jn, k) == current_ghost_positive);
            current_ghost_negative++;
            current_ghost_positive++;
        }
    }
    // axis k
    current_ghost_negative = &(*my_array_with_ghosts)(i0, j0, k0 - 1);
    current_ghost_positive = &(*my_array_with_ghosts)(i0, j0, kn);
    for (int j = j0; j < jn; j++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(&(*my_array_with_ghosts)(i, j, k0 - 1) == current_ghost_negative);
            REQUIRE(&(*my_array_with_ghosts)(i, j, kn) == current_ghost_positive);
            current_ghost_negative++;
            current_ghost_positive++;
        }
    }
}

TEST_CASE_METHOD(Array3DWithGhostsTest, "Array3DWithGhosts::updateBoundaries populates boundaries correctly")
{
    // Verify boundaries do not contain the expected values
    Slab slab = my_array_with_ghosts->cellSlab();
    int i0 = slab.start(0), j0 = slab.start(1), k0 = slab.start(2);
    int in = slab.end(0), jn = slab.end(1), kn = slab.end(2);
    Array2D<double> i_negative_boundary = my_array_with_ghosts->boundary(0, Direction::NEGATIVE);
    Array2D<double> i_positive_boundary = my_array_with_ghosts->boundary(0, Direction::POSITIVE);
    Array2D<double> j_negative_boundary = my_array_with_ghosts->boundary(1, Direction::NEGATIVE);
    Array2D<double> j_positive_boundary = my_array_with_ghosts->boundary(1, Direction::POSITIVE);
    Array2D<double> k_negative_boundary = my_array_with_ghosts->boundary(2, Direction::NEGATIVE);
    Array2D<double> k_positive_boundary = my_array_with_ghosts->boundary(2, Direction::POSITIVE);

    // axis i
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE(i_negative_boundary(j, k) != (*my_array_with_ghosts)(i0, j, k));
            REQUIRE(i_positive_boundary(j, k) != (*my_array_with_ghosts)(in - 1, j, k));
        }
    }
    // axis j
    for (int k = k0; k < kn; k++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(j_negative_boundary(i, k) != (*my_array_with_ghosts)(i, j0, k));
            REQUIRE(j_positive_boundary(i, k) != (*my_array_with_ghosts)(i, jn - 1, k));
        }
    }
    // axis k
    for (int j = j0; j < jn; j++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(k_negative_boundary(i, j) != (*my_array_with_ghosts)(i, j, k0));
            REQUIRE(k_positive_boundary(i, j) != (*my_array_with_ghosts)(i, j, kn - 1));
        }
    }

    // Update boundaries
    callUpdateBoundaries();

    // Verify boundaries contain the expected values
    // axis i
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE(i_negative_boundary(j, k) == (*my_array_with_ghosts)(i0, j, k));
            REQUIRE(i_positive_boundary(j, k) == (*my_array_with_ghosts)(in - 1, j, k));
        }
    }
    // axis j
    for (int k = k0; k < kn; k++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(j_negative_boundary(i, k) == (*my_array_with_ghosts)(i, j0, k));
            REQUIRE(j_positive_boundary(i, k) == (*my_array_with_ghosts)(i, jn - 1, k));
        }
    }
    // axis k
    for (int j = j0; j < jn; j++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(k_negative_boundary(i, j) == (*my_array_with_ghosts)(i, j, k0));
            REQUIRE(k_positive_boundary(i, j) == (*my_array_with_ghosts)(i, j, kn - 1));
        }
    }
}

TEST_CASE_METHOD(Array3DWithGhostsTest, "Array3DWithGhosts::exchangeGhosts")
{
    // Define a small problem
    unsigned int n = 30;
    size_t size    = n * n * n;

    // Allocate and initialise the data
    int my_rank = myRank(m_cartesian_communicator);
    std::vector<double> input_1d_vector(size);
    for (int idx = 0; idx < size; idx++)
        input_1d_vector[idx] = my_rank;
    Array3D<double> arr(n, n, n, input_1d_vector);
    GhostExchangePlan<double> plan(*my_array, m_local_boundary_conditions, m_neighbours, m_cartesian_communicator);
    Array3DWithGhosts<double> arr_with_ghosts(plan);

    // Verify ghosts do not contain the expected values
    Slab slab = my_array_with_ghosts->cellSlab();
    int i0 = slab.start(0), j0 = slab.start(1), k0 = slab.start(2);
    int in = slab.end(0), jn = slab.end(1), kn = slab.end(2);
    Array2D<double> i_negative_ghost = my_array_with_ghosts->ghost(0, Direction::NEGATIVE);
    Array2D<double> i_positive_ghost = my_array_with_ghosts->ghost(0, Direction::POSITIVE);
    Array2D<double> j_negative_ghost = my_array_with_ghosts->ghost(1, Direction::NEGATIVE);
    Array2D<double> j_positive_ghost = my_array_with_ghosts->ghost(1, Direction::POSITIVE);
    Array2D<double> k_negative_ghost = my_array_with_ghosts->ghost(2, Direction::NEGATIVE);
    Array2D<double> k_positive_ghost = my_array_with_ghosts->ghost(2, Direction::POSITIVE);

    // axis i
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE(i_negative_ghost(j, k) != m_neighbours[0]);
            REQUIRE(i_positive_ghost(j, k) != m_neighbours[1]);
        }
    }
    // axis j
    for (int k = k0; k < kn; k++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(j_negative_ghost(i, k) != m_neighbours[2]);
            REQUIRE(j_positive_ghost(i, k) != m_neighbours[3]);
        }
    }
    // axis k
    for (int j = j0; j < jn; j++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(k_negative_ghost(i, j) != m_neighbours[4]);
            REQUIRE(k_positive_ghost(i, j) != m_neighbours[5]);
        }
    }
    arr_with_ghosts.exchangeGhosts();

    // axis i
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE(i_negative_ghost(j, k) == m_neighbours[0]);
            REQUIRE(i_positive_ghost(j, k) == m_neighbours[1]);
        }
    }
    // axis j
    for (int k = k0; k < kn; k++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(j_negative_ghost(i, k) == m_neighbours[2]);
            REQUIRE(j_positive_ghost(i, k) == m_neighbours[3]);
        }
    }
    // axis k
    for (int j = j0; j < jn; j++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(k_negative_ghost(i, j) == m_neighbours[4]);
            REQUIRE(k_positive_ghost(i, j) == m_neighbours[5]);
        }
    }

    REQUIRE(true);
}
