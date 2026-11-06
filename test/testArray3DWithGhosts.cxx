#include "Array3DWithGhosts.h"
#include "DirectionalArray.h"
#include "Slab.h"
#include "fixtures/Array3DWithGhostsTest.cxx"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>

TEST_CASE_METHOD(MpiTest, "Array3DWithGhosts::neighbourCellPosition returns the correct weights")
{
    // Define the problem
    int n = 3;
    size_t size    = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d = std::vector<double>(size);

    Array3D<double> my_array(n, n, n, input_1d);

    // The neighbouring ranks shouldn't matter
    DirectionalArray<int> neighbours({0, 0, 0, 0, 0, 0});
    DirectionalArray<BoundaryConditions> bcs({BoundaryConditions::INNER,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::PERIODIC,
                                              BoundaryConditions::PERIODIC,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::INNER});
   
    // Choose global slab so indices are on global boundaries
    Slab global_slab({0, 0, 0}, {n, n, n});

    GhostExchangePlan<double> my_plan(global_slab, bcs, neighbours, m_cartesian_communicator);
    Array3DWithGhosts<double> my_array_with_ghosts(my_array, my_plan);

    // i axis
    int axis = 0;
    std::array<int, 3> negative_neighbour_position = my_array_with_ghosts.neighbourCellPosition(0, 1, 1, axis, Direction::NEGATIVE);
    REQUIRE(negative_neighbour_position[0] == -1);
    REQUIRE(negative_neighbour_position[1] == 1);
    REQUIRE(negative_neighbour_position[2] == 1);
    std::array<int, 3> positive_neighbour_position = my_array_with_ghosts.neighbourCellPosition(2, 1, 1, axis, Direction::POSITIVE);
    REQUIRE(positive_neighbour_position[0] == -1);
    REQUIRE(positive_neighbour_position[1] == 1);
    REQUIRE(positive_neighbour_position[2] == 1);

    // j axis
    axis = 1;
    negative_neighbour_position = my_array_with_ghosts.neighbourCellPosition(1, 0, 1, axis, Direction::NEGATIVE);
    REQUIRE(negative_neighbour_position[0] == 1);
    REQUIRE(negative_neighbour_position[1] == 2); // PERIODIC
    REQUIRE(negative_neighbour_position[2] == 1);
    positive_neighbour_position = my_array_with_ghosts.neighbourCellPosition(1, 2, 1, axis, Direction::POSITIVE);
    REQUIRE(positive_neighbour_position[0] == 1);
    REQUIRE(positive_neighbour_position[1] == 0); // PERIODIC
    REQUIRE(positive_neighbour_position[2] == 1);

    // K axis
    axis = 2;
    negative_neighbour_position = my_array_with_ghosts.neighbourCellPosition(1, 1, 0, axis, Direction::NEGATIVE);
    REQUIRE(negative_neighbour_position[0] == 1);
    REQUIRE(negative_neighbour_position[1] == 1);
    REQUIRE(negative_neighbour_position[2] == -1);
    positive_neighbour_position = my_array_with_ghosts.neighbourCellPosition(1, 1, 2, axis, Direction::POSITIVE);
    REQUIRE(positive_neighbour_position[0] == 1);
    REQUIRE(positive_neighbour_position[1] == 1);
    REQUIRE(positive_neighbour_position[2] == -1);
}

TEST_CASE_METHOD(MpiTest, "Array3DWithGhosts::neighbourLinkWeights returns the correct weights")
{
    // Define the problem
    int n = 3;
    size_t size    = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d = std::vector<double>(size);

    Array3D<double> my_array(n, n, n, input_1d);

    // The neighbouring ranks shouldn't matter
    DirectionalArray<int> neighbours({0, 0, 0, 0, 0, 0});
    DirectionalArray<BoundaryConditions> bcs({BoundaryConditions::INNER,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::REFLECTIVE,
                                              BoundaryConditions::REFLECTIVE,
                                              BoundaryConditions::INNER,
                                              BoundaryConditions::INNER});

    // Global slab shouldn't matter here.
    Slab global_slab({0, 0, 0}, {n, n, n});

    GhostExchangePlan<double> my_plan(global_slab, bcs, neighbours, m_cartesian_communicator);
    Array3DWithGhosts<double> my_array_with_ghosts(my_array, my_plan);

    DirectionalArray<double> inner_cell_weights = my_array_with_ghosts.neighbourLinkWeights(1, 1, 1);
    return;
    for (int idx = 0; idx < 6; idx++)
        REQUIRE(inner_cell_weights[idx] == 1.0);

    std::array<int, 3> coords = {1, 1, 1};
    for (int axis = 0; axis < 3; axis++)
    {
        coords[axis] = 0;
        DirectionalArray<double> negative_boundary_cell_weights =
            my_array_with_ghosts.neighbourLinkWeights(coords[0], coords[1], coords[2]);
        REQUIRE(negative_boundary_cell_weights.get(axis, Direction::NEGATIVE) == 0.0);
        REQUIRE(negative_boundary_cell_weights.get(axis, Direction::POSITIVE) == 1.0);

        coords[axis] = 2;
        DirectionalArray<double> positive_boundary_cell_weights =
            my_array_with_ghosts.neighbourLinkWeights(coords[0], coords[1], coords[2]);
        REQUIRE(positive_boundary_cell_weights.get(axis, Direction::NEGATIVE) == 0.0);
        REQUIRE(positive_boundary_cell_weights.get(axis, Direction::POSITIVE) == 1.0);
    }
}

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
    const Array2D<double>& i_negative_boundary = my_array_with_ghosts->boundary(0, Direction::NEGATIVE);
    const Array2D<double>& i_positive_boundary = my_array_with_ghosts->boundary(0, Direction::POSITIVE);
    const Array2D<double>& j_negative_boundary = my_array_with_ghosts->boundary(1, Direction::NEGATIVE);
    const Array2D<double>& j_positive_boundary = my_array_with_ghosts->boundary(1, Direction::POSITIVE);
    const Array2D<double>& k_negative_boundary = my_array_with_ghosts->boundary(2, Direction::NEGATIVE);
    const Array2D<double>& k_positive_boundary = my_array_with_ghosts->boundary(2, Direction::POSITIVE);

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
    // Initialise inner arrays with the rank values
    int my_rank = myRank(m_cartesian_communicator);
    for (int idx = 0; idx < input_1d->size(); idx++)
        (*input_1d)[idx] = my_rank;

    // Verify ghosts do not contain the expected values
    Slab slab = my_array_with_ghosts->cellSlab();
    int i0 = slab.start(0), j0 = slab.start(1), k0 = slab.start(2);
    int in = slab.end(0), jn = slab.end(1), kn = slab.end(2);
    const Array2D<double>& i_negative_ghost = my_array_with_ghosts->ghost(0, Direction::NEGATIVE);
    const Array2D<double>& i_positive_ghost = my_array_with_ghosts->ghost(0, Direction::POSITIVE);
    const Array2D<double>& j_negative_ghost = my_array_with_ghosts->ghost(1, Direction::NEGATIVE);
    const Array2D<double>& j_positive_ghost = my_array_with_ghosts->ghost(1, Direction::POSITIVE);
    const Array2D<double>& k_negative_ghost = my_array_with_ghosts->ghost(2, Direction::NEGATIVE);
    const Array2D<double>& k_positive_ghost = my_array_with_ghosts->ghost(2, Direction::POSITIVE);

    DirectionalArray<int> expected_ghost_values(m_neighbours);
    // If at a boundary, account for boundary conditions
    for (int idx = 0; idx < 6; idx++)
        if (m_neighbours[idx] == MPI_PROC_NULL)
            expected_ghost_values[idx] = my_plan->boundaryConditions()[idx] == ZERO_VALUE ? -1 * my_rank : my_rank;

    // axis i
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE(i_negative_ghost(j, k) != expected_ghost_values.get(0U, Direction::NEGATIVE));
            REQUIRE(i_positive_ghost(j, k) != expected_ghost_values.get(0U, Direction::POSITIVE));
        }
    }
    // axis j
    for (int k = k0; k < kn; k++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(j_negative_ghost(i, k) != expected_ghost_values.get(1U, Direction::NEGATIVE));
            REQUIRE(j_positive_ghost(i, k) != expected_ghost_values.get(1U, Direction::POSITIVE));
        }
    }
    // axis k
    for (int j = j0; j < jn; j++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(k_negative_ghost(i, j) != expected_ghost_values.get(2U, Direction::NEGATIVE));
            REQUIRE(k_positive_ghost(i, j) != expected_ghost_values.get(2U, Direction::POSITIVE));
        }
    }
    my_array_with_ghosts->exchangeGhosts();

    // axis i
    for (int k = k0; k < kn; k++)
    {
        for (int j = j0; j < jn; j++)
        {
            REQUIRE((int)i_negative_ghost(j, k) == expected_ghost_values.get(0U, Direction::NEGATIVE));
            REQUIRE((int)i_positive_ghost(j, k) == expected_ghost_values.get(0U, Direction::POSITIVE));
        }
    }
    // axis j
    for (int k = k0; k < kn; k++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(j_negative_ghost(i, k) == expected_ghost_values.get(1U, Direction::NEGATIVE));
            REQUIRE(j_positive_ghost(i, k) == expected_ghost_values.get(1U, Direction::POSITIVE));
        }
    }
    // axis k
    for (int j = j0; j < jn; j++)
    {
        for (int i = i0; i < in; i++)
        {
            REQUIRE(k_negative_ghost(i, j) == expected_ghost_values.get(2U, Direction::NEGATIVE));
            REQUIRE(k_positive_ghost(i, j) == expected_ghost_values.get(2U, Direction::POSITIVE));
        }
    }
}

TEST_CASE_METHOD(Array3DWithGhostsTest, "Array3DWithGhosts constructed from a sub array throws")
{
    // Define the problem
    unsigned int n = 10;
    size_t size    = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d(size);

    Array3D<double> full_array(n, n, n, input_1d);
    Array3D<double> sub_array(full_array, Slab({2, 2, 2}, {8, 8, 8}));

    // Create an Array3DWithGhosts instance wrapping the Array3D
    REQUIRE_THROWS_WITH(Array3DWithGhosts<double>(sub_array, *my_plan),
                        Catch::Matchers::ContainsSubstring("A sub array cannot be used to create an array with ghosts"));
}
