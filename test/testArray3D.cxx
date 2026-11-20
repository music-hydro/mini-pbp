#include "Array3D.h"
#include "fixtures/Array3DTest.cxx"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("Array3D constructor errors for incorrectly sized inputs")
{
    unsigned int n = 3;
    std::vector<double> x(3);
    REQUIRE_THROWS_WITH(Array3D(n, n, n, x),
                        Catch::Matchers::ContainsSubstring("Size of data_1D is not equal to (ni * nj * nk)"));
}

TEST_CASE("Array3D::subArray errors for an invalid sub slab")
{
    unsigned int n = 3;
    size_t size    = n * n * n;
    std::vector<double> x_raw(size);
    Array3D<double> x(n, n, n, x_raw);

    Slab sub_slab({1, 1, 1}, {10, 10, 10});

    REQUIRE_THROWS_WITH(x.subArray(sub_slab),
                        Catch::Matchers::ContainsSubstring("The requested sub slab is not within the currently defined slab."));
}

TEST_CASE("Array3D::subArray preserves memory locations")
{
    unsigned int n = 5;
    size_t size    = n * n * n;
    std::vector<double> x_raw(size);
    Array3D<double> x(n, n, n, x_raw);

    Slab sub_slab({1, 1, 1}, {3, 3, 3});
    Array3D<double> y = x.subArray(sub_slab);

    for (int k = sub_slab.start(2); k < sub_slab.end(2); k++)
    {
        for (int j = sub_slab.start(1); j < sub_slab.end(1); j++)
        {
            for (int i = sub_slab.start(0); i < sub_slab.end(0); i++)
            {
                REQUIRE(y(i, j, k) == x(i, j, k));
                REQUIRE(&y(i, j, k) == &x(i, j, k));
            }
        }
    }
}

TEST_CASE("Array3D constructors correctly set m_is_sub_array")
{
    unsigned int n = 10;
    size_t size    = n * n * n;
    std::vector<double> raw(size);

    Array3D<double> a(n, n, n);
    Array3D<double> b(n, n, n, &raw[0]);
    Array3D<double> c(n, n, n, raw);
    Array3D<double> d(a);
    Array3D<double> sub_array(a, Slab({1, 4, 3}, {4, 7, 9}));

    REQUIRE(!a.isSubArray());
    REQUIRE(!b.isSubArray());
    REQUIRE(!c.isSubArray());
    REQUIRE(!d.isSubArray());
    REQUIRE(sub_array.isSubArray());
}

TEST_CASE("Array3D constructors preserve memory locations")
{
    unsigned int n = 3;
    size_t size    = n * n * n;
    std::vector<double> x_raw(size);
    Array3D<double> z(n, n, n, &x_raw[0]);
    Array3D<double> x(n, n, n, x_raw);
    Array3D<double> y(x);

    unsigned int idx = 0;
    for (unsigned int k = 0; k < n; k++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            for (unsigned int i = 0; i < n; i++)
            {
                REQUIRE(y(i, j, k) == x(i, j, k));
                REQUIRE(&y(i, j, k) == &x_raw[idx]);
                REQUIRE(&x(i, j, k) == &x_raw[idx]);
                REQUIRE(&z(i, j, k) == &x_raw[idx]);
                idx++;
            }
        }
    }
}

TEST_CASE("Array3D(i,j,k) accesses contiguously")
{

    // Define the problem starting from an arbitrary point along the array
    int ni_full = 30, ni_view = 20;
    int nj_full = 40, nj_view = 20;
    int nk_full = 30, nk_view = 30;
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

    Array3D<double> x(n, n, n, x_raw);
    Array3D<double> y(n, n, n, y_raw);

    for (unsigned int i = 0; i < size; i++)
    {
        x_raw[i] = 1;
        y_raw[i] = 2;
    }

    x.deepCopy(y);

    int counter = 0;
    for (unsigned int k = 0; k < n; k++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            for (unsigned int i = 0; i < n; i++)
            {
                REQUIRE(y(i, j, k) == x(i, j, k));
                REQUIRE(&y(i, j, k) == &y_raw[counter]);
                REQUIRE(&y(i, j, k) != &x_raw[counter]);
                counter++;
            }
        }
    }
}

TEST_CASE("Array3D::operator()(i,j,k) throws for out of range")
{
    // Define the problem
    unsigned int n = 10;
    size_t size    = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d(size);

    Array3D<double> full_array(n, n, n, input_1d);
    Array3D<double> a(full_array, Slab({2, 2, 2}, {8, 8, 8}));

    // Negative
    REQUIRE_THROWS_WITH(a(-4, 3, 3), Catch::Matchers::ContainsSubstring("Requested cell out of slab range for axis 0"));
    REQUIRE_THROWS_WITH(a(3, -7, 3), Catch::Matchers::ContainsSubstring("Requested cell out of slab range for axis 1"));
    REQUIRE_THROWS_WITH(a(3, 3, -2), Catch::Matchers::ContainsSubstring("Requested cell out of slab range for axis 2"));

    // Positive less than slab
    REQUIRE_THROWS_WITH(a(1, 3, 3), Catch::Matchers::ContainsSubstring("Requested cell out of slab range for axis 0"));
    REQUIRE_THROWS_WITH(a(3, 1, 3), Catch::Matchers::ContainsSubstring("Requested cell out of slab range for axis 1"));
    REQUIRE_THROWS_WITH(a(3, 3, 1), Catch::Matchers::ContainsSubstring("Requested cell out of slab range for axis 2"));

    // Positive more than slab
    REQUIRE_THROWS_WITH(a(100, 3, 3), Catch::Matchers::ContainsSubstring("Requested cell out of slab range for axis 0"));
    REQUIRE_THROWS_WITH(a(3, 45, 3), Catch::Matchers::ContainsSubstring("Requested cell out of slab range for axis 1"));
    REQUIRE_THROWS_WITH(a(3, 3, 22), Catch::Matchers::ContainsSubstring("Requested cell out of slab range for axis 2"));
}

TEST_CASE("Array3D::fillBoundaryPlane correctly fills boundary planes")
{
    // Define the problem
    unsigned int n = 10;
    size_t size    = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d(size);

    Array3D<double> array_3d(n, n, n, input_1d);
    array_3d.fill(100.0);
    Array3DTest::requireAll(array_3d, 100.0);

    for (unsigned int axis = 0; axis < 3; axis++)
    {
        array_3d.fillBoundaryPlane(axis, Direction::NEGATIVE, axis * 10.0 + (double)Direction::NEGATIVE);
        array_3d.fillBoundaryPlane(axis, Direction::POSITIVE, axis * 10.0 + (double)Direction::POSITIVE);

        for (int idx2 = 0; idx2 < n; idx2++)
        {
            for (int idx1 = 0; idx1 < n; idx1++)
            {
                if (axis == 0)
                {
                    // axis i
                    REQUIRE(array_3d(0, idx1, idx2) == 0.0);
                    REQUIRE(array_3d(n - 1, idx1, idx2) == 1.0);
                }
                else if (axis == 1)
                {
                    // axis j
                    REQUIRE(array_3d(idx1, 0, idx2) == 10.0);
                    REQUIRE(array_3d(idx1, n - 1, idx2) == 11.0);
                }
                else
                {
                    // axis k
                    REQUIRE(array_3d(idx1, idx2, 0) == 20.0);
                    REQUIRE(array_3d(idx1, idx2, n - 1) == 21.0);
                }
            }
        }
    }
}
