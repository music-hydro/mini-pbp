#include "fixtures/GhostExchangePlanTest.cxx"
#include "PackedVariables.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <functional>

template<class T>
void verifySingleVariableViewIsContiguous(std::function<T(std::vector<double>&, int, int, int)> packed_variable_constructor)
{
    // Define the problem
    int num_variables = 6;
    int ni            = 2;
    int nj            = 2;
    int nk            = 3;
    int size          = ni * nj * nk;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d_pointer(size * num_variables);

    // Populate the 1D array with the variable index
    for (int i = 0; i < num_variables; i++)
    {
        int offset = size * i;
        for (int j = 0; j < size; j++)
            input_1d_pointer[i + offset] = (double)i;
    }

    // Create a PackedVariables instance wrapping the 1D array
    T packed_variables = packed_variable_constructor(input_1d_pointer, ni, nj, nk);

    for (int variable_index = 0; variable_index < num_variables; variable_index++)
    {
        Array3D<double> variable_array = packed_variables.singleVariableView(variable_index);

        int offset = variable_index * size;

        // Verify that looping through row-major i,j,k remains contiguous.
        int counter = 0;
        for (int k = 0; k < nk; k++)
        {
            for (int j = 0; j < nj; j++)
            {
                for (int i = 0; i < ni; i++)
                {
                    REQUIRE(&variable_array(i, j, k) == &input_1d_pointer[counter + offset]);
                    REQUIRE_THAT(variable_array(i, j, k), Catch::Matchers::WithinRel(input_1d_pointer[counter + offset], 1e-14));
                    counter++;
                }
            }
        }
    }
}

TEST_CASE("PackedVariables::singleVariableView stores contiguously without copies")
{
    auto packed_variable_constructor = [](std::vector<double>& input_array, int ni, int nj, int nk)
    { return PackedVariables(input_array, ni, nj, nk); };
    verifySingleVariableViewIsContiguous<PackedVariables>(packed_variable_constructor);
}

TEST_CASE("PackedVariablesWithGhosts::singleVariableView stores contiguously without copies")
{
    auto packed_variable_constructor = [](std::vector<double>& input_array, int ni, int nj, int nk)
    { return PackedVariablesWithGhosts(input_array, ni, nj, nk, GhostExchangePlanTest::arbitraryPlan()); };
    verifySingleVariableViewIsContiguous<PackedVariablesWithGhosts>(packed_variable_constructor);
}

template<class T>
void verifyPackedVariablesFailsForIncorrectSize(std::function<T(std::vector<double>&, int, int, int)> packed_variable_constructor)
{
    int n = 2;
    std::vector<double> input_1d_pointer(2);
    REQUIRE_THROWS_WITH(packed_variable_constructor(input_1d_pointer, n, n, n),
                        Catch::Matchers::ContainsSubstring("Size of packed_array_pointer is not a multiple of (ni * nj * nk)"));
}

TEST_CASE("PackedVariables::PackedVariables fails for incorrect array size")
{
    auto packed_variable_constructor = [](std::vector<double>& input_array, int ni, int nj, int nk)
    { return PackedVariables(input_array, ni, nj, nk); };
    verifyPackedVariablesFailsForIncorrectSize<PackedVariables>(packed_variable_constructor);
}

TEST_CASE("PackedVariablesWithGhosts::PackedVariables fails for incorrect array size")
{
    auto packed_variable_constructor = [](std::vector<double>& input_array, int ni, int nj, int nk)
    { return PackedVariablesWithGhosts(input_array, ni, nj, nk, GhostExchangePlanTest::arbitraryPlan()); };
    verifyPackedVariablesFailsForIncorrectSize<PackedVariablesWithGhosts>(packed_variable_constructor);
}

template<class T>
void verifySingleVariableViewFailsInvalidIndex(std::function<T(std::vector<double>&, int, int, int)> packed_variable_constructor)
{
    // Define the problem
    int num_variables = 6;
    int ni            = 2;
    int nj            = 2;
    int nk            = 3;
    int size          = ni * nj * nk;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> input_1d_pointer(size * num_variables);

    // Create a PackedVariables instance wrapping the 1D array
    T packed_variables = packed_variable_constructor(input_1d_pointer, ni, nj, nk);

    REQUIRE_THROWS_WITH(packed_variables.singleVariableView(12),
                        Catch::Matchers::ContainsSubstring("Provided variable index not found:"));
}

TEST_CASE("PackedVariables::singleVariableView fails for invalid variable_index")
{
    auto packed_variable_constructor = [](std::vector<double>& input_array, int ni, int nj, int nk)
    { return PackedVariables(input_array, ni, nj, nk); };
    verifySingleVariableViewFailsInvalidIndex<PackedVariables>(packed_variable_constructor);
}

TEST_CASE("PackedVariablesWithGhosts::singleVariableView fails for invalid variable_index")
{
    auto packed_variable_constructor = [](std::vector<double>& input_array, int ni, int nj, int nk)
    { return PackedVariablesWithGhosts(input_array, ni, nj, nk, GhostExchangePlanTest::arbitraryPlan()); };
    verifySingleVariableViewFailsInvalidIndex<PackedVariablesWithGhosts>(packed_variable_constructor);
}
