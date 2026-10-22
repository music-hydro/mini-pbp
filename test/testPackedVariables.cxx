#include "PackedVariables.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("PackedVariables::singleVariableView stores contiguously without copies")
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
    PackedVariables packed_variables(input_1d_pointer, ni, nj, nk);

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
                    REQUIRE(variable_array(i, j, k) == input_1d_pointer[counter + offset]);
                    REQUIRE(&variable_array(i, j, k) == &input_1d_pointer[counter + offset]);
                    counter++;
                }
            }
        }
    }
}

TEST_CASE("PackedVariables::PackedVariables fails for incorrect array size")
{
    int n = 2;
    std::vector<double> input_1d_pointer(2);
    REQUIRE_THROWS_WITH(PackedVariables(input_1d_pointer, n, n, n),
                        Catch::Matchers::ContainsSubstring("Size of packed_array_pointer is not a multiple of (ni * nj * nk)"));
}

TEST_CASE("PackedVariables::singleVariableView fails for invalid variable_index")
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
    PackedVariables packed_variables(input_1d_pointer, ni, nj, nk);

    REQUIRE_THROWS_WITH(packed_variables.singleVariableView(12),
                        Catch::Matchers::ContainsSubstring("Provided variable index not found:"));
}
