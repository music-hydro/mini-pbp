#include "PhysicsBasedPreconditioner.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("PhysicsBasedPreconditioner::setPressureRHS trivial zeros to check pointers remain unchanged")
{
    // Define the problem
    int num_variables = 2;
    int n             = 3;
    int size          = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    double* delta_prim_raw = new double[size * num_variables];
    double* dp_rhs_raw     = new double[size * num_variables];
    double* solution_raw   = new double[size * num_variables];

    // Initialise arrays
    for (int i = 0; i < size * num_variables; i++)
    {
        delta_prim_raw[i] = 0;
        dp_rhs_raw[i]     = 0;
        solution_raw[i]   = 0;
    }

    // Create Packed variable arrays
    PackedVariables delta_prim_view(delta_prim_raw, n, n, n, num_variables);
    PackedVariables dp_rhs_view(dp_rhs_raw, n, n, n, num_variables);
    PackedVariables solution(solution_raw, n, n, n, num_variables);

    PhysicsBasedPreconditioner pbp(solution, 0, 1, 0, 1, 0, 1);

    REQUIRE_NOTHROW(pbp.setPressureRHS(delta_prim_view, dp_rhs_view));

    PackedVariables new_solution = pbp.solution();

    for (int var = 0; var < num_variables; var++)
    {
        Array3D<double> delta_prim_var_array   = delta_prim_view.singleVariableView(var);
        Array3D<double> dp_rhs_var_array       = dp_rhs_view.singleVariableView(var);
        Array3D<double> solution_var_array     = solution.singleVariableView(var);
        Array3D<double> new_solution_var_array = new_solution.singleVariableView(var);

        int offset = var * size;

        // Verify that looping through row-major i,j,k remains contiguous.
        int counter = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    // delta_prim
                    CHECK(delta_prim_var_array(i, j, k) == delta_prim_raw[counter + offset]);
                    REQUIRE(&delta_prim_var_array(i, j, k) == &delta_prim_raw[counter + offset]);

                    // dp_rhs
                    CHECK(dp_rhs_var_array(i, j, k) == dp_rhs_raw[counter + offset]);
                    REQUIRE(&dp_rhs_var_array(i, j, k) == &dp_rhs_raw[counter + offset]);

                    // solution
                    CHECK(solution_var_array(i, j, k) == solution_raw[counter + offset]);
                    REQUIRE(&solution_var_array(i, j, k) == &solution_raw[counter + offset]);
                    CHECK(solution_var_array(i, j, k) == new_solution_var_array(i, j, k));
                    REQUIRE(&solution_var_array(i, j, k) == &new_solution_var_array(i, j, k));

                    counter++;
                }
            }
        }
    }
}

void verifySetPressureRHS(int prec_nvar, double* expected_energy)
{
    const int num_variables = 2;
    const int n             = 3;
    const int size          = n * n * n;

    double delta_prim_raw[size * num_variables] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                   19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                                   37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};

    double solution_raw[size * num_variables] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};

    double expected_pressure[size] = {-86,  -91,  -96,  -101, -106, -111, -116, -121, -126, -131, -136, -141, -146, -151,
                                      -156, -161, -166, -171, -176, -181, -186, -191, -196, -201, -206, -211, -216};

    double dp_rhs_raw[size * num_variables] = {};

    PackedVariables delta_prim_view(delta_prim_raw, n, n, n, num_variables);
    PackedVariables dp_rhs_view(dp_rhs_raw, n, n, n, num_variables);
    PackedVariables solution(solution_raw, n, n, n, num_variables);

    Array3D<double> actual_pressure = dp_rhs_view.singleVariableView(0);
    Array3D<double> actual_energy   = dp_rhs_view.singleVariableView(1);

    PhysicsBasedPreconditioner pbp(solution, 0, 1, 0, 1, 0, 1, prec_nvar);


    pbp.setPressureRHS(delta_prim_view, dp_rhs_view);

    // Verify that looping through row-major i,j,k remains contiguous.
    int counter = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                CHECK(actual_pressure(i, j, k) == expected_pressure[counter]);
                REQUIRE(actual_energy(i, j, k) == expected_energy[counter]);
                counter++;
            }
        }
    }
}

TEST_CASE("PhysicsBasedPreconditioner::setPressureRHS non-trivial matrices", "[prec_nvar]")
{
    SECTION("prec_nvar == 1")
    {
        double expected_energy[27];
        for (int i = 0; i < 27; i++)
            expected_energy[i] = 0;
        verifySetPressureRHS(1, expected_energy);
    }

    SECTION("prec_nvar == 2")
    {
        double expected_energy[27] = {-28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41,
                                      -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54};
        verifySetPressureRHS(2, expected_energy);
    }
}
