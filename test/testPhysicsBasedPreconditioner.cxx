#include "PhysicsBasedPreconditioner.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <stdio.h>
#include <vector>

double tolerance = 1e-14;

TEST_CASE("PhysicsBasedPreconditioner::setPressureRHS trivial zeros to check pointers remain unchanged")
{
    // Define the problem
    int num_variables = 2;
    int n             = 3;
    int size          = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> delta_prim_raw(size * num_variables);
    std::vector<double> dp_rhs_raw(size * num_variables);
    std::vector<double> solution_raw(size * num_variables);

    // Initialise arrays
    for (int i = 0; i < size * num_variables; i++)
    {
        delta_prim_raw[i] = 0;
        dp_rhs_raw[i]     = 0;
        solution_raw[i]   = 0;
    }

    // Create Packed variable arrays
    PackedVariables delta_prim_view(&delta_prim_raw[0], n, n, n, num_variables);
    PackedVariables dp_rhs_view(&dp_rhs_raw[0], n, n, n, num_variables);
    PackedVariables solution(&solution_raw[0], n, n, n, num_variables);

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
        for (int k = 0; k < n; k++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    // delta_prim
                    REQUIRE(delta_prim_var_array(i, j, k) == delta_prim_raw[counter + offset]);
                    REQUIRE(&delta_prim_var_array(i, j, k) == &delta_prim_raw[counter + offset]);

                    // dp_rhs
                    REQUIRE(dp_rhs_var_array(i, j, k) == dp_rhs_raw[counter + offset]);
                    REQUIRE(&dp_rhs_var_array(i, j, k) == &dp_rhs_raw[counter + offset]);

                    // solution
                    REQUIRE(solution_var_array(i, j, k) == solution_raw[counter + offset]);
                    REQUIRE(&solution_var_array(i, j, k) == &solution_raw[counter + offset]);
                    REQUIRE(solution_var_array(i, j, k) == new_solution_var_array(i, j, k));
                    REQUIRE(&solution_var_array(i, j, k) == &new_solution_var_array(i, j, k));

                    counter++;
                }
            }
        }
    }
}

/**
 * @brief Convenience function to test PhysicsBasedPreconditioner::setPressureRHS correctly sets the pressure and energy.
 */
void verifySetPressureRHS(int ni,
                          int nj,
                          int nk,
                          PackedVariables delta_prim_view,
                          PackedVariables dp_rhs_view,
                          PackedVariables solution,
                          double* dp_rhs_raw,
                          double* expected_pressure,
                          double* expected_energy,
                          int prec_nvar,
                          int n_rho,
                          int n_e,
                          int n_dPdrho,
                          int n_dPde,
                          int n_pressure,
                          int n_energy)
{
    const int size = ni * nj * nk;

    Array3D<double> actual_pressure = dp_rhs_view.singleVariableView(0);
    Array3D<double> actual_energy   = dp_rhs_view.singleVariableView(1);

    PhysicsBasedPreconditioner pbp(solution, n_rho, n_e, n_dPdrho, n_dPde, n_pressure, n_energy, prec_nvar);

    pbp.setPressureRHS(delta_prim_view, dp_rhs_view);

    // Verify that looping through row-major i,j,k remains contiguous.
    int counter = 0;
    for (int k = 0; k < nk; k++)
    {
        for (int j = 0; j < nj; j++)
        {
            for (int i = 0; i < ni; i++)
            {
                REQUIRE_THAT(actual_pressure(i, j, k), Catch::Matchers::WithinRel(expected_pressure[counter], tolerance));
                REQUIRE(&actual_pressure(i, j, k) == &dp_rhs_raw[counter]);
                REQUIRE_THAT(actual_energy(i, j, k), Catch::Matchers::WithinRel(expected_energy[counter], tolerance));
                REQUIRE(&actual_energy(i, j, k) == &dp_rhs_raw[counter + size]);
                counter++;
            }
        }
    }
}

TEST_CASE("PhysicsBasedPreconditioner::setPressureRHS prec_nvar updates energy", "[prec_nvar]")
{
    const int num_variables = 2;
    const int n             = 3;
    const int size          = n * n * n;

    std::vector<double> delta_prim_raw(size * num_variables);
    delta_prim_raw = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                      28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};
    std::vector<double> solution_raw(size * num_variables);
    solution_raw = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    std::vector<double> expected_pressure(size);
    expected_pressure = {-86,  -91,  -96,  -101, -106, -111, -116, -121, -126, -131, -136, -141, -146, -151,
                         -156, -161, -166, -171, -176, -181, -186, -191, -196, -201, -206, -211, -216};
    std::vector<double> dp_rhs_raw(size * num_variables);

    PackedVariables delta_prim_view(&delta_prim_raw[0], n, n, n, num_variables);
    PackedVariables dp_rhs_view(&dp_rhs_raw[0], n, n, n, num_variables);
    PackedVariables solution(&solution_raw[0], n, n, n, num_variables);

    int n_rho      = 0;
    int n_e        = 1;
    int n_dPdrho   = 0;
    int n_dPde     = 1;
    int n_pressure = 0;
    int n_energy   = 1;

    SECTION("prec_nvar == 1")
    {
        double expected_energy[27];
        for (int i = 0; i < 27; i++)
            expected_energy[i] = 0;
        verifySetPressureRHS(n,
                             n,
                             n,
                             delta_prim_view,
                             dp_rhs_view,
                             solution,
                             &dp_rhs_raw[0],
                             &expected_pressure[0],
                             &expected_energy[0],
                             1,
                             n_rho,
                             n_e,
                             n_dPdrho,
                             n_dPde,
                             n_pressure,
                             n_energy);
    }

    SECTION("prec_nvar == 2")
    {
        double expected_energy[27] = {-28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41,
                                      -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54};
        verifySetPressureRHS(n,
                             n,
                             n,
                             delta_prim_view,
                             dp_rhs_view,
                             solution,
                             &dp_rhs_raw[0],
                             &expected_pressure[0],
                             &expected_energy[0],
                             2,
                             n_rho,
                             n_e,
                             n_dPdrho,
                             n_dPde,
                             n_pressure,
                             n_energy);
    }
}

TEST_CASE("PhysicsBasedPreconditioner::setPressureRHS roberts dynamo problem", "[prec_nvar]")
{
    const int num_variables = 2;
    const int ni            = 32;
    const int nj            = 32;
    const int nk            = 64;
    const int size          = ni * nj * nk;

    std::vector<double> delta_prim_raw(size * num_variables);
    std::vector<double> solution_raw(size * num_variables);
    std::vector<double> expected_pressure(size);

    // Energy will remain unchanged in this example as we are using `prec_nvar == 1`
    // but we set to zeros to help catch if we break something in the future.
    std::vector<double> expected_energy(size);
    for (int i = 0; i < size; i++)
        expected_energy[i] = 0;

    std::vector<double> dp_rhs_raw(size * num_variables);

    int n_rho      = 0;
    int n_e        = 1;
    int n_dPdrho   = 0;
    int n_dPde     = 1;
    int n_pressure = 0;
    int n_energy   = 1;

    // Read input data from file
    FILE* inputs_file = fopen(TEST_DATA_DIR "/set_pressure_rhs_roberts_dynamo_inputs.out", "r");
    REQUIRE(inputs_file != nullptr);
    for (int i = 0; i < size; i++)
    {
        // Reads each row of the input file ensuring there are 4 columns in each
        REQUIRE(fscanf(inputs_file,
                       "%lf %lf %lf %lf",
                       &solution_raw[i + n_dPdrho * size],
                       &delta_prim_raw[i + n_rho * size],
                       &solution_raw[i + n_dPde * size],
                       &delta_prim_raw[i + n_e * size]) == 4);
    }
    fclose(inputs_file);

    // Read expected output from file
    FILE* outputs_file = fopen(TEST_DATA_DIR "/set_pressure_rhs_roberts_dynamo_expected_outputs.out", "r");
    REQUIRE(outputs_file != nullptr);
    for (int i = 0; i < size; i++)
    {
        REQUIRE(fscanf(outputs_file, "%lf", &expected_pressure[i]) == 1);
    }
    fclose(outputs_file);

    PackedVariables delta_prim_view(&delta_prim_raw[0], ni, nj, nk, num_variables);
    PackedVariables dp_rhs_view(&dp_rhs_raw[0], ni, nj, nk, num_variables);
    PackedVariables solution(&solution_raw[0], ni, nj, nk, num_variables);

    verifySetPressureRHS(ni,
                         nj,
                         nk,
                         delta_prim_view,
                         dp_rhs_view,
                         solution,
                         &dp_rhs_raw[0],
                         &expected_pressure[0],
                         &expected_energy[0],
                         1,
                         n_rho,
                         n_e,
                         n_dPdrho,
                         n_dPde,
                         n_pressure,
                         n_energy);
}
