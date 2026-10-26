#include "Array3D.h"
#include "fixtures.cxx"
#include "PhysicsBasedPreconditioner.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <stdio.h>
#include <vector>

TEST_CASE_METHOD(PhysicsBasedPreconditionerTest,
                 "PhysicsBasedPreconditioner::setPressureRHS trivial zeros to check pointers remain unchanged")
{
    // Define the problem
    int n    = 3;
    int size = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> delta_rho_raw(size);
    std::vector<double> delta_energy_raw(size);
    std::vector<double> rhs_pressure_raw(size);
    std::vector<double> rhs_energy_raw(size);

    alloc_solution(n, n, n);

    // Initialise arrays
    for (int i = 0; i < size; i++)
    {
        delta_rho_raw[i]    = 0;
        delta_energy_raw[i] = 0;
        rhs_pressure_raw[i] = 0;
        rhs_energy_raw[i]   = 0;
    }

    // Create Packed variable arrays
    Array3D<double> delta_rho(&delta_rho_raw[0], n, n, n);
    Array3D<double> delta_energy(&delta_energy_raw[0], n, n, n);
    Array3D<double> rhs_pressure(&rhs_pressure_raw[0], n, n, n);
    Array3D<double> rhs_energy(&rhs_energy_raw[0], n, n, n);

    PhysicsBasedPreconditioner pbp(*solution, *solution_dPd);

    REQUIRE_NOTHROW(pbp.setPressureRHS(delta_rho, delta_energy, rhs_pressure, rhs_energy));

    int n_dPdrho    = pbp.n_dPdrho();
    int n_dPdenergy = pbp.n_dPdenergy();

    Array3D<double> solution_dPdrho    = (*solution_dPd).singleVariableView(n_dPdrho);
    Array3D<double> solution_dPdenergy = (*solution_dPd).singleVariableView(n_dPdenergy);

    // Verify that looping through row-major i,j,k remains contiguous.
    int counter = 0;
    for (int k = 0; k < n; k++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                // delta_rho
                REQUIRE(delta_rho(i, j, k) == delta_rho_raw[counter]);
                REQUIRE(&delta_rho(i, j, k) == &delta_rho_raw[counter]);
                // delta_e
                REQUIRE(delta_energy(i, j, k) == delta_energy_raw[counter]);
                REQUIRE(&delta_energy(i, j, k) == &delta_energy_raw[counter]);

                // rhs
                REQUIRE(rhs_pressure(i, j, k) == rhs_pressure_raw[counter]);
                REQUIRE(&rhs_pressure(i, j, k) == &rhs_pressure_raw[counter]);
                REQUIRE(rhs_energy(i, j, k) == rhs_energy_raw[counter]);
                REQUIRE(&rhs_energy(i, j, k) == &rhs_energy_raw[counter]);

                // solution
                REQUIRE(solution_dPdrho(i, j, k) == (*solution_dPd_raw)[counter + size * n_dPdrho]);
                REQUIRE(&solution_dPdrho(i, j, k) == &(*solution_dPd_raw)[counter + size * n_dPdrho]);
                REQUIRE(solution_dPdenergy(i, j, k) == (*solution_dPd_raw)[counter + size * n_dPdenergy]);
                REQUIRE(&solution_dPdenergy(i, j, k) == &(*solution_dPd_raw)[counter + size * n_dPdenergy]);

                counter++;
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
                          const Array3D<double> delta_rho,
                          const Array3D<double> delta_energy,
                          Array3D<double> rhs_pressure,
                          Array3D<double> rhs_energy,
                          double* expected_pressure,
                          double* expected_energy,
                          PhysicsBasedPreconditioner pbp)
{
    REQUIRE_NOTHROW(pbp.setPressureRHS(delta_rho, delta_energy, rhs_pressure, rhs_energy));

    // Verify that looping through row-major i,j,k remains contiguous.
    int counter = 0;
    for (int k = 0; k < nk; k++)
    {
        for (int j = 0; j < nj; j++)
        {
            for (int i = 0; i < ni; i++)
            {
                REQUIRE_THAT(rhs_pressure(i, j, k), Catch::Matchers::WithinRel(expected_pressure[counter], 1e-14));
                REQUIRE_THAT(rhs_energy(i, j, k), Catch::Matchers::WithinRel(expected_energy[counter], 1e-14));
                counter++;
            }
        }
    }
}

TEST_CASE_METHOD(PhysicsBasedPreconditionerTest,
                 "PhysicsBasedPreconditioner::setPressureRHS prec_nvar updates energy",
                 "[prec_nvar]")
{
    const int n    = 3;
    const int size = n * n * n;

    alloc_solution(n, n, n);

    std::vector<double> delta_rho_raw(size);
    delta_rho_raw = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};
    std::vector<double> delta_energy_raw(size);
    delta_energy_raw = {28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};

    *solution_dPd_raw = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};

    std::vector<double> expected_pressure(size);
    expected_pressure = {-86,  -91,  -96,  -101, -106, -111, -116, -121, -126, -131, -136, -141, -146, -151,
                         -156, -161, -166, -171, -176, -181, -186, -191, -196, -201, -206, -211, -216};

    std::vector<double> rhs_pressure_raw(size);
    std::vector<double> rhs_energy_raw(size);

    Array3D<double> delta_rho(&delta_rho_raw[0], n, n, n);
    Array3D<double> delta_energy(&delta_energy_raw[0], n, n, n);
    Array3D<double> rhs_pressure(&rhs_pressure_raw[0], n, n, n);
    Array3D<double> rhs_energy(&rhs_energy_raw[0], n, n, n);

    int prec_nvar = 0;
    std::vector<double> expected_energy(27);
    SECTION("prec_nvar == 1")
    {
        prec_nvar = 1;
        for (int i = 0; i < 27; i++)
            expected_energy[i] = 0;
    }
    SECTION("prec_nvar == 2")
    {
        prec_nvar       = 2;
        expected_energy = {-28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41,
                           -42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54};
    }

    PhysicsBasedPreconditioner pbp(*solution, *solution_dPd, prec_nvar);

    verifySetPressureRHS(
        n, n, n, delta_rho, delta_energy, rhs_pressure, rhs_energy, &expected_pressure[0], &expected_energy[0], pbp);
}

TEST_CASE_METHOD(PhysicsBasedPreconditionerTest, "PhysicsBasedPreconditioner::setPressureRHS roberts dynamo problem")
{
    const int ni   = 32;
    const int nj   = 32;
    const int nk   = 64;
    const int size = ni * nj * nk;

    std::vector<double> delta_rho_raw(size);
    std::vector<double> delta_energy_raw(size);
    std::vector<double> expected_pressure(size);

    alloc_solution(ni, nj, nk);

    // Energy will remain unchanged in this example as we are using `prec_nvar == 1`
    // but we set to zeros to help catch if we break something in the future.
    std::vector<double> expected_energy(size);
    for (int i = 0; i < size; i++)
        expected_energy[i] = 0;

    std::vector<double> rhs_pressure_raw(size);
    std::vector<double> rhs_energy_raw(size);

    Array3D<double> delta_rho(&delta_rho_raw[0], ni, nj, nk);
    Array3D<double> delta_energy(&delta_energy_raw[0], ni, nj, nk);
    Array3D<double> rhs_pressure(&rhs_pressure_raw[0], ni, nj, nk);
    Array3D<double> rhs_energy(&rhs_energy_raw[0], ni, nj, nk);

    int prec_nvar = 1;
    PhysicsBasedPreconditioner pbp(*solution, *solution_dPd, prec_nvar);

    int n_dPdrho    = pbp.n_dPdrho();
    int n_dPdenergy = pbp.n_dPdenergy();

    // Read input data from file
    FILE* inputs_file = fopen(TEST_DATA_DIR "/set_pressure_rhs_roberts_dynamo_inputs.dat", "r");
    REQUIRE(inputs_file != nullptr);
    for (int i = 0; i < size; i++)
    {
        // Reads each row of the input file ensuring there are 4 columns in each
        REQUIRE(fscanf(inputs_file,
                       "%lf %lf %lf %lf",
                       &(*solution_dPd_raw)[i + n_dPdrho * size],
                       &delta_rho_raw[i],
                       &(*solution_dPd_raw)[i + n_dPdenergy * size],
                       &delta_energy_raw[i]) == 4);
    }
    fclose(inputs_file);

    // Read expected output from file
    FILE* outputs_file = fopen(TEST_DATA_DIR "/set_pressure_rhs_roberts_dynamo_expected_outputs.dat", "r");
    REQUIRE(outputs_file != nullptr);
    for (int i = 0; i < size; i++)
    {
        REQUIRE(fscanf(outputs_file, "%lf", &expected_pressure[i]) == 1);
    }
    fclose(outputs_file);

    verifySetPressureRHS(
        ni, nj, nk, delta_rho, delta_energy, rhs_pressure, rhs_energy, &expected_pressure[0], &expected_energy[0], pbp);
}
