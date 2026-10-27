#include "fixtures/PhysicsBasedPreconditionerBenchmark.cxx"
#include "fixtures/PhysicsBasedPreconditionerTest.cxx"
#include "PhysicsBasedPreconditioner.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <string>

TEST_CASE_METHOD(PhysicsBasedPreconditionerTest,
                 "PhysicsBasedPreconditioner::reconstructEnergyDelta trivial ones to check pointers remain unchanged",
                 "[prec_nvar]")
{
    // Define the problem
    int n    = 3;
    int size = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> preconditioned_delta_pressure_raw(size);
    std::vector<double> preconditioned_delta_energy_raw(size);
    std::vector<double> dp_rhs_press_raw(size);
    std::vector<double> delta_energy_raw(size);
    std::vector<double> expected_energy(size);

    alloc_solution(n, n, n);

    // Initialise arrays
    for (int i = 0; i < size; i++)
    {
        preconditioned_delta_pressure_raw[i] = 1;
        preconditioned_delta_energy_raw[i]   = 1;
        dp_rhs_press_raw[i]                  = 1;
        delta_energy_raw[i]                  = 1;
        expected_energy[i]                   = 1;
        // Ensure solutions are not zero to prevent divide by zero
        for (int var = 0; var < 3; var++)
        {
            (*solution_raw)[i + var * size] = 1;
        }
        for (int var = 0; var < 2; var++)
        {
            (*solution_dPd_raw)[i + var * size] = 1;
        }
    }

    Array3D<double> preconditioned_delta_pressure(&preconditioned_delta_pressure_raw[0], n, n, n);
    Array3D<double> preconditioned_delta_energy(&preconditioned_delta_energy_raw[0], n, n, n);
    Array3D<double> dp_rhs_press(&dp_rhs_press_raw[0], n, n, n);
    Array3D<double> delta_energy(&delta_energy_raw[0], n, n, n);

    int prec_nvar = 0;
    SECTION("prec_nvar == 1")
    { prec_nvar = 1; }
    SECTION("prec_nvar == 2")
    { prec_nvar = 2; }

    REQUIRE(prec_nvar != 0);

    PhysicsBasedPreconditioner pbp(*solution, *solution_dPd, *squared_sound_speed, prec_nvar);

    REQUIRE_NOTHROW(
        pbp.reconstructEnergyDelta(preconditioned_delta_pressure, preconditioned_delta_energy, dp_rhs_press, delta_energy));

    PackedVariables new_solution = pbp.solution();

    int n_rho      = pbp.n_rho();
    int n_pressure = pbp.n_pressure();

    Array3D<double> solution_press     = (*solution).singleVariableView(n_pressure);
    Array3D<double> solution_rho       = (*solution).singleVariableView(n_rho);
    Array3D<double> new_solution_press = new_solution.singleVariableView(n_pressure);
    Array3D<double> new_solution_rho   = new_solution.singleVariableView(n_rho);

    // Verify that looping through row-major i,j,k remains contiguous.
    int counter = 0;
    for (int k = 0; k < n; k++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                // delta_prec
                REQUIRE(preconditioned_delta_pressure(i, j, k) == preconditioned_delta_pressure_raw[counter]);
                REQUIRE(&preconditioned_delta_pressure(i, j, k) == &preconditioned_delta_pressure_raw[counter]);
                REQUIRE(preconditioned_delta_energy(i, j, k) == preconditioned_delta_energy_raw[counter]);
                REQUIRE(&preconditioned_delta_energy(i, j, k) == &preconditioned_delta_energy_raw[counter]);

                // solution
                REQUIRE(solution_press(i, j, k) == (*solution_raw)[counter + size * n_pressure]);
                REQUIRE(&solution_press(i, j, k) == &(*solution_raw)[counter + size * n_pressure]);
                REQUIRE(solution_rho(i, j, k) == (*solution_raw)[counter + size * n_rho]);
                REQUIRE(&solution_rho(i, j, k) == &(*solution_raw)[counter + size * n_rho]);
                REQUIRE(new_solution_press(i, j, k) == (*solution_raw)[counter + size * n_pressure]);
                REQUIRE(&new_solution_press(i, j, k) == &(*solution_raw)[counter + size * n_pressure]);
                REQUIRE(new_solution_rho(i, j, k) == (*solution_raw)[counter + size * n_rho]);
                REQUIRE(&new_solution_rho(i, j, k) == &(*solution_raw)[counter + size * n_rho]);

                // dp_rhs
                REQUIRE(dp_rhs_press(i, j, k) == dp_rhs_press_raw[counter]);
                REQUIRE(&dp_rhs_press(i, j, k) == &dp_rhs_press_raw[counter]);

                // delta_vars
                REQUIRE(delta_energy(i, j, k) == delta_energy_raw[counter]);
                REQUIRE(&delta_energy(i, j, k) == &delta_energy_raw[counter]);

                counter++;
            }
        }
    }
}

TEST_CASE_METHOD(PhysicsBasedPreconditionerTest,
                 "PhysicsBasedPreconditioner::reconstructEnergyDelta simple copy for prec_nvar == 2")
{
    // Define the problem
    int n    = 3;
    int size = n * n * n;

    // Allocate a 1D array equal to the size of a 3D array
    std::vector<double> preconditioned_delta_pressure_raw(size);
    std::vector<double> preconditioned_delta_energy_raw(size);
    std::vector<double> dp_rhs_press_raw(size);
    std::vector<double> delta_energy_raw(size);
    std::vector<double> expected_energy(size);

    alloc_solution(n, n, n);

    Array3D<double> preconditioned_delta_pressure(&preconditioned_delta_pressure_raw[0], n, n, n);
    Array3D<double> preconditioned_delta_energy(&preconditioned_delta_energy_raw[0], n, n, n);
    Array3D<double> dp_rhs_press(&dp_rhs_press_raw[0], n, n, n);
    Array3D<double> delta_energy(&delta_energy_raw[0], n, n, n);

    // Initialise arrays
    for (int i = 0; i < size; i++)
    {
        preconditioned_delta_pressure_raw[i] = 0;
        preconditioned_delta_energy_raw[i]   = 2;
        dp_rhs_press_raw[i]                  = 0;
        delta_energy_raw[i]                  = 1;
        expected_energy[i]                   = 0;
    }

    int prec_nvar = 2;
    PhysicsBasedPreconditioner pbp(*solution, *solution_dPd, *squared_sound_speed, prec_nvar);

    REQUIRE_NOTHROW(
        pbp.reconstructEnergyDelta(preconditioned_delta_pressure, preconditioned_delta_energy, dp_rhs_press, delta_energy));

    // Verify that looping through row-major i,j,k remains contiguous.
    int counter = 0;
    for (int k = 0; k < n; k++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                // delta_prec
                REQUIRE(preconditioned_delta_pressure(i, j, k) == preconditioned_delta_pressure_raw[counter]);
                REQUIRE(&preconditioned_delta_pressure(i, j, k) == &preconditioned_delta_pressure_raw[counter]);
                REQUIRE(preconditioned_delta_energy(i, j, k) == preconditioned_delta_energy_raw[counter]);
                REQUIRE(&preconditioned_delta_energy(i, j, k) == &preconditioned_delta_energy_raw[counter]);

                // dp_rhs
                REQUIRE(dp_rhs_press(i, j, k) == dp_rhs_press_raw[counter]);
                REQUIRE(&dp_rhs_press(i, j, k) == &dp_rhs_press_raw[counter]);

                // delta_vars
                REQUIRE(delta_energy(i, j, k) == delta_energy_raw[counter]);
                REQUIRE(&delta_energy(i, j, k) == &delta_energy_raw[counter]);
                REQUIRE(delta_energy(i, j, k) == preconditioned_delta_energy(i, j, k));
                REQUIRE(&delta_energy(i, j, k) != &preconditioned_delta_energy_raw[counter]);

                counter++;
            }
        }
    }
}

TEST_CASE_METHOD(PhysicsBasedPreconditionerTest, "PhysicsBasedPreconditioner::reconstructEnergyDelta roberts dynamo problem")
{
    const int ni   = 32;
    const int nj   = 32;
    const int nk   = 64;
    const int size = ni * nj * nk;

    // Allocate a 1D arrays equal to the size of a 3D arrays
    std::vector<double> preconditioned_delta_pressure_raw(size);
    std::vector<double> preconditioned_delta_energy_raw(size);
    std::vector<double> dp_rhs_press_raw(size);
    std::vector<double> delta_energy_raw(size);
    std::vector<double> expected_energy(size);

    alloc_solution(ni, nj, nk);

    int prec_nvar = 1;
    PhysicsBasedPreconditioner pbp(*solution, *solution_dPd, *squared_sound_speed, prec_nvar);

    int n_rho      = pbp.n_rho();
    int n_pressure = pbp.n_pressure();

    // Read input data from file
    FILE* inputs_file = fopen(TEST_DATA_DIR "/reconstruct_energy_delta_roberts_dynamo_inputs.dat", "r");
    REQUIRE(inputs_file != nullptr);
    for (int i = 0; i < size; i++)
    {
        // Reads each row of the input file ensuring there are 4 columns in each
        REQUIRE(fscanf(inputs_file,
                       "%lf %lf %lf %lf %lf",
                       &preconditioned_delta_pressure_raw[i],
                       &dp_rhs_press_raw[i],
                       &delta_energy_raw[i],
                       &(*solution_raw)[i + n_pressure * size],
                       &(*solution_raw)[i + n_rho * size]) == 5);
    }
    fclose(inputs_file);

    // Read expected output from file
    FILE* outputs_file = fopen(TEST_DATA_DIR "/reconstruct_energy_delta_roberts_dynamo_expected_outputs.dat", "r");
    REQUIRE(outputs_file != nullptr);
    for (int i = 0; i < size; i++)
    {
        REQUIRE(fscanf(outputs_file, "%lf", &expected_energy[i]) == 1);
    }
    fclose(outputs_file);

    // Create variable arrays
    Array3D<double> preconditioned_delta_pressure(&preconditioned_delta_pressure_raw[0], ni, nj, nk);
    Array3D<double> preconditioned_delta_energy(&preconditioned_delta_energy_raw[0], ni, nj, nk);
    Array3D<double> dp_rhs_press(&dp_rhs_press_raw[0], ni, nj, nk);
    Array3D<double> delta_energy(&delta_energy_raw[0], ni, nj, nk);

    REQUIRE_NOTHROW(
        pbp.reconstructEnergyDelta(preconditioned_delta_pressure, preconditioned_delta_energy, dp_rhs_press, delta_energy));

    // Verify that looping through row-major i,j,k remains contiguous and results are accurate.
    int counter = 0;
    for (int k = 0; k < nk; k++)
    {
        for (int j = 0; j < nj; j++)
        {
            for (int i = 0; i < ni; i++)
            {
                REQUIRE_THAT(delta_energy(i, j, k), Catch::Matchers::WithinRel(expected_energy[counter], 1e-14));
                REQUIRE(&delta_energy(i, j, k) == &delta_energy_raw[counter]);
                counter++;
            }
        }
    }
}
