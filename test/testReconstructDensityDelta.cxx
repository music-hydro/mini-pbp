#include "fixtures.cxx"
#include "PhysicsBasedPreconditioner.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <string>

TEST_CASE_METHOD(PhysicsBasedPreconditionerTest,
                 "PhysicsBasedPreconditioner::reconstructDensityDelta trivial ones to check pointers remain unchanged")
{
    unsigned int n    = 3;
    unsigned int size = n * n * n;

    std::vector<double> delta_pressure_raw(size);
    std::vector<double> delta_energy_raw(size);
    std::vector<double> delta_rho_raw(size);

    alloc_solution(n, n, n);

    for (int i = 0; i < size; i++)
    {
        delta_pressure_raw[i] = 1;
        delta_energy_raw[i]   = 1;
        delta_rho_raw[i]      = 1;
        // Ensure solution_dPd are not zero to prevent divide by zero
        for (int var = 0; var < 2; var++)
        {
            (*solution_dPd_raw)[i + var * size] = 1;
        }
    }

    Array3D<double> delta_pressure(&delta_pressure_raw[0], n, n, n);
    Array3D<double> delta_energy(&delta_energy_raw[0], n, n, n);
    Array3D<double> delta_rho(&delta_rho_raw[0], n, n, n);

    int prec_nvar = 1;
    PhysicsBasedPreconditioner pbp(*solution, *solution_dPd, *squared_sound_speed, prec_nvar);

    REQUIRE_NOTHROW(pbp.reconstructDensityDelta(delta_pressure, delta_energy, delta_rho));

    // Verify that looping through row-major i,j,k remains contiguous.
    int counter = 0;
    for (int k = 0; k < n; k++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                // delta_pressure
                REQUIRE(delta_pressure(i, j, k) == delta_pressure_raw[counter]);
                REQUIRE(&delta_pressure(i, j, k) == &delta_pressure_raw[counter]);
                REQUIRE(delta_pressure(i, j, k) == delta_pressure_raw[counter]);
                REQUIRE(&delta_pressure(i, j, k) == &delta_pressure_raw[counter]);

                // delta_energy
                REQUIRE(delta_energy(i, j, k) == delta_energy_raw[counter]);
                REQUIRE(&delta_energy(i, j, k) == &delta_energy_raw[counter]);
                REQUIRE(delta_energy(i, j, k) == delta_energy_raw[counter]);
                REQUIRE(&delta_energy(i, j, k) == &delta_energy_raw[counter]);

                // delta_rho
                REQUIRE(delta_rho(i, j, k) == delta_rho_raw[counter]);
                REQUIRE(&delta_rho(i, j, k) == &delta_rho_raw[counter]);
                REQUIRE(delta_rho(i, j, k) == delta_rho_raw[counter]);
                REQUIRE(&delta_rho(i, j, k) == &delta_rho_raw[counter]);

                counter++;
            }
        }
    }
}

TEST_CASE_METHOD(PhysicsBasedPreconditionerTest, "PhysicsBasedPreconditioner::reconstructDensityDelta roberts dynamo problem")
{
    const int ni   = 32;
    const int nj   = 32;
    const int nk   = 64;
    const int size = ni * nj * nk;

    std::vector<double> delta_pressure_raw(size);
    std::vector<double> delta_energy_raw(size);
    std::vector<double> delta_rho_raw(size);
    std::vector<double> expected_delta_rho(size);

    alloc_solution(ni, nj, nk);

    int prec_nvar = 1;
    PhysicsBasedPreconditioner pbp(*solution, *solution_dPd, *squared_sound_speed, prec_nvar);

    int n_dPdenergy = pbp.n_dPdenergy();
    int n_dPdrho    = pbp.n_dPdrho();

    // Read input data from file
    FILE* inputs_file = fopen(TEST_DATA_DIR "/calculate_density_delta_roberts_dynamo_inputs.dat", "r");
    REQUIRE(inputs_file != nullptr);
    for (int i = 0; i < size; i++)
    {
        // Reads each row of the input file ensuring there are 4 columns in each
        REQUIRE(fscanf(inputs_file,
                       "%lf %lf %lf %lf",
                       &delta_pressure_raw[i],
                       &(*solution_dPd_raw)[i + n_dPdenergy * size],
                       &delta_energy_raw[i],
                       &(*solution_dPd_raw)[i + n_dPdrho * size]) == 4);
    }
    fclose(inputs_file);

    // Read expected output from file
    FILE* outputs_file = fopen(TEST_DATA_DIR "/calculate_density_delta_roberts_dynamo_expected_outputs.dat", "r");
    REQUIRE(outputs_file != nullptr);
    for (int i = 0; i < size; i++)
    {
        REQUIRE(fscanf(outputs_file, "%lf", &expected_delta_rho[i]) == 1);
    }
    fclose(outputs_file);

    Array3D<double> delta_pressure(&delta_pressure_raw[0], ni, nj, nk);
    Array3D<double> delta_energy(&delta_energy_raw[0], ni, nj, nk);
    Array3D<double> delta_rho(&delta_rho_raw[0], ni, nj, nk);

    REQUIRE_NOTHROW(pbp.reconstructDensityDelta(delta_pressure, delta_energy, delta_rho));

    // Verify that looping through row-major i,j,k remains contiguous and results are accurate.
    int counter = 0;
    for (int k = 0; k < nk; k++)
    {
        for (int j = 0; j < nj; j++)
        {
            for (int i = 0; i < ni; i++)
            {
                REQUIRE_THAT(delta_rho(i, j, k), Catch::Matchers::WithinRel(expected_delta_rho[counter], 1e-14));
                REQUIRE(&delta_rho(i, j, k) == &delta_rho_raw[counter]);
                counter++;
            }
        }
    }
}
