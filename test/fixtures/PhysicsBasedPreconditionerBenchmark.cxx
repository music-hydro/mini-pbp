/**
 * @class Fixture to deduplicate setup of benchmarks of the PhysicsBasedPreconditioner class
 */

#ifndef PHYSICSBASEDPRECONDITIONERBENCHMARK_H
#define PHYSICSBASEDPRECONDITIONERBENCHMARK_H

#include "PhysicsBasedPreconditioner.h"

#include "PackedVariables.h"
#include "PhysicsBasedPreconditionerTest.cxx"
#include <memory>
#include <vector>

class PhysicsBasedPreconditionerBenchmark : protected PhysicsBasedPreconditionerTest
{
  public:
    const unsigned int ni = 100;
    const unsigned int nj = 100;
    const unsigned int nk = 100;
    unsigned long long size;
    unsigned int prec_nvar;

    std::unique_ptr<std::vector<double>> delta_pressure_raw;
    std::unique_ptr<std::vector<double>> delta_energy_raw;
    std::unique_ptr<std::vector<double>> delta_rho_raw;
    std::unique_ptr<std::vector<double>> preconditioned_delta_pressure_raw;
    std::unique_ptr<std::vector<double>> preconditioned_delta_energy_raw;
    std::unique_ptr<std::vector<double>> rhs_pressure_raw;
    std::unique_ptr<std::vector<double>> rhs_energy_raw;

    std::unique_ptr<Array3D<double>> delta_pressure;
    std::unique_ptr<Array3D<double>> delta_energy;
    std::unique_ptr<Array3D<double>> delta_rho;
    std::unique_ptr<Array3D<double>> preconditioned_delta_pressure;
    std::unique_ptr<Array3D<double>> preconditioned_delta_energy;
    std::unique_ptr<Array3D<double>> rhs_pressure;
    std::unique_ptr<Array3D<double>> rhs_energy;

    std::unique_ptr<PhysicsBasedPreconditioner> pbp;

    PhysicsBasedPreconditionerBenchmark()
    {
        size = ni * nj * nk;

        delta_pressure_raw                = std::make_unique<std::vector<double>>(size);
        delta_energy_raw                  = std::make_unique<std::vector<double>>(size);
        delta_rho_raw                     = std::make_unique<std::vector<double>>(size);
        preconditioned_delta_pressure_raw = std::make_unique<std::vector<double>>(size);
        preconditioned_delta_energy_raw   = std::make_unique<std::vector<double>>(size);
        rhs_pressure_raw                  = std::make_unique<std::vector<double>>(size);
        rhs_energy_raw                    = std::make_unique<std::vector<double>>(size);

        alloc_solution(ni, nj, nk);

        for (int idx = 0; idx < size; idx++)
        {
            (*delta_pressure_raw)[idx]                = 0;
            (*delta_energy_raw)[idx]                  = 0;
            (*delta_rho_raw)[idx]                     = 0;
            (*preconditioned_delta_pressure_raw)[idx] = 0;
            (*preconditioned_delta_energy_raw)[idx]   = 0;
            (*rhs_pressure_raw)[idx]                  = 0;
            (*rhs_energy_raw)[idx]                    = 0;
        }

        delta_pressure                = std::make_unique<Array3D<double>>(ni, nj, nk, *delta_pressure_raw);
        delta_energy                  = std::make_unique<Array3D<double>>(ni, nj, nk, *delta_energy_raw);
        delta_rho                     = std::make_unique<Array3D<double>>(ni, nj, nk, *delta_rho_raw);
        preconditioned_delta_pressure = std::make_unique<Array3D<double>>(ni, nj, nk, *preconditioned_delta_pressure_raw);
        preconditioned_delta_energy   = std::make_unique<Array3D<double>>(ni, nj, nk, *preconditioned_delta_energy_raw);
        rhs_pressure                  = std::make_unique<Array3D<double>>(ni, nj, nk, *rhs_pressure_raw);
        rhs_energy                    = std::make_unique<Array3D<double>>(ni, nj, nk, *rhs_energy_raw);

        prec_nvar = 1;
        pbp       = std::make_unique<PhysicsBasedPreconditioner>(*solution, *solution_dPd, *squared_sound_speed, prec_nvar);
    }
};

#endif  // PHYSICSBASEDPRECONDITIONERBENCHMARK_H
