/**
 * @class Fixture to deduplicate setup of tests of the PhysicsBasedPreconditioner class
 */

#ifndef PHYSICSBASEDPRECONDITIONERTEST_H
#define PHYSICSBASEDPRECONDITIONERTEST_H

#include "PhysicsBasedPreconditioner.h"

#include "GhostExchangePlanTest.cxx"
#include "PackedVariables.h"
#include <memory>
#include <vector>

class PhysicsBasedPreconditionerTest
{
  protected:
    std::unique_ptr<std::vector<double>> solution_raw;
    std::unique_ptr<std::vector<double>> solution_dPd_raw;
    std::unique_ptr<std::vector<double>> squared_sound_speed_raw;
    std::unique_ptr<PackedVariablesWithGhosts> solution;
    std::unique_ptr<PackedVariables> solution_dPd;
    std::unique_ptr<Array3D<double>> squared_sound_speed;
    const int ndim = 3, nmom = 3;
    const bool use_mhd = false;
    const double dt    = 4.5228571428571410E-004;  // Taken from roberts dynamo

    /**
     * @brief Allocation the raw solution vectors with spacial indices `ni`, `nj` and `nk`.
     *
     * @param ni - Number of spacial elements in the ith direction.
     * @param nj - Number of spacial elements in the jth direction.
     * @param nk - Number of spacial elements in the kth direction.
     */
    void alloc_solution(int ni, int nj, int nk)
    {
        int size = ni * nj * nk;

        const unsigned int num_sol_vars     = PhysicsBasedPreconditioner::num_sol_vars();
        const unsigned int num_sol_dPd_vars = PhysicsBasedPreconditioner::num_sol_dPd_vars();
        solution_raw                        = std::make_unique<std::vector<double>>(size * num_sol_vars);
        solution_dPd_raw                    = std::make_unique<std::vector<double>>(size * num_sol_dPd_vars);
        squared_sound_speed_raw             = std::make_unique<std::vector<double>>(size);

        for (int i = 0; i < size; i++)
        {
            (*squared_sound_speed_raw)[i] = 0;
            for (int var = 0; var < num_sol_vars; var++)
            {
                (*solution_raw)[i + size * var] = 0;
            }
            for (int var = 0; var < num_sol_dPd_vars; var++)
            {
                (*solution_dPd_raw)[i + size * var] = 0;
            }
        }

        solution = std::make_unique<PackedVariablesWithGhosts>(*solution_raw, ni, nj, nk, GhostExchangePlanTest::arbitraryPlan());
        solution_dPd        = std::make_unique<PackedVariables>(*solution_dPd_raw, ni, nj, nk);
        squared_sound_speed = std::make_unique<Array3D<double>>(ni, nj, nk, *squared_sound_speed_raw);
    }
};

#endif  // PHYSICSBASEDPRECONDITIONERTEST_H
