/**
 * @class Fixture to deduplicate setup of tests of the PhysicsBasedPreconditioner class
 */

#ifndef PHYSICSBASEDPRECONDITIONERTEST_H
#define PHYSICSBASEDPRECONDITIONERTEST_H

#include "PhysicsBasedPreconditioner.h"

#include "PackedVariables.h"
#include <memory>
#include <vector>

class PhysicsBasedPreconditionerTest
{
  protected:
    std::unique_ptr<std::vector<double>> solution_raw;
    std::unique_ptr<std::vector<double>> solution_dPd_raw;
    std::unique_ptr<std::vector<double>> squared_sound_speed_raw;
    std::unique_ptr<PackedVariables> solution;
    std::unique_ptr<PackedVariables> solution_dPd;
    std::unique_ptr<Array3D<double>> squared_sound_speed;

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

        solution_raw            = std::make_unique<std::vector<double>>(size * 3);
        solution_dPd_raw        = std::make_unique<std::vector<double>>(size * 2);
        squared_sound_speed_raw = std::make_unique<std::vector<double>>(size);

        for (int i = 0; i < size; i++)
        {
            (*squared_sound_speed_raw)[i] = 0;
            for (int var = 0; var < 3; var++)
            {
                (*solution_raw)[i + size * var] = 0;
            }
            for (int var = 0; var < 2; var++)
            {
                (*solution_dPd_raw)[i + size * var] = 0;
            }
        }

        solution            = std::make_unique<PackedVariables>(*solution_raw, ni, nj, nk);
        solution_dPd        = std::make_unique<PackedVariables>(*solution_dPd_raw, ni, nj, nk);
        squared_sound_speed = std::make_unique<Array3D<double>>(ni, nj, nk, *squared_sound_speed_raw);
    }
};

#endif  // PHYSICSBASEDPRECONDITIONERTEST_H
