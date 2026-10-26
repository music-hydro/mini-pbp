/**
 * @file A collection of test fixtures to reduce duplication across tests that require similar state setup
 */

#include "PackedVariables.h"
#include <memory>
#include <vector>

/**
 * @class Fixture to deduplicate setup of tests of the PhysicsBasedPreconditioner class
 */
class PhysicsBasedPreconditionerTest
{
  protected:
    std::shared_ptr<std::vector<double>> solution_raw;
    std::shared_ptr<std::vector<double>> solution_dPd_raw;
    std::shared_ptr<PackedVariables> solution;
    std::shared_ptr<PackedVariables> solution_dPd;

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

        solution_raw     = std::make_shared<std::vector<double>>(size * 3);
        solution_dPd_raw = std::make_shared<std::vector<double>>(size * 2);

        for (int i = 0; i < size; i++)
        {
            for (int var = 0; var < 3; var++)
            {
                (*solution_raw)[i + size * var] = 0;
            }
            for (int var = 0; var < 2; var++)
            {
                (*solution_dPd_raw)[i + size * var] = 0;
            }
        }

        solution     = std::make_shared<PackedVariables>(*solution_raw, ni, nj, nk);
        solution_dPd = std::make_shared<PackedVariables>(*solution_dPd_raw, ni, nj, nk);
    }
};
