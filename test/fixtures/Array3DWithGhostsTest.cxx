/**
 * @class Fixture to deduplicate setup of tests of the Array3DWithGhosts class
 */

#ifndef Array3DWithGhostsTest_H
#define Array3DWithGhostsTest_H

#include "Array3D.h"

#include "Array3DWithGhosts.h"
#include "GhostExchangePlan.h"
#include "MpiTest.cxx"
#include <memory>
#include <vector>

class Array3DWithGhostsTest : public MpiTest
{
  protected:
    std::shared_ptr<std::vector<double>> input_1d;
    std::shared_ptr<Array3D<double>> my_array;
    std::shared_ptr<GhostExchangePlan<double>> my_plan;
    std::shared_ptr<Array3DWithGhosts<double>> my_array_with_ghosts;

    /**
     * @brief Default constructor
     *
     * @details Allocates and populates the vector and Array3D which define
     *          this Array3DWithGhosts and instantiates the Array3DWithGhosts.
     *          Ghost and boundary slabs are also filled with -1.0.
     */
    Array3DWithGhostsTest() :
        MpiTest()
    {
        // Define the problem
        unsigned int ni = 30;
        unsigned int nj = 40;
        unsigned int nk = 30;
        size_t size     = ni * nj * nk;

        // Allocate a 1D array equal to the size of a 3D array
        input_1d = std::make_shared<std::vector<double>>(size);

        my_array = std::make_shared<Array3D<double>>(ni, nj, nk, *input_1d);
        ;
        my_plan = std::make_shared<GhostExchangePlan<double>>(
            *my_array, m_local_boundary_conditions, m_neighbours, m_cartesian_communicator);

        // Create an Array3DWithGhosts instance wrapping the Array3D
        my_array_with_ghosts = std::make_shared<Array3DWithGhosts<double>>(*my_plan);

        // Populate the 1D array with the rank of the current process
        for (int i = 0; i < (size_t)size; i++)
            (*input_1d)[i] = my_array_with_ghosts->m_plan.m_my_rank;

        // Fill with -42.0 to help highlight unset values in tests.
        for (int plane_idx = 0; plane_idx < 6; plane_idx++)
        {
            my_array_with_ghosts->m_ghosts[plane_idx].fill(-42.0);
            my_array_with_ghosts->m_boundaries[plane_idx].fill(-42.0);
        }
    }

    void callUpdateBoundaries()
    { my_array_with_ghosts->updateBoundaries(); }
};

#endif  // Array3DWithGhostsTest_H
