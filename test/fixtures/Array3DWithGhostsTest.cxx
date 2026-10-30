/**
 * @class Fixture to deduplicate setup of tests of the Array3DWithGhosts class
 */

#ifndef Array3DWithGhostsTest_H
#define Array3DWithGhostsTest_H

#include "Array3D.h"

#include "Array3DWithGhosts.h"
#include <memory>
#include <vector>

class Array3DWithGhostsTest
{
  protected:
    std::shared_ptr<std::vector<double>> input_1d;
    std::shared_ptr<Array3D<double>> my_array;
    std::shared_ptr<Array3DWithGhosts<double>> my_array_with_ghosts;

    /// @brief Default constructor
    Array3DWithGhostsTest()
    {
        // Define the problem
        unsigned int ni = 30;
        unsigned int nj = 40;
        unsigned int nk = 30;
        size_t size     = ni * nj * nk;

        // Allocate a 1D array equal to the size of a 3D array
        input_1d = std::make_shared<std::vector<double>>(size);

        // Populate the 1D array with an incrementing contiguous counter
        for (int i = 0; i < (int)size; i++)
            (*input_1d)[i] = (double)i;

        my_array = std::make_shared<Array3D<double>>(ni, nj, nk, *input_1d);

        // Create an Array3DWithGhosts instance wrapping the Array3D
        my_array_with_ghosts = std::make_shared<Array3DWithGhosts<double>>(ni, nj, nk, *my_array);
        for (int axis = 0; axis < 3; axis++)
        {
            for (int side = Side::LEFT; side < Side::RIGHT; side++)
            {

                std::unique_ptr<Array2D<double>> new_ghost = nullptr;
                std::unique_ptr<Array2D<double>> new_edge  = nullptr;
                switch (axis)
                {
                    case 0:
                    {
                        new_ghost = std::make_unique<Array2D<double>>(nj, nk);
                        new_edge  = std::make_unique<Array2D<double>>(nj, nk);
                        break;
                    }
                    case 1:
                    {
                        new_ghost = std::make_unique<Array2D<double>>(ni - 1, nk);
                        new_edge  = std::make_unique<Array2D<double>>(ni - 1, nk);
                        break;
                    }
                    case 2:
                    {
                        new_ghost = std::make_unique<Array2D<double>>(nj - 1, nk - 1);
                        new_edge  = std::make_unique<Array2D<double>>(nj - 1, nk - 1);
                        break;
                    }
                }

                // Fill with -1.0 to help highlight unset values in tests.
                new_ghost->fill(-1.0);
                new_edge->fill(-1.0);
                my_array_with_ghosts->setGhostCells(axis, (Side)side, *new_ghost);
                my_array_with_ghosts->setEdgeCells(axis, (Side)side, *new_edge);
            }
        }
    }
};

#endif  // Array3DWithGhostsTest_H
