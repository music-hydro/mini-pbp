/**
 * @brief Application entrypoint for the music mini physics based preconditioner (mini-pbp)
 */

#include "Array3D.h"
#include "PackedVariables.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv)
{
    int length_each_dim = 3;
    int num_dims        = 3;
    int size            = length_each_dim * length_each_dim * length_each_dim;

    double* input_1d_pointer = new double[size];

    int counter = 0;
    for (int i = 0; i < size; i++)
    {
        input_1d_pointer[i] = (double)i;
    }

    int offset = (1 * 16) + (1 * 4) + 1;
    Array3D<double> my_array(input_1d_pointer, 3, 3, 3);

    int count = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                my_array(i, j, k);
            }
        }
    }
}
