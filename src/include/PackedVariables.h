/**
 * @class PackedVarsView
 */

#ifndef PackedVariables_H
#define PackedVariables_H

#include "Array3D.h"

enum VariableIndex
{
    RHO,
    E,
    DPDRHO,
    DPDE,
    PRESSURE,
    ENERGY
};

class PackedVariables
{
  public:
    /**
     * @brief Constructor
     *
     * @param packed_array_pointer - Pointer to the full 1D array containing all variables
     * @param ni - The number of physical cells in the first direction.
     * @param nj - The number of physical cells in the second direction.
     * @param nk - The number of physical cells in the third direction.
     */
    PackedVariables(double* packed_array_pointer, int ni, int nj, int nk);

    /**
     * @brief Return the full 3D array view for a chosen variable
     *
     * @param variable_index - The variable whose 3D array is to be extracted.
     *
     * @return An Array3D instance providing a 3D view of the requested variable's 1D array
     */
    Array3D<double> singleVariableView(VariableIndex variable_index) const;

  private:
    /// Pointer to origin of the 3D view in the larger 1D array
    double* m_packed_array;
    /// The lengths of the physical cell dimensions
    int m_ni;
    int m_nj;
    int m_nk;
};

#endif  // PackedVariables_H
