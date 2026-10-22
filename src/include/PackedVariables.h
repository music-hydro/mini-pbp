/**
 * @class PackedVarsView
 */

#ifndef PackedVariables_H
#define PackedVariables_H

#include "Array3D.h"
#include <vector>

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
    PackedVariables(std::vector<double>& packed_array_pointer, const int ni, const int nj, const int nk);

    /**
     * @brief Return the full 3D array view for a chosen variable
     *
     * @param variable_index - The variable whose 3D array is to be extracted.
     *
     * @return An Array3D instance providing a 3D view of the requested variable's 1D array
     */
    Array3D<double> singleVariableView(const int variable_index) const;

    /**
     * @brief Return the number of variables
     */
    int numVariables()
    { return m_num_variables; }


  private:
    /// Pointer to origin of the 3D view in the larger 1D array
    double* const m_packed_array;
    /// The lengths of the physical cell dimensions
    const int m_ni;
    const int m_nj;
    const int m_nk;
    /// The number of variables stored in the packed array
    const int m_num_variables;
};

#endif  // PackedVariables_H
