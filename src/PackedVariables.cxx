#include "PackedVariables.h"

#include <stdexcept>
#include <string>

PackedVariables::PackedVariables(double* const packed_array_pointer,
                                 const int ni,
                                 const int nj,
                                 const int nk,
                                 const int num_variables) :
    m_packed_array(packed_array_pointer),
    m_ni(ni),
    m_nj(nj),
    m_nk(nk),
    m_num_variables(num_variables)
{
}

Array3D<double> PackedVariables::singleVariableView(const int variable_index) const
{
    // Validate requested variable index against what is assumed
    if (variable_index < 0 || variable_index >= m_num_variables)
    {
        throw std::invalid_argument("Provided variable index not found: " + std::to_string(variable_index));
    }

    int offset = m_ni * m_nj * m_nk * variable_index;
    Array3D<double> variable_view(m_packed_array + offset, m_ni, m_nj, m_nk);
    return variable_view;
}
