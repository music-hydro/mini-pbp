#include "PackedVariables.h"

#include <stdexcept>
#include <string>

PackedVariables::PackedVariables(std::vector<double>& packed_array_pointer, const int ni, const int nj, const int nk) :
    m_packed_array(&packed_array_pointer[0]),
    m_ni(ni),
    m_nj(nj),
    m_nk(nk),
    m_num_variables(packed_array_pointer.size() / (ni * nj * nk))
{
    // Verify the size of solutions
    if (packed_array_pointer.size() % (ni * nj * nk) != 0)
    {
        throw std::invalid_argument("Size of packed_array_pointer is not a multiple of (ni * nj * nk)");
    }
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
