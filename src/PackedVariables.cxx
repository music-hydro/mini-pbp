#include "PackedVariables.h"

#include <stdexcept>
#include <string>

PackedVariables::PackedVariables(std::vector<double>& packed_array_pointer,
                                 const unsigned int ni,
                                 const unsigned int nj,
                                 const unsigned int nk) :
    m_packed_array(&packed_array_pointer[0]),
    m_ni(ni),
    m_nj(nj),
    m_nk(nk)
{
    // Verify the size of solutions
    unsigned long long domain_size = ni * nj * nk;
    if (packed_array_pointer.size() % domain_size != 0)
    {
        throw std::invalid_argument("Size of packed_array_pointer is not a multiple of (ni * nj * nk)");
    }
    m_num_variables = (packed_array_pointer.size() / domain_size);
}

Array3D<double> PackedVariables::singleVariableView(const unsigned int variable_index) const
{
    // Validate requested variable index against what is assumed
    if (variable_index >= m_num_variables)
    {
        throw std::invalid_argument("Provided variable index not found: " + std::to_string(variable_index));
    }

    unsigned int offset = m_ni * m_nj * m_nk * variable_index;
    Array3D<double> variable_view(m_ni, m_nj, m_nk, m_packed_array + offset);
    return variable_view;
}
