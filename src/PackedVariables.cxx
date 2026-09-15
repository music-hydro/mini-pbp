#include "PackedVariables.h"

PackedVariables::PackedVariables(double* packed_array_pointer, int ni, int nj, int nk) :
    m_packed_array(packed_array_pointer),
    m_ni(ni),
    m_nj(nj),
    m_nk(nk)
{
}

Array3D<double> PackedVariables::singleVariableView(VariableIndex variable_index) const
{
    int offset = m_ni * m_nj * m_nk * (int)variable_index;
    Array3D<double> variable_view(m_packed_array + offset, m_ni, m_nj, m_nk);
    return variable_view;
}
