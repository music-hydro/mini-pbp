/**
 * @class Array3D
 *
 * @brief Provides a 3D view on a 1D array with custom bounds
 */

#ifndef Array3D_H
#define Array3D_H

template<typename T>
class Array3D
{
  public:
    /**
     * @brief Constructs an Array3D from a pointer to a 1D array
     *
     * @param data_1D - Pointer to the input 1D array.
     * @param ni - The number of elements to be extracted in the first direction.
     * @param nj - The number of elements to be extracted in the second direction.
     * @param nk - The number of elements to be extracted in the third direction.
     */
    Array3D(T* const data_1D, const int ni, const int nj, const int nk) :
        m_data_origin(data_1D),
        m_ni(ni),
        m_nj(nj),
        m_nk(nk)
    {
    }

    /**
     * @brief access operator to return the value at the given element.
     *
     * @param i - The index in the first dimension.
     * @param j - The index in the second dimension.
     * @param k - The index in the third dimension.
     *
     * @return The value stored at the requested 3D coordinates
     */
    T& operator ()(int i, int j, int k) const
    {
        int offset = (i * m_nk * m_nj) + (j * m_nk) + k;
        return m_data_origin[offset];
    }

  private:
    /// Pointer to origin of the 3D view in the larger 1D array
    T* const m_data_origin;
    /// Dimension sizes in 3D view
    const int m_ni;
    const int m_nj;
    const int m_nk;
};

#endif  // Array3D_H
