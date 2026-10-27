/**
 * @class Array3D
 *
 * @brief Provides a 3D view on a 1D array with custom bounds
 */

#ifndef Array3D_H
#define Array3D_H

#include "Slab.h"

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
    Array3D(T* const& data_1D, const unsigned int ni, const unsigned int nj, const unsigned int nk) :
        m_data_origin(data_1D),
        m_ni(ni),
        m_nj(nj),
        m_nk(nk),
        m_stride_i(1),
        m_stride_j(ni),
        m_stride_k(ni * nj)
    {
    }

    /**
     * @brief Constructs an Array3D from a pointer to a 1D array
     *
     * @param data_1D - Pointer to the input 1D array.
     * @param ni - The number of elements to be extracted in the first direction.
     * @param nj - The number of elements to be extracted in the second direction.
     * @param nk - The number of elements to be extracted in the third direction.
     * @param[in] stride_i Stride along the first direction.
     * @param[in] stride_j Stride along the second direction.
     * @param[in] stride_k Stride along the third direction.
     */
    Array3D(T* const& data_1D,
            const unsigned int ni,
            const unsigned int nj,
            const unsigned int nk,
            const unsigned int stride_i,
            const unsigned int stride_j,
            const unsigned int stride_k) :
        m_data_origin(data_1D),
        m_ni(ni),
        m_nj(nj),
        m_nk(nk),
        m_stride_i(stride_i),
        m_stride_j(stride_j),
        m_stride_k(stride_k)
    {
    }

    /**
     * @brief access operator to return a reference to the given element.
     *
     * @param i - The index in the first dimension.
     * @param j - The index in the second dimension.
     * @param k - The index in the third dimension.
     *
     * @return Reference to the element at the requested 3D coordinates
     */
    T& operator ()(unsigned int i, unsigned int j, unsigned int k) const
    {
        const unsigned int offset = (i * m_stride_i) + (j * m_stride_j) + (k * m_stride_k);
        return m_data_origin[offset];
    }

    /**
     * @brief Getter for the slab this array represents
     */
    Slab cellSlab() const
    {
        Slab slab({0, 0, 0}, {m_ni, m_nj, m_nk});
        return slab;
    }

    /**
     * @brief Deep copy this array into the target array
     *
     * @details This function will not allocate any memory. This must be done
     *          prior to this call.
     *
     * @param[out] targetArray - The array into which the copy should be made.
     * @param[in] factor - Factor by which to multiply each array value when copying.
     */
    void deepCopy(Array3D<T>& targetArray, const T& factor = 1) const
    {
        Slab slab = cellSlab();
        for (int k = slab.start(2); k < slab.end(2); k++)
        {
            for (int j = slab.start(1); j < slab.end(1); j++)
            {
                for (int i = slab.start(0); i < slab.end(0); i++)
                {
                    targetArray(i, j, k) = operator ()(i, j, k) * factor;
                }
            }
        }
    }

    /**
     * @brief Fill this array with a scalar value
     *
     * @param val - Scalar value to fill the array with.
     */
    void fill(const T& val) const
    {
        Slab slab = cellSlab();
        for (int k = slab.start(2); k < slab.end(2); k++)
        {
            for (int j = slab.start(1); j < slab.end(1); j++)
            {
                for (int i = slab.start(0); i < slab.end(0); i++)
                {
                    operator ()(i, j, k) = val;
                }
            }
        }
    }

  private:
    T* const m_data_origin;        /**< @brief Pointer to array memory origin */
    const unsigned int m_ni;       /**< @brief Dimension length the first dimension */
    const unsigned int m_nj;       /**< @brief Dimension length the second dimension */
    const unsigned int m_nk;       /**< @brief Dimension length the third dimension */
    const unsigned int m_stride_i; /**< @brief Index stride along the first dimension */
    const unsigned int m_stride_j; /**< @brief Index stride along the second dimension */
    const unsigned int m_stride_k; /**< @brief Index stride along the third dimension */
};

#endif  // Array3D_H
