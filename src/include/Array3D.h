/**
 * @class Array3D
 *
 * @brief Provides a 3D view on a 1D array with custom bounds
 */

#ifndef Array3D_H
#define Array3D_H

#include "Slab.h"
#include <memory>
#include <stdexcept>
#include <vector>

template<typename T>
class Array3D
{
  public:
    /**
     * @brief Constructs an Array3D from array bounds alone.
     *
     * @details the wrapped 1D data will be allocated by the constructor.
     *
     * @param[in] ni - The number of elements in the first direction.
     * @param[in] nj - The number of elements in the second direction.
     * @param[in] nk - The number of elements in the third direction.
     */
    Array3D(const unsigned int ni, const unsigned int nj, const unsigned int nk) :
        m_data_origin(static_cast<T*>(std::malloc(sizeof(T) * static_cast<size_t>(ni) * nj * nk))),
        m_stride_i(1),
        m_stride_j(ni),
        m_stride_k(ni * nj),
        m_slab({0, 0, 0}, {(int)ni, (int)nj, (int)nk})
    {
    }

    /**
     * @brief Constructs an Array3D from a pointer to a 1D array.
     *
     * @param[in] ni - The number of elements in the first direction.
     * @param[in] nj - The number of elements in the second direction.
     * @param[in] nk - The number of elements in the third direction.
     * @param[in] data_1D - Pointer to the input 1D array.
     */
    Array3D(const unsigned int ni, const unsigned int nj, const unsigned int nk, double* const data_1D) :
        m_data_origin(data_1D),
        m_stride_i(1),
        m_stride_j(ni),
        m_stride_k(ni * nj),
        m_slab({0, 0, 0}, {(int)ni, (int)nj, (int)nk})
    {
    }

    /**
     * @brief Constructs an Array3D from a 1D vector.
     *
     * @details The use of vector allows explicit bounds checking.
     *
     * @param[in] ni - The number of elements in the first direction.
     * @param[in] nj - The number of elements in the second direction.
     * @param[in] nk - The number of elements in the third direction.
     * @param[in] data_1D - The 1D array vector to wrap.
     */
    Array3D(const unsigned int ni, const unsigned int nj, const unsigned int nk, std::vector<T>& data_1D) :
        m_data_origin(&data_1D[0]),
        m_stride_i(1),
        m_stride_j(ni),
        m_stride_k(ni * nj),
        m_slab({0, 0, 0}, {(int)ni, (int)nj, (int)nk})
    {
        size_t domain_size = ni * nj * nk;
        if (data_1D.size() != domain_size) { throw std::invalid_argument("Size of data_1D is not equal to (ni * nj * nk)"); }
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
        m_stride_i(stride_i),
        m_stride_j(stride_j),
        m_stride_k(stride_k),
        m_slab({0, 0, 0}, {(int)ni, (int)nj, (int)nk})
    {
    }

    /**
     * @brief Sub slab constructor
     *
     * @details Returns a copy of a sub slab of the parent's slab. This is a
     *          shallow copy, meaning the 1D data will not be copied.
     *
     * @param[in] parent - The parent Array3D whose sub slab should be copied.
     * @param[in] sub_slab - The sub slab to be copied from the parent slab.
     */
    Array3D(Array3D<T>& parent, const Slab& sub_slab) :
        m_data_origin(parent.start()),
        m_stride_i(parent.m_stride_i),
        m_stride_j(parent.m_stride_j),
        m_stride_k(parent.m_stride_k),
        m_slab(sub_slab)
    {
        Slab parent_slab = parent.cellSlab();
        for (int axis = 0; axis < 3; axis++)
        {
            if (sub_slab.start(axis) < parent_slab.start(axis) || sub_slab.end(axis) > parent_slab.end(axis))
            {
                throw std::invalid_argument("The requested sub slab is not within the currently defined slab.");
            }
        }
    }

    /**
     * @brief Copy constructor.
     */
    Array3D(Array3D<T>& parent) :
        Array3D<T>(parent, parent.cellSlab())
    {
    }

    /**
     * @brief Get a subsection of this array
     *
     * @details It is likely the data wrapped by the returned Array3D will not
     *          be contiguous.
     *
     * @param[in] sub_slab - A slab defining the sub domain which should be
     *                       returned.
     *
     * @returns An Array3D wrapper around a potentially non-contiguous slab of
     *          cells defined within the this Array3D's current slab.
     */
    Array3D subArray(const Slab& sub_slab)
    { return Array3D<T>(*this, sub_slab); }

    /**
     * @brief access operator to return a reference to the given element.
     *
     * @param i - The index in the first dimension.
     * @param j - The index in the second dimension.
     * @param k - The index in the third dimension.
     *
     * @return Reference to the element at the requested 3D coordinates
     */
    virtual T& operator ()(int i, int j, int k) const
    {
        const unsigned int offset = (i * m_stride_i) + (j * m_stride_j) + (k * m_stride_k);
        return m_data_origin[offset];
    }

    /**
     * @brief Getter for the slab this array represents
     */
    const Slab cellSlab() const
    { return m_slab; }

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

    /**
     * @brief Get a pointer to the origin of the Arrays cell slab.
     */
    double* start() const
    { return &(*this)(m_slab.start(0), m_slab.start(1), m_slab.start(2)); }

  protected:
    T* const m_data_origin;        /**< @brief Pointer to array memory origin */
    const unsigned int m_stride_i; /**< @brief Index stride along the first dimension */
    const unsigned int m_stride_j; /**< @brief Index stride along the second dimension */
    const unsigned int m_stride_k; /**< @brief Index stride along the third dimension */
    const Slab m_slab;
};

#endif  // Array3D_H
