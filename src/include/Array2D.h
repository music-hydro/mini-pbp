/**
 * @class Array2D
 *
 * @brief Provides a 2D view on a 1D array with custom bounds
 */

#ifndef Array2D_H
#define Array2D_H

#include "Array3D.h"

template<typename T>
class Array2D : public Array3D<T>
{
  public:
    /**
     * @brief Constructs an Array2D from array bounds alone.
     *
     * @details the wrapped 1D data will be allocated by the constructor.
     *
     * @param[in] ax1 - The number of elements in the fastest direction.
     * @param[in] ax2 - The number of elements in the second direction.
     */
    Array2D(const unsigned int& ax1, const unsigned int& ax2) :
        Array3D<T>(ax1, ax2, 1)
    {
    }

    /**
     * @brief Copy constructor
     */
    Array2D(Array2D<T>& parent) :
        Array3D<T>(parent, parent.cellSlab())
    {
    }

    /**
     * @brief access operator to return a reference to the given element.
     *
     * @param idx1 - The index in the first dimension.
     * @param idx2 - The index in the second dimension.
     *
     * @return Reference to the element at the requested 2D coordinates
     */
    T& operator ()(int idx1, int idx2) const
    { return this->Array3D<T>::operator ()(idx1, idx2, 0); }
};

#endif  // Array2D_H
