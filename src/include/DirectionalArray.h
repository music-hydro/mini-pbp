/**
 * @class DirectionalArray
 *
 * @brief an array containing scalar elements in every cartesian direction.
 *
 * @details The directions are laid out as {-x, +x, -y, +y, -z, +z}.
 */

#ifndef DirectionalArray_H
#define DirectionalArray_H

#include <array>

enum Direction
{
    NEGATIVE,
    POSITIVE
};

template<typename T>
class DirectionalArray : public std::array<T, 6>
{
  public:
    /**
     * @brief Consistent definition of how to index a DirectionalArray
     *
     * @param[in] axis - The axis to be indexed where i = 0, j = 1 and k = 2.
     * @param[in] direction - The direction of the axis to be indexed.
     *
     * @returns The value stored in the direction of the requested axis.
     */
    T& get(const unsigned int& axis, const Direction& direction)
    { return (*this)[axis * 2U + direction]; }

    /**
     * @copydoc DirectionalArray::get(const unsigned int&, const Direction&)
     */
    const T& get(const unsigned int& axis, const Direction& direction) const
    { return (*this)[axis * 2U + direction]; }
};

#endif  // DirectionalArray_H
