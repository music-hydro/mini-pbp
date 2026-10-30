/**
 * @class Slab
 *
 * @brief Provides a consistent interface to a 3D range of cell indices
 */

#ifndef Slab_H
#define Slab_H

#include <array>
#include <stdexcept>

class Slab
{
  public:
    /**
     * @brief Constructor
     *
     * @details start and end are signed ints to allow for custom bounds.
     *
     * @param start - Array of the starting indices for each axis
     * @param end - Array of the starting indices for each axis
     */
    Slab(const std::array<int, 3> start, const std::array<int, 3> end) :
        m_start(start),
        m_end(end)
    {
    }

    /**
     * @brief Read only access to the start of the slabs range
     *
     * @param dimension - An index representing the dimension (i, j or k) who's start should be returned
     *
     * @return The index begining the range for the given `dimension`
     */
    int start(const unsigned int dimension) const
    { return m_start[dimension]; }

    /**
     * @brief Read only access to the end of the slabs range
     *
     * @param dimension - An index representing the dimension (i, j or k) who's end should be returned
     *
     * @return The index begining the range for the given `dimension`
     */
    int end(const unsigned int dimension) const
    { return m_end[dimension]; }

    /**
     * @brief Get the full start array for this slab.
     */
    std::array<int, 3> start() const
    { return m_start; }

    /**
     * @brief Get the full end array for this slab.
     */
    std::array<int, 3> end() const
    { return m_end; }

    /**
     * @brief Get the size of the domain along the given `axis`.
     *
     * @param[in] axis - The axis who's size should be returned.
     *
     * @returns The size of the domain along the given `axis`.
     */
    unsigned int extent(unsigned int axis) const
    {
        if (axis > 2) throw std::invalid_argument("Invalid axis: " + std::to_string(axis));
        return end(axis) - start(axis);
    }

  private:
    /// The ranges over which this slab defines
    const std::array<int, 3> m_start;
    const std::array<int, 3> m_end;
};

#endif  // Slab_H
