/**
 * @class Slab
 *
 * @brief Provides a consistent interface to a 3D range of cell indices
 */

#ifndef Slab_H
#define Slab_H

#include <array>

class Slab
{
  public:
    /**
     * @brief Constructor
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
    const int start(int dimension) const
    { return m_start[dimension]; }

    /**
     * @brief Read only access to the end of the slabs range
     *
     * @param dimension - An index representing the dimension (i, j or k) who's end should be returned
     *
     * @return The index begining the range for the given `dimension`
     */
    const int end(int dimension) const
    { return m_end[dimension]; }

  private:
    /// The ranges over which this slab defines
    const std::array<int, 3> m_start;
    const std::array<int, 3> m_end;
};

#endif  // Slab_H
