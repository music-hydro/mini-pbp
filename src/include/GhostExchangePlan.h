/**
 * @class GhostExchangePlan
 *
 * @brief Defines the plan for exchanging ghost cells between MPI ranks.
 */

#ifndef GhostExchangePlan_H
#define GhostExchangePlan_H

#include "Array3D.h"
#include <array>

enum BoundaryConditions
{
    ZERO_VALUE,
    ZERO_DERIVATIVE,
    PERIODIC,
    NONE
};

template<typename T>
class GhostExchangePlan
{
  public:
    /**
     * @brief Copy constructor
     */
    GhostExchangePlan(GhostExchangePlan<T>& parent) :
        m_array(parent.array()),
        m_boundary_conditions(parent.boundary_conditions()),
        m_neighbours(parent.neighbours())
    {
    }

    /**
     * @brief Constructor
     *
     * @details `boundary_conditions` and `neighbours` should have the
     *          following structure:
     *              {i_left, i_right, j_left, j_right, k_left, k_right}
     *
     * @param[in] array - The 3D view of the array without ghost cells.
     * @param[in] boundary_conditions - The boundary conditions.
     * @param[in] neighbours - The neighbouring MPI ranks for the current process.
     */
    GhostExchangePlan(Array3D<T>& array,
                      std::array<BoundaryConditions, 6> boundary_conditions,
                      std::array<unsigned int, 6> neighbours) :
        m_array(array),
        m_boundary_conditions(boundary_conditions),
        m_neighbours(neighbours)
    {
    }

    /**
     * @brief Get the array for which this plan has been created.
     */
    Array3D<T> array() const
    { return m_array; }

    /**
     * @brief Get the boundary conditions for this plan.
     */
    std::array<BoundaryConditions, 6> boundary_conditions() const
    { return m_boundary_conditions; }

    /**
     * @brief Get the neighbours for this plan.
     */
    std::array<unsigned int, 6> neighbours() const
    { return m_neighbours; }

  private:
    Array3D<T> m_array;
    std::array<BoundaryConditions, 6> m_boundary_conditions;
    std::array<unsigned int, 6> m_neighbours;
};

#endif  // GhostExchangePlan_H
