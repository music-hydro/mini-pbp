/**
 * @file  PackedVariables
 *
 * @details Classes for storing packed variables arrays, i.e. a single contiguous 1D array
 *          where the values for each variable is stored contiguously in (i, j, k), i being
 *          fastest. i.e. Iterating through the 1D array would consume a single variable in
 *          its entirety. Once this variable has been consumed the next variable begins in
 *          an unbroken chain.
 */

#ifndef PackedVariables_H
#define PackedVariables_H

#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "GhostExchangePlan.h"
#include <vector>

/**
 * @class BasePackedVariables
 *
 * @brief A base class for storing packed variable arrays.
 */
template<class T>
class BasePackedVariables
{
  public:
    /**
     * @brief Constructor
     *
     * @param packed_array_pointer - Pointer to the full 1D array containing all variables
     * @param ni - The number of physical cells in the first direction.
     * @param nj - The number of physical cells in the second direction.
     * @param nk - The number of physical cells in the third direction.
     */
    BasePackedVariables(std::vector<double>& packed_array_pointer,
                        const unsigned int ni,
                        const unsigned int nj,
                        const unsigned int nk) :
        m_packed_array(&packed_array_pointer[0]),
        m_ni(ni),
        m_nj(nj),
        m_nk(nk)
    {
        // Verify the size of solutions
        unsigned long long domain_size = m_ni * m_nj * m_nk;
        if (packed_array_pointer.size() % domain_size != 0)
        {
            throw std::invalid_argument("Size of packed_array_pointer is not a multiple of (ni * nj * nk)");
        }
        m_num_variables = (packed_array_pointer.size() / domain_size);
    }

    /**
     * @brief Return the full 3D array view for a chosen variable
     *
     * @param variable_index - The variable whose 3D array is to be extracted.
     *
     * @return An Array3D instance providing a 3D view of the requested variable's 1D array
     */
    T singleVariableView(const unsigned int variable_index) const
    {
        // Validate requested variable index against what is assumed
        if (variable_index >= m_num_variables)
        {
            throw std::invalid_argument("Provided variable index not found: " + std::to_string(variable_index));
        }

        return *m_packed_arrays[variable_index];
    }

    /**
     * @brief Return the number of variables
     */
    unsigned int numVariables()
    { return m_num_variables; }

    /**
     * @brief Get the domain size in the ith direction.
     */
    unsigned int ni()
    { return m_ni; }

    /**
     * @brief Get the domain size in the jth direction.
     */
    unsigned int nj()
    { return m_nj; }

    /**
     * @brief Get the domain size in the kth direction.
     */
    unsigned int nk()
    { return m_nk; }

  protected:
    /// 3D views of the larger 1D array
    std::vector<std::shared_ptr<T>> m_packed_arrays;
    /// Pointer to origin of the 3D view in the larger 1D array
    double* const m_packed_array;
    /// The lengths of the physical cell dimensions
    const unsigned int m_ni;
    const unsigned int m_nj;
    const unsigned int m_nk;
    /// The number of variables stored in the packed array
    unsigned int m_num_variables;
};

/**
 * @class PackedVariablesWithGhosts
 *
 * @brief A class for storing packed variable arrays with ghost arrays.
 */
class PackedVariablesWithGhosts : public BasePackedVariables<Array3DWithGhosts<double>>
{
  public:
    /**
     * @brief Constructor
     *
     * @param packed_array_pointer - Pointer to the full 1D array containing all variables
     * @param ni - The number of physical cells in the first direction.
     * @param nj - The number of physical cells in the second direction.
     * @param nk - The number of physical cells in the third direction.
     * @param plan - The plan for exchanging ghosts.
     */
    PackedVariablesWithGhosts(std::vector<double>& packed_array_pointer,
                              const unsigned int ni,
                              const unsigned int nj,
                              const unsigned int nk,
                              GhostExchangePlan<double> plan) :
        BasePackedVariables<Array3DWithGhosts<double>>(packed_array_pointer, ni, nj, nk)
    {
        size_t domain_size = ni * nj * nk;
        // Pre-allocate ghost arrays
        for (unsigned int ivar = 0; ivar < m_num_variables; ivar++)
        {
            Array3D<double> new_inner_array(m_ni, m_nj, m_nk, &packed_array_pointer[domain_size * ivar]);
            std::shared_ptr<Array3DWithGhosts<double>> new_array_with_ghosts =
                std::make_shared<Array3DWithGhosts<double>>(new_inner_array, plan);
            m_packed_arrays.push_back(new_array_with_ghosts);
        }
    }
};

/**
 * @class PackedVariables
 *
 * @brief A class for storing packed variable arrays without ghost arrays.
 */
class PackedVariables : public BasePackedVariables<Array3D<double>>
{
  public:
    /**
     @copydoc BasePackedVariables::BasePackedVariables
     */
    PackedVariables(std::vector<double>& packed_array_pointer,
                    const unsigned int ni,
                    const unsigned int nj,
                    const unsigned int nk) :
        BasePackedVariables<Array3D<double>>(packed_array_pointer, ni, nj, nk)
    {
        size_t domain_size = ni * nj * nk;
        // Pre-allocate ghost arrays
        for (unsigned int ivar = 0; ivar < m_num_variables; ivar++)
        {
            std::shared_ptr<Array3D<double>> new_inner_array =
                std::make_shared<Array3D<double>>(m_ni, m_nj, m_nk, &packed_array_pointer[domain_size * ivar]);
            m_packed_arrays.push_back(new_inner_array);
        }
    }
};

#endif  // PackedVariables_H
