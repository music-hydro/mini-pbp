/**
 * @class PhysicsBasedPreconditioner
 *
 * @brief A class containing the core preconditioning logic.
 */

#ifndef PhysicsBasedPreconditioner_H
#define PhysicsBasedPreconditioner_H

#include "Array3D.h"
#include "Array3DWithGhosts.h"
#include "DirectionalArray.h"
#include "PackedVariables.h"

enum SolutionVariableIndices
{
  RHO,
  PRESSURE,
  ENERGY
};

enum SolutionDpdVariableIndices
{
  DPDRHO,
  DPDENERGY
};

class PhysicsBasedPreconditioner
{
  public:
    /**
     * @class Constructor
     *
     * @param[in] solution - The solution containing 3 packed variables (rho, then pressure and then energy)
     * @param[in] solution_dPd - The solution gradients containing 2 packed variables (dPd_rho and then dPd_energy)
     * @param[in] squared_sound_speed - 3D view of the squared sound speed within each each cell.
     * @param[in] prec_nvar - The number of variables to precondition (1: just pressure, 2: pressure and energy)
     */
    PhysicsBasedPreconditioner(PackedVariables& solution,
                               PackedVariables& solution_dPd,
                               Array3D<double>& squared_sound_speed,
                               const unsigned int prec_nvar = 1);

    /**
     * @brief Set the RHS pressure
     *
     * @param[in]  delta_rho - 3D view of the density delta.
     * @param[in]  delta_energy - 3D view of the energy delta.
     * @param[out] rhs_pressure - 3D view of the RHS pressure delta to be set.
     * @param[out] rhs_energy - 3D view of the RHS energy delta to be set. Only set uf prec_nvar is 2.
     */
    void setPressureRHS(const Array3D<double>& delta_rho,
                        const Array3D<double>& delta_energy,
                        Array3D<double>& rhs_pressure,
                        Array3D<double>& rhs_energy);

    /**
     * @brief Reconstruct the energy delta for the next timestep after preconditioning.
     *
     * @param[in]  preconditioned_delta_pressure - 3D view of pressure delta calculated in the preconditioner pressure solve.
     * @param[in]  preconditioned_delta_energy - 3D view of energy delta calculated in the preconditioner pressure solve.
     * @param[in]  dp_rhs_press - 3D view of the pressure array set by setPressureRHS.
     * @param[out] delta_energy - 3D View of the energy delta to be calculated.
     */
    void reconstructEnergyDelta(const Array3D<double>& preconditioned_delta_pressure,
                                const Array3D<double>& preconditioned_delta_energy,
                                const Array3D<double>& dp_rhs_press,
                                Array3D<double>& delta_energy);


    /**
     * @brief Reconstruct the density delta for the next timestep after preconditioning.
     *
     * @param[in]  delta_pressure - 3D View of the pressure delta.
     * @param[in]  delta_energy - 3D View of the energy delta.
     * @param[out] delta_rho - 3D view of the density delta to be calculated.
     */
    void reconstructDensityDelta(const Array3D<double>& delta_pressure,
                                 const Array3D<double>& delta_energy,
                                 Array3D<double>& delta_rho);


    /**
     * @brief Assembles the pressure matrix ready for preconditioning
     *
     * @param[in,out] pressure - The pressure matrix to be assembled.
     */
    void assemblePressureMatrix(Array3DWithGhosts<double>& pressure,
                                const Array3D<DirectionalArray<double>>& laplacian_coefficients);

    PackedVariables solution() const
    { return m_solution; }

  private:
    /**
     * @brief Compute squared sound speed from the solution variables.
     *
     * @details For now this is set using some data from MUSIC (should be calculated instead)
     */
    void calculateSquaredSoundSpeed();

    /**
     * @brief Calculate the change in energy for the timestep next timestep.
     *
     * @details Energy equation, we do not consider heat conduction here
     *            A_ee = 1/dt and A_pe = -dt*p/rho^2*L(dp)
     *            bis: A_dd = 1/dt and A_pd = -dt*L(dp)
     *
     * @param[in]  preconditioned_delta_pressure - 3D view of pressure delta calculated in the preconditioner pressure solve.
     * @param[in]  dp_rhs_press - 3D view of the pressure array set by setPressureRHS.
     * @param[out] delta_energy - 3D View of the energy delta to be calculated.
     */
    void calculateEnergyDelta(const Array3D<double>& preconditioned_delta_pressure,
                              const Array3D<double>& dp_rhs_press,
                              Array3D<double>& delta_energy);

    /**
     * @brief Get the pressure coefficients for a given cell
     *
     * @param i - The index in the first dimension.
     * @param j - The index in the second dimension.
     * @param k - The index in the third dimension.
     */
    DirectionalArray<double> pressureCoefficients(const int& i,
                                                  const int& j,
                                                  const int& k,
                                                  const DirectionalArray<double>& laplacian_coefficients,
                                                  const DirectionalArray<double>& weights);

    /**
     * @brief Get the index of a variable value at a a specific cell centre.
     *
     * @param[in] i - The cell index in the first dimension.
     * @param[in] j - The cell index in the second dimension.
     * @param[in] k - The cell index in the third dimension.
     * @param[in] var_index - The index of the variable to be accessed.
     * @param[in] global_slab - Slab representing the global domain.
     */
    unsigned long long globalVarIndex(const int& i, const int& j, const int& k, const int& var_index, const Slab& global_slab)
    { return i + global_slab.end(0) * (j + global_slab.end(1) * (k + global_slab.end(2) * var_index)); }

    /// @brief Store of the solution
    PackedVariables m_solution;
    /// @brief Store of the solution gradients
    PackedVariables m_solution_dPd;
    /// @brief Squared sound speed
    Array3D<double> m_squared_sound_speed;
    /// @brief Pressure matrix to be sent to solver
    std::vector<double> m_pressure_matrix_for_solver;
    /// @brief Map of global column indices for m_pressure_matrix_for_solver
    std::vector<long long> m_global_pressure_column_indices;
    /// @brief Map of global row indices for m_pressure_matrix_for_solver
    std::vector<long long> m_global_pressure_row_indices;
    /// @brief Flag to indicate whether to precondition the pressure equation only (`prec_nvar=1`), or both pressure and energy equations (`prec_nvar=2`)
    unsigned int m_prec_nvar;
    // This is a temporary solution. The value has been taken from a run of music and should be calculated instead.
    double m_dt = 1.0917241379310345E-004; /** timestep */
};

#endif  // PhysicsBasedPreconditioner_H
