/**
 * @file VelocityCorrection.h
 *
 * @brief Correction of the velocity deltas for the pressure gradient, the step of the preconditioner that follows the
 * pressure solve. It is a translation of the section "Correct velocity components for pressure gradient" of `apply` in
 * MUSIC `src/solvers/preconditioner.f90`.
 *
 * For each axis ax < ndim, on every face of the velocity component along ax:
 *
 *     delta_v = delta_v - dt * grad_ax(delta_P) / rho_f
 *
 * with grad_ax(delta_P) = (delta_P(i) - delta_P(i - 1)) / span, the difference of the two cells next to the face over
 * the span of the face (MUSIC `grid%eval_s_grad_comp`, the grid spacing on a uniform Cartesian grid), and rho_f the
 * average of rho in the same two cells (MUSIC `density%eval_at` on the faces). Velocity components along axes beyond ndim
 * are left unchanged, since the pressure gradient has no component there.
 *
 * Where the negative face of the rank along ax is a domain boundary (no neighbour rank) and the rank solves it, the
 * velocity delta on that face is set to 0: the boundary conditions pin the normal velocity there, and so its
 * perturbation.
 *
 * Indices are local to the rank and start from 0. A cell centred field holds cell (i, j, k) at (i, j, k). The velocity
 * component along an axis holds at (i, j, k) the face on the negative side of cell (i, j, k) along that axis; its
 * cellSlab() is the faces solved on the rank, so it may start at 1 or end at n + 1. Each loop runs over that slab, as
 * in MUSIC.
 *
 * The gradient on the first face, or on a solved positive boundary face, reads delta pressure one cell beyond the rank.
 * As in MUSIC (`delta_press_g%init(plan, delta_press)`), the correction fills those cells itself: it wraps delta
 * pressure in an Array3DWithGhosts on the given GhostExchangePlan and exchanges the ghosts. The ghost exchange is
 * collective over the neighbours in the plan, so every rank of the plan must call correctVelocity.
 */

#ifndef VelocityCorrection_H
#define VelocityCorrection_H

#include "Array3D.h"
#include "GhostExchangePlan.h"
#include <array>

namespace VelocityCorrection
{
    /**
     * @brief Correct the velocity deltas for the pressure gradient, as MUSIC `apply`.
     *
     * @param[in] ndim - The number of dimensions (2 or 3).
     * @param[in] dt - The time step.
     * @param[in] rho - The solution density laid out like MUSIC `sol%vars`: one ghost layer on each side of the cells of
     *            this rank (none along k in 2D), so cell (i, j, k) of the rank is element (i + 1, j + 1, k + 1) of the
     *            view (k + 0 in 2D).
     * @param[in] delta_pressure_plan - The ghost exchange plan of the cells of this rank (neighbour ranks and boundary
     *            conditions on each face), as MUSIC `prob%plans%delta_press`.
     * @param[in] delta_pressure - The pressure delta from the solve on the cells of this rank, a whole array (not a sub
     *            array) starting at cell 0.
     * @param[in] face_span - For each axis ax < ndim, the span of each face of the velocity component along ax, laid out
     *            like that component.
     * @param[in,out] delta_vel - The velocity deltas, one per axis (only the first ndim are read and corrected).
     */
    void correctVelocity(const unsigned int ndim,
                         const double dt,
                         const Array3D<double>& rho,
                         GhostExchangePlan<double>& delta_pressure_plan,
                         const Array3D<double>& delta_pressure,
                         const std::array<const Array3D<double>*, 3>& face_span,
                         const std::array<Array3D<double>*, 3>& delta_vel);
}  // namespace VelocityCorrection

#endif  // VelocityCorrection_H
