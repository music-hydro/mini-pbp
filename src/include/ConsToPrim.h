/**
 * @file ConsToPrim.h
 *
 * @brief Conversion of the residual from conserved to primitive variables, the first step of the preconditioner. It is
 * a translation of `cons_to_prim` in MUSIC `src/solvers/preconditioner.f90`.
 *
 * - delta_rho = delta_cons_rho
 * - delta_e = (delta_rho_e - e delta_rho) / rho, with e the specific energy (internal or total, as given)
 * - each active scalar: delta_phi = (delta_rho_phi - phi delta_rho) / rho
 * - each velocity component lives on the faces along its own axis: delta_v = (delta_rho_v - v delta_rho_f) / rho_f, with
 *   rho_f and delta_rho_f the average of the two cells next to the face. In 2D the third component is invariant along k
 *   and uses the cell values.
 * - each magnetic field component is copied unchanged.
 *
 * Indices are local to the rank and start from 0. A cell centred field holds cell (i, j, k) at (i, j, k). A field
 * staggered along an axis holds at (i, j, k) the face on the negative side of cell (i, j, k) along that axis; its
 * cellSlab() is the faces solved on the rank (MUSIC `solved_local_domain_nodes`), so it may start at 1 (negative
 * boundary face not solved) or end at n + 1 (positive boundary face solved). Every loop runs over the slab of the primitive delta it writes, as in
 * MUSIC.
 *
 * The face averages of delta rho read one cell beyond the rank. As in MUSIC (`rho_g(ax)%init(plan, cv_rho)`), the
 * conversion fills those cells itself: it wraps delta rho in an Array3DWithGhosts on the given GhostExchangePlan and
 * exchanges the ghosts, so the NEGATIVE ghost along an axis is read at index -1 and the POSITIVE ghost at index n. The
 * ghost exchange is collective over the neighbours in the plan, so every rank of the plan must call consToPrim.
 *
 * The conversion from total to internal energy (MUSIC `add_subtract_ekin_contribution`) is not part of this step.
 */

#ifndef ConsToPrim_H
#define ConsToPrim_H

#include "Array3D.h"
#include "GhostExchangePlan.h"
#include <array>
#include <vector>

namespace ConsToPrim
{
    /**
     * @brief The solution fields cons_to_prim reads, laid out like MUSIC `sol%vars`: one ghost layer on each side of the
     * cells of this rank (none along k in 2D), so cell (i, j, k) of the rank is element (i + 1, j + 1, k + 1) of the
     * view (k + 0 in 2D). A velocity component is stored at the faces on the negative side of the cells, as in MUSIC.
     */
    struct SolutionFields
    {
        const Array3D<double>* rho;                          // Density
        const Array3D<double>* e;                            // Specific energy
        std::array<const Array3D<double>*, 3> vel;           // Velocity components (only the first nmom are read)
        std::vector<const Array3D<double>*> active_scalars;  // Active scalars phi
    };

    /**
     * @brief Deltas of the conserved variables (delta_cons) or of the primitive variables (delta_prim), one view per
     * variable on its own nodes. `Field` is `const Array3D<double>*` for the inputs and `Array3D<double>*` for the
     * outputs.
     */
    template<typename Field>
    struct Deltas
    {
        Field rho;                            // delta rho
        Field e;                              // delta rho e (conserved) or delta e (primitive)
        std::array<Field, 3> vel;             // delta rho v (conserved) or delta v (primitive), first nmom only
        std::vector<Field> active_scalars;    // delta rho phi (conserved) or delta phi (primitive)
        std::array<Field, 3> magnetic_field;  // delta B (only with MHD)
    };

    using ConservedDeltas = Deltas<const Array3D<double>*>;
    using PrimitiveDeltas = Deltas<Array3D<double>*>;

    /**
     * @brief Convert the residual from conserved to primitive variables, as MUSIC `cons_to_prim`.
     *
     * @param[in] ndim - The number of dimensions (2 or 3).
     * @param[in] nmom - The number of velocity components (2 or 3, at least ndim).
     * @param[in] mhd_enabled - Whether the magnetic field components are present.
     * @param[in] solution - The solution fields, with one ghost layer.
     * @param[in] delta_rho_plan - The ghost exchange plan of the cells of this rank (neighbour ranks and boundary
     *            conditions on each face), used to fill delta rho one cell beyond the rank, as MUSIC `rho_plans`.
     * @param[in] delta_cons - The conserved deltas.
     * @param[out] delta_prim - The primitive deltas.
     */
    void consToPrim(const unsigned int ndim,
                    const unsigned int nmom,
                    const bool mhd_enabled,
                    const SolutionFields& solution,
                    GhostExchangePlan<double>& delta_rho_plan,
                    const ConservedDeltas& delta_cons,
                    PrimitiveDeltas& delta_prim);
}  // namespace ConsToPrim

#endif  // ConsToPrim_H
