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
 * staggered along an axis holds at (i, j, k) the face on the left of cell (i, j, k) along that axis; its cellSlab() is
 * the faces solved on the rank (MUSIC `solved_local_domain_nodes`), so it may start at 1 (left boundary face not solved)
 * or end at n + 1 (right boundary face solved). Every loop runs over the slab of the primitive delta it writes, as in
 * MUSIC.
 *
 * The conversion from total to internal energy (MUSIC `add_subtract_ekin_contribution`) is not part of this step.
 */

#ifndef ConsToPrim_H
#define ConsToPrim_H

#include "Array3D.h"
#include <array>
#include <vector>

namespace ConsToPrim
{
    /**
     * @brief The solution fields cons_to_prim reads, laid out like MUSIC `sol%vars`: one ghost layer on each side of the
     * cells of this rank (none along k in 2D), so cell (i, j, k) of the rank is element (i + 1, j + 1, k + 1) of the
     * view (k + 0 in 2D). A velocity component is stored at the faces on the left of the cells, as in MUSIC.
     */
    struct SolutionFields
    {
        const Array3D<double>* rho;                         // Density
        const Array3D<double>* e;                           // Specific energy
        std::array<const Array3D<double>*, 3> vel;          // Velocity components (only the first nmom are read)
        std::vector<const Array3D<double>*> activeScalars;  // Active scalars phi
    };

    /**
     * @brief Deltas of the conserved variables (delta_cons) or of the primitive variables (delta_prim), one view per
     * variable on its own nodes. `Field` is `const Array3D<double>*` for the inputs and `Array3D<double>*` for the
     * outputs.
     */
    template<typename Field>
    struct Deltas
    {
        Field rho;                           // delta rho
        Field e;                             // delta rho e (conserved) or delta e (primitive)
        std::array<Field, 3> vel;            // delta rho v (conserved) or delta v (primitive), first nmom only
        std::vector<Field> activeScalars;    // delta rho phi (conserved) or delta phi (primitive)
        std::array<Field, 3> magneticField;  // delta B (only with MHD)
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
     * @param[in] delta_rho_with_ghosts - For each axis below ndim, delta rho readable one cell beyond each end of the rank
     *            along that axis (index -1 and n), as MUSIC `rho_g(ax)%value_at`: the halo from the neighbour rank, or the
     *            boundary condition fill. An Array3DWithGhosts filled by the ghost exchange can be given here.
     * @param[in] delta_cons - The conserved deltas.
     * @param[out] delta_prim - The primitive deltas.
     */
    void consToPrim(const unsigned int ndim,
                    const unsigned int nmom,
                    const bool mhd_enabled,
                    const SolutionFields& solution,
                    const std::array<const Array3D<double>*, 3>& delta_rho_with_ghosts,
                    const ConservedDeltas& delta_cons,
                    PrimitiveDeltas& delta_prim);
}  // namespace ConsToPrim

#endif  // ConsToPrim_H
