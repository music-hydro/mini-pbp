#include "ConsToPrim.h"

#include "Array3DWithGhosts.h"
#include "Slab.h"
#include <stdexcept>
#include <string>

namespace
{
    /**
     * @brief Element of a solution field with one ghost layer at local position (i, j, k).
     *
     * @param[in] k_ghosts - The number of ghost layers along k (1 in 3D, 0 in 2D).
     */
    double ghosted(const Array3D<double>& field, const int i, const int j, const int k, const int k_ghosts)
    { return field(i + 1, j + 1, k + k_ghosts); }

    /**
     * @brief Throw if two views do not cover the same nodes (MUSIC `ASSERT(a%nodes == b%nodes)`).
     */
    void requireSameSlab(const Array3D<double>& a, const Array3D<double>& b, const std::string& what)
    {
        const Slab sa = a.cellSlab(), sb = b.cellSlab();
        if (sa.start() != sb.start() || sa.end() != sb.end())
        {
            throw std::invalid_argument("cons_to_prim: " + what + " do not cover the same nodes");
        }
    }
}  // namespace

void ConsToPrim::consToPrim(const unsigned int ndim,
                            const unsigned int nmom,
                            const bool mhd_enabled,
                            const SolutionFields& solution,
                            GhostExchangePlan<double>& delta_rho_plan,
                            const ConservedDeltas& delta_cons,
                            PrimitiveDeltas& delta_prim)
{
    if (ndim != 2 && ndim != 3) { throw std::invalid_argument("cons_to_prim: ndim must be 2 or 3"); }
    if (nmom != 2 && nmom != 3) { throw std::invalid_argument("cons_to_prim: nmom must be 2 or 3"); }
    if (nmom < ndim) { throw std::invalid_argument("cons_to_prim: nmom must be at least ndim"); }
    if (solution.active_scalars.size() != delta_cons.active_scalars.size() ||
        solution.active_scalars.size() != delta_prim.active_scalars.size())
    {
        throw std::invalid_argument("cons_to_prim: the number of active scalars differs between the inputs");
    }
    const int kg = (ndim == 3) ? 1 : 0;

    const Array3D<double>& sol_rho = *solution.rho;
    const Array3D<double>& sol_e   = *solution.e;
    const Array3D<double>& cv_rho  = *delta_cons.rho;

    // delta rho one cell beyond the rank: the halo from the neighbour ranks, or the boundary condition fill
    Array3DWithGhosts<double> delta_rho_g(cv_rho, delta_rho_plan);
    delta_rho_g.exchangeGhosts();

    // Density
    requireSameSlab(*delta_prim.rho, cv_rho, "delta_prim rho and delta_cons rho");
    cv_rho.deepCopy(*delta_prim.rho);

    // Energy
    requireSameSlab(*delta_prim.e, cv_rho, "delta_prim e and delta_cons rho");
    requireSameSlab(*delta_prim.e, *delta_cons.e, "delta_prim e and delta_cons rho e");
    {
        Array3D<double>& delta_e           = *delta_prim.e;
        const Array3D<double>& delta_rho_e = *delta_cons.e;
        const Slab s                       = delta_e.cellSlab();
        for (int k = s.start(2); k < s.end(2); k++)
        {
            for (int j = s.start(1); j < s.end(1); j++)
            {
                for (int i = s.start(0); i < s.end(0); i++)
                {
                    const double e   = ghosted(sol_e, i, j, k, kg);
                    const double rho = ghosted(sol_rho, i, j, k, kg);
                    delta_e(i, j, k) = (-e * cv_rho(i, j, k) + delta_rho_e(i, j, k)) / rho;
                }
            }
        }
    }

    // Active scalars
    for (size_t isc = 0; isc < solution.active_scalars.size(); isc++)
    {
        const Array3D<double>& sol_phi       = *solution.active_scalars[isc];
        const Array3D<double>& delta_rho_phi = *delta_cons.active_scalars[isc];
        Array3D<double>& delta_phi           = *delta_prim.active_scalars[isc];
        requireSameSlab(delta_phi, cv_rho, "delta_prim scalar and delta_cons rho");
        requireSameSlab(delta_phi, delta_rho_phi, "delta_prim scalar and delta_cons rho scalar");
        const Slab s = delta_phi.cellSlab();
        for (int k = s.start(2); k < s.end(2); k++)
        {
            for (int j = s.start(1); j < s.end(1); j++)
            {
                for (int i = s.start(0); i < s.end(0); i++)
                {
                    const double rho   = ghosted(sol_rho, i, j, k, kg);
                    const double phi   = ghosted(sol_phi, i, j, k, kg);
                    delta_phi(i, j, k) = (delta_rho_phi(i, j, k) - phi * cv_rho(i, j, k)) / rho;
                }
            }
        }
    }

    // Velocity: each component lives on the faces along its own axis, so convert it over that component's own nodes
    // rather than over the cells slab, which may omit a solved boundary face. Averaging delta rho to the first face, or
    // to a solved positive boundary face, reads the NEGATIVE or POSITIVE ghost of delta_rho_g.
    for (unsigned int ax = 0; ax < nmom; ax++)
    {
        Array3D<double>& delta_v           = *delta_prim.vel[ax];
        const Array3D<double>& delta_rho_v = *delta_cons.vel[ax];
        const Array3D<double>& sol_v       = *solution.vel[ax];
        const Slab s                       = delta_v.cellSlab();
        // Neighbour cell on the negative side of the face along ax
        const int di = (ax == 0) ? 1 : 0, dj = (ax == 1) ? 1 : 0, dk = (ax == 2) ? 1 : 0;
        for (int k = s.start(2); k < s.end(2); k++)
        {
            for (int j = s.start(1); j < s.end(1); j++)
            {
                for (int i = s.start(0); i < s.end(0); i++)
                {
                    double rho_f, delta_rho_f;
                    if (ax < ndim)
                    {
                        rho_f       = 0.5 * (ghosted(sol_rho, i, j, k, kg) + ghosted(sol_rho, i - di, j - dj, k - dk, kg));
                        delta_rho_f = 0.5 * (delta_rho_g(i - di, j - dj, k - dk) + delta_rho_g(i, j, k));
                    }
                    else
                    {
                        // Invariant along axis k if ndim < 3
                        rho_f       = ghosted(sol_rho, i, j, k, kg);
                        delta_rho_f = cv_rho(i, j, k);
                    }
                    const double v   = ghosted(sol_v, i, j, k, kg);
                    delta_v(i, j, k) = (-v * delta_rho_f + delta_rho_v(i, j, k)) / rho_f;
                }
            }
        }
    }

    // Magnetic field: the conversion is the identity. Copy over each component's own nodes rather than over the cells
    // slab: a component staggered along an axis may own a boundary face that the cells slab does not cover.
    if (mhd_enabled)
    {
        for (unsigned int ax = 0; ax < 3; ax++)
        {
            requireSameSlab(*delta_prim.magnetic_field[ax], *delta_cons.magnetic_field[ax], "delta_prim B and delta_cons B");
            delta_cons.magnetic_field[ax]->deepCopy(*delta_prim.magnetic_field[ax]);
        }
    }
}
