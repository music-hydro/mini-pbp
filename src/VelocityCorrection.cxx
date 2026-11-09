#include "VelocityCorrection.h"

#include "Array3DWithGhosts.h"
#include "Slab.h"
#include <mpi.h>
#include <stdexcept>

namespace
{
    /**
     * @brief Element of a solution field with one ghost layer at local position (i, j, k).
     *
     * @param[in] k_ghosts - The number of ghost layers along k (1 in 3D, 0 in 2D).
     */
    double ghosted(const Array3D<double>& field, const int i, const int j, const int k, const int k_ghosts)
    { return field(i + 1, j + 1, k + k_ghosts); }
}  // namespace

void VelocityCorrection::correctVelocity(const unsigned int ndim,
                                         const double dt,
                                         const Array3D<double>& rho,
                                         GhostExchangePlan<double>& delta_pressure_plan,
                                         const Array3D<double>& delta_pressure,
                                         const std::array<const Array3D<double>*, 3>& face_span,
                                         const std::array<Array3D<double>*, 3>& delta_vel)
{
    if (ndim != 2 && ndim != 3) { throw std::invalid_argument("velocity correction: ndim must be 2 or 3"); }
    const int kg = (ndim == 3) ? 1 : 0;

    // delta pressure one cell beyond the rank: the halo from the neighbour ranks, or the boundary condition fill
    Array3DWithGhosts<double> delta_pressure_g(delta_pressure, delta_pressure_plan);
    delta_pressure_g.exchangeGhosts();

    const std::array<int, 6> neighbours = delta_pressure_plan.neighbours();

    // The pressure gradient has no component along the axes beyond ndim
    for (unsigned int ax = 0; ax < ndim; ax++)
    {
        Array3D<double>& delta_v    = *delta_vel[ax];
        const Array3D<double>& span = *face_span[ax];
        const Slab s                = delta_v.cellSlab();
        // Neighbour cell on the negative side of the face along ax
        const int di = (ax == 0) ? 1 : 0, dj = (ax == 1) ? 1 : 0, dk = (ax == 2) ? 1 : 0;

        // Correct velocity component with pressure gradient along ax: delta_v <- delta_v - dt * grad(delta_P) / rho_f
        for (int k = s.start(2); k < s.end(2); k++)
        {
            for (int j = s.start(1); j < s.end(1); j++)
            {
                for (int i = s.start(0); i < s.end(0); i++)
                {
                    const double grad_p_f =
                        (delta_pressure_g(i, j, k) - delta_pressure_g(i - di, j - dj, k - dk)) / span(i, j, k);
                    const double rho_f = 0.5 * (ghosted(rho, i - di, j - dj, k - dk, kg) + ghosted(rho, i, j, k, kg));
                    delta_v(i, j, k) += -dt * grad_p_f / rho_f;
                }
            }
        }

        // Set the velocity delta to 0 on the negative domain boundary face, where the boundary conditions pin the normal
        // velocity and so its perturbation. That face is face 0 along ax, if this rank solves it.
        if (neighbours[2 * ax + Direction::NEGATIVE] == MPI_PROC_NULL && s.start(ax) == 0)
        {
            std::array<int, 3> start = s.start(), end = s.end();
            end[ax] = 1;
            for (int k = start[2]; k < end[2]; k++)
                for (int j = start[1]; j < end[1]; j++)
                    for (int i = start[0]; i < end[0]; i++)
                        delta_v(i, j, k) = 0.0;
        }
    }
}
