#include "Array3D.h"
#include "ConsToPrim.h"
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
    using Function = std::function<double(int, int, int)>;

    const double NaN = std::numeric_limits<double>::quiet_NaN();

    /**
     * @brief A solution field with one ghost layer (none along k in 2D), filled from f at the local positions.
     */
    struct GhostedField
    {
        GhostedField(const std::array<int, 3> nc, const int kg, const Function& f) :
            data(static_cast<size_t>(nc[0] + 2) * (nc[1] + 2) * (nc[2] + 2 * kg)),
            view(nc[0] + 2, nc[1] + 2, nc[2] + 2 * kg, data)
        {
            for (int k = -kg; k < nc[2] + kg; k++)
                for (int j = -1; j < nc[1] + 1; j++)
                    for (int i = -1; i < nc[0] + 1; i++)
                        view(i + 1, j + 1, k + kg) = f(i, j, k);
        }

        std::vector<double> data;
        Array3D<double> view;
    };

    /**
     * @brief A delta field on the nodes [start, end) of the rank, stored from node 0 so that the view keeps the local
     * indices. Nodes outside [start, end) hold NaN.
     */
    struct NodeField
    {
        NodeField(const std::array<int, 3> start, const std::array<int, 3> end, const Function& f = nullptr) :
            data(static_cast<size_t>(end[0]) * end[1] * end[2], NaN),
            parent(std::make_unique<Array3D<double>>(end[0], end[1], end[2], data)),
            view(std::make_unique<Array3D<double>>(*parent, Slab(start, end)))
        {
            if (f)
            {
                for (int k = start[2]; k < end[2]; k++)
                    for (int j = start[1]; j < end[1]; j++)
                        for (int i = start[0]; i < end[0]; i++)
                            (*view)(i, j, k) = f(i, j, k);
            }
        }

        std::vector<double> data;
        std::unique_ptr<Array3D<double>> parent;
        std::unique_ptr<Array3D<double>> view;
    };

    /**
     * @brief delta rho readable one cell beyond each end of the rank along every axis (none along k in 2D), as an
     * Array3DWithGhosts filled by the ghost exchange would be. Elements that were never set hold NaN.
     */
    class HaloView : public Array3D<double>
    {
      public:
        HaloView(const std::array<int, 3> nc, const int kg) :
            HaloView(nc,
                     kg,
                     std::make_shared<std::vector<double>>(static_cast<size_t>(nc[0] + 2) * (nc[1] + 2) * (nc[2] + 2 * kg), NaN))
        {
        }

        double& operator ()(int i, int j, int k) const override
        {
            if (i < -1 || i > m_nc[0] || j < -1 || j > m_nc[1] || k < -m_kg || k >= m_nc[2] + m_kg)
            {
                throw std::out_of_range("HaloView: (" + std::to_string(i) + ", " + std::to_string(j) + ", " + std::to_string(k) +
                                        ") is outside the rank and its ghosts");
            }
            return (*m_data)[(i + 1) + (m_nc[0] + 2) * ((j + 1) + (m_nc[1] + 2) * (k + m_kg))];
        }

        void fill(const Function& f)
        {
            for (int k = -m_kg; k < m_nc[2] + m_kg; k++)
                for (int j = -1; j < m_nc[1] + 1; j++)
                    for (int i = -1; i < m_nc[0] + 1; i++)
                        (*this)(i, j, k) = f(i, j, k);
        }

      private:
        HaloView(const std::array<int, 3> nc, const int kg, std::shared_ptr<std::vector<double>> data) :
            Array3D<double>(nc[0], nc[1], nc[2], data->data()),
            m_nc(nc),
            m_kg(kg),
            m_data(data)
        {
        }

        std::array<int, 3> m_nc;
        int m_kg;
        std::shared_ptr<std::vector<double>> m_data;
    };

    bool close(const double a, const double b)
    { return std::abs(a - b) <= 1e-14 * std::max(std::abs(a), std::abs(b)); }

    /**
     * @brief A small analytic problem: every field is a different function of the position, so a wrong index shows.
     */
    struct Problem
    {
        Problem(const unsigned int ndim_, const unsigned int nmom_, const std::array<int, 3> nc_, const unsigned int nscalars) :
            ndim(ndim_),
            nmom(nmom_),
            nc(nc_),
            kg(ndim_ == 3 ? 1 : 0),
            rho(nc_, kg, [](int i, int j, int k) { return 1.0 + 0.10 * i + 0.07 * j + 0.05 * k + 0.01 * i * j; }),
            e(nc_, kg, [](int i, int j, int k) { return 3.0 + 0.20 * i - 0.10 * j + 0.30 * k; }),
            halo(nc_, kg)
        {
            for (unsigned int ax = 0; ax < 3; ax++)
            {
                vel[ax] = std::make_unique<GhostedField>(
                    nc, kg, [ax](int i, int j, int k) { return 0.5 + ax + 0.3 * i - 0.2 * j + 0.1 * k * (ax + 1); });
            }
            halo.fill(deltaRho);
            for (unsigned int isc = 0; isc < nscalars; isc++)
            {
                phi.push_back(std::make_unique<GhostedField>(
                    nc, kg, [isc](int i, int j, int k) { return 0.2 + isc + 0.05 * i + 0.04 * j - 0.03 * k; }));
            }
        }

        static double deltaRho(int i, int j, int k)
        { return 0.3 - 0.11 * i + 0.13 * j + 0.17 * k + 0.02 * i * k; }

        static double deltaRhoE(int i, int j, int k)
        { return 1.5 + 0.25 * i + 0.15 * j - 0.35 * k; }

        static double deltaRhoV(unsigned int ax, int i, int j, int k)
        { return -0.4 + 0.6 * ax + 0.12 * i + 0.21 * j - 0.09 * k; }

        static double deltaRhoPhi(unsigned int isc, int i, int j, int k)
        { return 0.7 + isc - 0.06 * i + 0.08 * j + 0.02 * k; }

        static double deltaB(unsigned int ax, int i, int j, int k)
        { return 10.0 * (ax + 1) + i + 0.1 * j + 0.01 * k; }

        /// Run the conversion with cells for the cell centred deltas and `velSlab` / `bSlab` for the staggered ones
        void run(const std::array<std::array<std::array<int, 3>, 2>, 3>& velSlab,
                 const std::array<std::array<std::array<int, 3>, 2>, 3>& bSlab,
                 const bool mhd = true)
        {
            const std::array<int, 3> zero = {0, 0, 0};
            cvRho                         = std::make_unique<NodeField>(zero, nc, deltaRho);
            cvE                           = std::make_unique<NodeField>(zero, nc, deltaRhoE);
            pvRho                         = std::make_unique<NodeField>(zero, nc);
            pvE                           = std::make_unique<NodeField>(zero, nc);
            ConsToPrim::SolutionFields solution {
                &rho.view, &e.view, {&vel[0]->view, &vel[1]->view, &vel[2]->view},
                  {}
            };
            ConsToPrim::ConservedDeltas cons;
            ConsToPrim::PrimitiveDeltas prim;
            cons.rho = cvRho->view.get();
            cons.e   = cvE->view.get();
            prim.rho = pvRho->view.get();
            prim.e   = pvE->view.get();
            for (unsigned int ax = 0; ax < 3; ax++)
            {
                cvVel[ax] = std::make_unique<NodeField>(
                    velSlab[ax][0], velSlab[ax][1], [ax](int i, int j, int k) { return deltaRhoV(ax, i, j, k); });
                pvVel[ax] = std::make_unique<NodeField>(velSlab[ax][0], velSlab[ax][1]);
                cvB[ax]   = std::make_unique<NodeField>(
                    bSlab[ax][0], bSlab[ax][1], [ax](int i, int j, int k) { return deltaB(ax, i, j, k); });
                pvB[ax]                = std::make_unique<NodeField>(bSlab[ax][0], bSlab[ax][1]);
                cons.vel[ax]           = cvVel[ax]->view.get();
                prim.vel[ax]           = pvVel[ax]->view.get();
                cons.magneticField[ax] = cvB[ax]->view.get();
                prim.magneticField[ax] = pvB[ax]->view.get();
            }
            for (unsigned int isc = 0; isc < phi.size(); isc++)
            {
                cvPhi.push_back(
                    std::make_unique<NodeField>(zero, nc, [isc](int i, int j, int k) { return deltaRhoPhi(isc, i, j, k); }));
                pvPhi.push_back(std::make_unique<NodeField>(zero, nc));
                solution.activeScalars.push_back(&phi[isc]->view);
                cons.activeScalars.push_back(cvPhi[isc]->view.get());
                prim.activeScalars.push_back(pvPhi[isc]->view.get());
            }
            std::array<const Array3D<double>*, 3> rhoG = {&halo, &halo, ndim == 3 ? &halo : nullptr};
            ConsToPrim::consToPrim(ndim, nmom, mhd, solution, rhoG, cons, prim);
        }

        double solRho(int i, int j, int k) const
        { return rho.view(i + 1, j + 1, k + kg); }

        double solE(int i, int j, int k) const
        { return e.view(i + 1, j + 1, k + kg); }

        double solV(unsigned int ax, int i, int j, int k) const
        { return vel[ax]->view(i + 1, j + 1, k + kg); }

        double solPhi(unsigned int isc, int i, int j, int k) const
        { return phi[isc]->view(i + 1, j + 1, k + kg); }

        /// The expected delta v at face (i, j, k) of component ax, written out from the formula of MUSIC
        double expectedDeltaV(unsigned int ax, int i, int j, int k) const
        {
            const int di = (ax == 0), dj = (ax == 1), dk = (ax == 2);
            double rho_f, delta_rho_f;
            if (ax < ndim)
            {
                rho_f       = 0.5 * (solRho(i, j, k) + solRho(i - di, j - dj, k - dk));
                delta_rho_f = 0.5 * (deltaRho(i - di, j - dj, k - dk) + deltaRho(i, j, k));
            }
            else
            {
                rho_f       = solRho(i, j, k);
                delta_rho_f = deltaRho(i, j, k);
            }
            return (deltaRhoV(ax, i, j, k) - solV(ax, i, j, k) * delta_rho_f) / rho_f;
        }

        unsigned int ndim, nmom;
        std::array<int, 3> nc;
        int kg;
        GhostedField rho, e;
        std::array<std::unique_ptr<GhostedField>, 3> vel;
        std::vector<std::unique_ptr<GhostedField>> phi;
        HaloView halo;
        std::unique_ptr<NodeField> cvRho, cvE, pvRho, pvE;
        std::array<std::unique_ptr<NodeField>, 3> cvVel, pvVel, cvB, pvB;
        std::vector<std::unique_ptr<NodeField>> cvPhi, pvPhi;
    };

    /// The default solved faces of an interior or periodic rank: [0, n) along every axis
    std::array<std::array<std::array<int, 3>, 2>, 3> cellSlabs(const std::array<int, 3> nc)
    {
        return {
            {{{{0, 0, 0}, nc}}, {{{0, 0, 0}, nc}}, {{{0, 0, 0}, nc}}}
        };
    }
}  // namespace

TEST_CASE("ConsToPrim rejects invalid input")
{
    const std::array<int, 3> nc = {3, 3, 3};
    for (const auto& [ndim, nmom] : std::vector<std::pair<unsigned int, unsigned int>> {
             {1, 3},
             {4, 3},
             {3, 1},
             {3, 4},
             {3, 2}
    })
    {
        CAPTURE(ndim, nmom);
        Problem p(3, 3, nc, 0);
        p.ndim = ndim;
        p.nmom = nmom;
        REQUIRE_THROWS_AS(p.run(cellSlabs(nc), cellSlabs(nc)), std::invalid_argument);
    }

    SECTION("delta_prim e on other nodes than delta_cons rho")
    {
        Problem p(3, 3, nc, 0);
        auto slabs = cellSlabs(nc);
        // Build the problem, then give delta_prim e a smaller slab
        p.run(slabs, slabs);
        NodeField smaller({0, 0, 0}, {2, 3, 3});
        ConsToPrim::SolutionFields solution {
            &p.rho.view, &p.e.view, {&p.vel[0]->view, &p.vel[1]->view, &p.vel[2]->view},
              {}
        };
        ConsToPrim::ConservedDeltas cons {
            p.cvRho->view.get(),
            p.cvE->view.get(),
            {p.cvVel[0]->view.get(), p.cvVel[1]->view.get(), p.cvVel[2]->view.get()},
            {},
            {p.cvB[0]->view.get(), p.cvB[1]->view.get(), p.cvB[2]->view.get()}
        };
        ConsToPrim::PrimitiveDeltas prim {
            p.pvRho->view.get(),
            smaller.view.get(),
            {p.pvVel[0]->view.get(), p.pvVel[1]->view.get(), p.pvVel[2]->view.get()},
            {},
            {p.pvB[0]->view.get(), p.pvB[1]->view.get(), p.pvB[2]->view.get()}
        };
        REQUIRE_THROWS_AS(ConsToPrim::consToPrim(3, 3, true, solution, {&p.halo, &p.halo, &p.halo}, cons, prim),
                          std::invalid_argument);
    }

    SECTION("active scalars missing in delta_prim")
    {
        Problem p(3, 3, nc, 1);
        p.run(cellSlabs(nc), cellSlabs(nc));
        ConsToPrim::SolutionFields solution {
            &p.rho.view, &p.e.view, {&p.vel[0]->view, &p.vel[1]->view, &p.vel[2]->view},
              {&p.phi[0]->view}
        };
        ConsToPrim::ConservedDeltas cons {
            p.cvRho->view.get(),
            p.cvE->view.get(),
            {p.cvVel[0]->view.get(), p.cvVel[1]->view.get(), p.cvVel[2]->view.get()},
            {p.cvPhi[0]->view.get()},
            {p.cvB[0]->view.get(), p.cvB[1]->view.get(), p.cvB[2]->view.get()}
        };
        ConsToPrim::PrimitiveDeltas prim {
            p.pvRho->view.get(),
            p.pvE->view.get(),
            {p.pvVel[0]->view.get(), p.pvVel[1]->view.get(), p.pvVel[2]->view.get()},
            {},
            {p.pvB[0]->view.get(), p.pvB[1]->view.get(), p.pvB[2]->view.get()}
        };
        REQUIRE_THROWS_AS(ConsToPrim::consToPrim(3, 3, true, solution, {&p.halo, &p.halo, &p.halo}, cons, prim),
                          std::invalid_argument);
    }
}

TEST_CASE("ConsToPrim 3D with an active scalar and MHD")
{
    const std::array<int, 3> nc = {3, 4, 5};
    Problem p(3, 3, nc, 2);
    p.run(cellSlabs(nc), cellSlabs(nc));

    for (int k = 0; k < nc[2]; k++)
    {
        for (int j = 0; j < nc[1]; j++)
        {
            for (int i = 0; i < nc[0]; i++)
            {
                CAPTURE(i, j, k);
                REQUIRE((*p.pvRho->view)(i, j, k) == Problem::deltaRho(i, j, k));
                REQUIRE(close((*p.pvE->view)(i, j, k),
                              (Problem::deltaRhoE(i, j, k) - p.solE(i, j, k) * Problem::deltaRho(i, j, k)) / p.solRho(i, j, k)));
                for (unsigned int isc = 0; isc < 2; isc++)
                {
                    REQUIRE(close((*p.pvPhi[isc]->view)(i, j, k),
                                  (Problem::deltaRhoPhi(isc, i, j, k) - p.solPhi(isc, i, j, k) * Problem::deltaRho(i, j, k)) /
                                      p.solRho(i, j, k)));
                }
                for (unsigned int ax = 0; ax < 3; ax++)
                {
                    CAPTURE(ax);
                    // At i, j or k = 0 the face average reads delta rho in the ghost of the rank
                    REQUIRE(close((*p.pvVel[ax]->view)(i, j, k), p.expectedDeltaV(ax, i, j, k)));
                    REQUIRE((*p.pvB[ax]->view)(i, j, k) == Problem::deltaB(ax, i, j, k));
                }
            }
        }
    }
}

TEST_CASE("ConsToPrim converts each velocity over its own solved faces")
{
    // Along i: the left boundary face is not solved (faces start at 1), the right boundary face is (faces end at n + 1),
    // as MUSIC `solved_local_domain_nodes` gives on a rank touching both boundaries
    const std::array<int, 3> nc = {4, 3, 3};
    Problem p(3, 3, nc, 0);
    auto velSlabs  = cellSlabs(nc);
    velSlabs[0][0] = {1, 0, 0};
    velSlabs[0][1] = {nc[0] + 1, nc[1], nc[2]};
    auto bSlabs    = cellSlabs(nc);
    bSlabs[0][1]   = {nc[0] + 1, nc[1], nc[2]};
    p.run(velSlabs, bSlabs);

    for (int k = 0; k < nc[2]; k++)
    {
        for (int j = 0; j < nc[1]; j++)
        {
            CAPTURE(j, k);
            // Left boundary face not solved: not written
            REQUIRE(std::isnan((*p.pvVel[0]->parent)(0, j, k)));
            // Right boundary face solved: averages with the ghost cell beyond the rank
            for (int i = 1; i <= nc[0]; i++)
            {
                CAPTURE(i);
                REQUIRE(close((*p.pvVel[0]->view)(i, j, k), p.expectedDeltaV(0, i, j, k)));
            }
            // Magnetic field copied over its own faces, including the right boundary face
            REQUIRE((*p.pvB[0]->view)(nc[0], j, k) == Problem::deltaB(0, nc[0], j, k));
        }
    }
}

TEST_CASE("ConsToPrim 2D uses cell values for the third velocity component")
{
    const std::array<int, 3> nc = {4, 3, 1};
    Problem p(2, 3, nc, 0);
    p.run(cellSlabs(nc), cellSlabs(nc));

    for (int j = 0; j < nc[1]; j++)
    {
        for (int i = 0; i < nc[0]; i++)
        {
            CAPTURE(i, j);
            REQUIRE(close((*p.pvE->view)(i, j, 0),
                          (Problem::deltaRhoE(i, j, 0) - p.solE(i, j, 0) * Problem::deltaRho(i, j, 0)) / p.solRho(i, j, 0)));
            for (unsigned int ax = 0; ax < 3; ax++)
            {
                CAPTURE(ax);
                REQUIRE(close((*p.pvVel[ax]->view)(i, j, 0), p.expectedDeltaV(ax, i, j, 0)));
            }
        }
    }
}

TEST_CASE("ConsToPrim 2D with two velocity components and no MHD")
{
    const std::array<int, 3> nc = {3, 3, 1};
    Problem p(2, 2, nc, 0);
    p.run(cellSlabs(nc), cellSlabs(nc), false);

    for (int j = 0; j < nc[1]; j++)
    {
        for (int i = 0; i < nc[0]; i++)
        {
            CAPTURE(i, j);
            REQUIRE(close((*p.pvVel[0]->view)(i, j, 0), p.expectedDeltaV(0, i, j, 0)));
            REQUIRE(close((*p.pvVel[1]->view)(i, j, 0), p.expectedDeltaV(1, i, j, 0)));
            // The third component and the magnetic field are not touched
            REQUIRE(std::isnan((*p.pvVel[2]->view)(i, j, 0)));
            REQUIRE(std::isnan((*p.pvB[0]->view)(i, j, 0)));
        }
    }
}

namespace
{
    /**
     * @brief The inputs and expected outputs of MUSIC `cons_to_prim` for one rank (rho, e and the velocity components,
     * written by the MUSIC diff given in the pull request; MUSIC indices start at 1).
     */
    struct MusicData
    {
        unsigned int ndim = 0, nmom = 0;
        int kg = 0;
        std::array<int, 3> nc {};
        std::vector<std::array<std::array<int, 3>, 2>> slabs;  // [variable][start / end], local indices
        std::vector<std::unique_ptr<GhostedField>> solution;   // rho, e, vel1, vel2, vel3
        std::vector<std::unique_ptr<NodeField>> cons, expected, prim;
        std::unique_ptr<HaloView> halo;
    };

    void readMusicData(const std::string& inputsPath, const std::string& outputsPath, MusicData& m)
    {
        FILE* file = fopen(inputsPath.c_str(), "r");
        REQUIRE(file != nullptr);
        int mhd, nscalars;
        REQUIRE(fscanf(file, " ndim_nmom_mhd %u %u %d", &m.ndim, &m.nmom, &mhd) == 3);
        REQUIRE(fscanf(file, " active_scalars %d", &nscalars) == 1);
        REQUIRE(nscalars == 0);
        REQUIRE(fscanf(file, " local_cells %d %d %d", &m.nc[0], &m.nc[1], &m.nc[2]) == 3);
        m.kg = (m.ndim == 3) ? 1 : 0;
        for (unsigned int v = 0; v < 2 + m.nmom; v++)
        {
            char name[16];
            std::array<int, 3> b, e;
            REQUIRE(fscanf(file, " slab %15s %d %d %d %d %d %d", name, &b[0], &b[1], &b[2], &e[0], &e[1], &e[2]) == 7);
            m.slabs.push_back({
                {{b[0] - 1, b[1] - 1, b[2] - 1}, e}
            });
        }

        // Solution with one ghost layer, five values per line, i fastest
        const size_t numGhosted = static_cast<size_t>(m.nc[0] + 2) * (m.nc[1] + 2) * (m.nc[2] + 2 * m.kg);
        std::vector<std::vector<double>> values(5, std::vector<double>(numGhosted));
        for (size_t n = 0; n < numGhosted; n++)
        {
            REQUIRE(
                fscanf(file, " %lf %lf %lf %lf %lf", &values[0][n], &values[1][n], &values[2][n], &values[3][n], &values[4][n]) ==
                5);
        }
        const int gi = m.nc[0] + 2, gj = m.nc[1] + 2;
        for (unsigned int v = 0; v < 5; v++)
        {
            m.solution.push_back(std::make_unique<GhostedField>(
                m.nc, m.kg, [&, v](int i, int j, int k) { return values[v][(i + 1) + gi * ((j + 1) + gj * (k + m.kg))]; }));
        }

        // delta_cons over the slab of each variable, i fastest
        for (unsigned int v = 0; v < 2 + m.nmom; v++)
        {
            m.cons.push_back(std::make_unique<NodeField>(m.slabs[v][0], m.slabs[v][1]));
            const Slab s = m.cons[v]->view->cellSlab();
            for (int k = s.start(2); k < s.end(2); k++)
                for (int j = s.start(1); j < s.end(1); j++)
                    for (int i = s.start(0); i < s.end(0); i++)
                        REQUIRE(fscanf(file, " %lf", &(*m.cons[v]->view)(i, j, k)) == 1);
        }

        // delta rho inside the rank, then the halo values MUSIC read outside it
        m.halo       = std::make_unique<HaloView>(m.nc, m.kg);
        const Slab s = m.cons[0]->view->cellSlab();
        for (int k = s.start(2); k < s.end(2); k++)
            for (int j = s.start(1); j < s.end(1); j++)
                for (int i = s.start(0); i < s.end(0); i++)
                    (*m.halo)(i, j, k) = (*m.cons[0]->view)(i, j, k);
        for (unsigned int ax = 0; ax < m.ndim; ax++)
        {
            unsigned int axis, count;
            REQUIRE(fscanf(file, " halo %u %u", &axis, &count) == 2);
            REQUIRE(axis == ax + 1);
            for (unsigned int n = 0; n < count; n++)
            {
                int i, j, k;
                double value;
                REQUIRE(fscanf(file, " %d %d %d %lf", &i, &j, &k, &value) == 4);
                (*m.halo)(i - 1, j - 1, k - 1) = value;
            }
        }
        fclose(file);

        // Expected delta_prim over the slab of each variable, i fastest
        file = fopen(outputsPath.c_str(), "r");
        REQUIRE(file != nullptr);
        for (unsigned int v = 0; v < 2 + m.nmom; v++)
        {
            m.expected.push_back(std::make_unique<NodeField>(m.slabs[v][0], m.slabs[v][1]));
            m.prim.push_back(std::make_unique<NodeField>(m.slabs[v][0], m.slabs[v][1]));
            const Slab so = m.expected[v]->view->cellSlab();
            for (int k = so.start(2); k < so.end(2); k++)
                for (int j = so.start(1); j < so.end(1); j++)
                    for (int i = so.start(0); i < so.end(0); i++)
                        REQUIRE(fscanf(file, " %lf", &(*m.expected[v]->view)(i, j, k)) == 1);
        }
        fclose(file);
    }
}  // namespace

TEST_CASE("ConsToPrim matches MUSIC cons_to_prim on the Roberts dynamo test")
{
    const std::string dataDir = TEST_DATA_DIR;
    for (const auto& [ranks, label] : std::vector<std::pair<int, std::string>> {
             {1, "1rank" },
             {4, "4ranks"}
    })
    {
        for (int rank = 0; rank < ranks; rank++)
        {
            CAPTURE(label, rank);
            const std::string prefix = dataDir + "/cons_to_prim_roberts_dynamo_" + label + "_rank" + std::to_string(rank);
            MusicData m;
            readMusicData(prefix + "_inputs.dat", prefix + "_expected_outputs.dat", m);

            ConsToPrim::SolutionFields solution {
                &m.solution[0]->view,
                &m.solution[1]->view,
                {&m.solution[2]->view, &m.solution[3]->view, &m.solution[4]->view},
                {}
            };
            ConsToPrim::ConservedDeltas cons {
                m.cons[0]->view.get(),
                m.cons[1]->view.get(),
                {m.cons[2]->view.get(), m.cons[3]->view.get(), m.cons[4]->view.get()},
                {},
                {}
            };
            ConsToPrim::PrimitiveDeltas prim {
                m.prim[0]->view.get(),
                m.prim[1]->view.get(),
                {m.prim[2]->view.get(), m.prim[3]->view.get(), m.prim[4]->view.get()},
                {},
                {}
            };
            // The magnetic field is a copy and is not in the data; it is covered by the cases above
            ConsToPrim::consToPrim(m.ndim, m.nmom, false, solution, {m.halo.get(), m.halo.get(), m.halo.get()}, cons, prim);

            // The velocity terms v delta_rho_f and delta_rho_v nearly cancel at some faces, so the round-off of the two
            // compilers is compared with the size of each field rather than with each value
            const std::array<const char*, 5> names = {"rho", "e", "vel1", "vel2", "vel3"};
            for (unsigned int v = 0; v < 2 + m.nmom; v++)
            {
                CAPTURE(names[v]);
                const Slab s = m.prim[v]->view->cellSlab();
                double scale = 0;
                for (int k = s.start(2); k < s.end(2); k++)
                    for (int j = s.start(1); j < s.end(1); j++)
                        for (int i = s.start(0); i < s.end(0); i++)
                            scale = std::max(scale, std::abs((*m.expected[v]->view)(i, j, k)));
                REQUIRE(scale > 0);
                for (int k = s.start(2); k < s.end(2); k++)
                {
                    for (int j = s.start(1); j < s.end(1); j++)
                    {
                        for (int i = s.start(0); i < s.end(0); i++)
                        {
                            CAPTURE(i, j, k);
                            REQUIRE(std::abs((*m.prim[v]->view)(i, j, k) - (*m.expected[v]->view)(i, j, k)) <= 1e-14 * scale);
                        }
                    }
                }
            }
        }
    }
}
