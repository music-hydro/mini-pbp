/**
 * @class GhostExchangePlanTest
 *
 * @brief Convenience functions for tests that utilise GhostExchangePlan
 */

#ifndef GhostExchangePlanTest_H
#define GhostExchangePlanTest_H

#include "GhostExchangePlan.h"

#include "DirectionalArray.h"
#include "MpiTest.cxx"
#include <mpi.h>

class GhostExchangePlanTest : public MpiTest
{
  public:
    /**
     * @brief Create an arbitrary plan for tests which require one to exist but don't care about its state.
     *
     * @returns The arbitrary GhostExchangePlan
     */
    static GhostExchangePlan arbitraryPlan(const Slab& global_domain)
    {
        // The plan shouldn't matter here
        DirectionalArray<BoundaryConditions> bcs;
        bcs.fill(BoundaryConditions::INNER);
        DirectionalArray<int> neighbours;
        neighbours.fill(0);
        // Equivalent to one rank
        std::array<int, 3> dims = createCartesianDimensions();
        GhostExchangePlan plan(global_domain, bcs, neighbours, dims, createCartesianTopology(dims, bcs));
        return plan;
    }

    static GhostExchangePlan arbitraryPlan()
    {
        Slab global_domain({0, 0, 0}, {3, 3, 3});
        return arbitraryPlan(global_domain);
    }

    /**
     * @brief Create an periodic plan for tests which require one to exist.
     *
     * @returns The periodic GhostExchangePlan
     */
    static GhostExchangePlan periodicPlan(const Slab& global_domain)
    {
        DirectionalArray<BoundaryConditions> global_bcs;
        global_bcs.fill(BoundaryConditions::PERIODIC);
        std::array<int, 3> communicator_dimensions(createCartesianDimensions());
        MPI_Comm communicator(createCartesianTopology(communicator_dimensions, global_bcs));
        DirectionalArray<int> nghbrs(neighbours(communicator));
        return GhostExchangePlan(global_domain, global_bcs, nghbrs, communicator_dimensions, communicator);
    }
};

#endif  // GhostExchangePlanTest_H
