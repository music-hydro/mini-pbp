/**
 * @class GhostExchangePlanTest
 *
 * @brief Convenience functions for tests that utilise GhostExchangePlan
 */

#ifndef GhostExchangePlanTest_H
#define GhostExchangePlanTest_H

#include "GhostExchangePlan.h"

#include "DirectionalArray.h"

class GhostExchangePlanTest
{
  public:
    /**
     * @brief Create an arbitrary plan for tests which require one to exist but don't care about its state.
     *
     * @returns The arbitrary GhostExchangePlan
     */
    static GhostExchangePlan<double> arbitraryPlan(const Slab& global_domain)
    {
        // The plan shouldn't matter here
        DirectionalArray<BoundaryConditions> bcs;
        bcs.fill(BoundaryConditions::INNER);
        DirectionalArray<int> neighbours;
        neighbours.fill(0);
        GhostExchangePlan<double> plan(global_domain, bcs, neighbours, MPI_COMM_WORLD);
        return plan;
    }
    static GhostExchangePlan<double> arbitraryPlan()
    {
        Slab global_domain({0,0,0}, {3,3,3});
        return arbitraryPlan(global_domain);
    }

    /**
     * @brief Create an periodic plan for tests which require one to exist.
     *
     * @returns The periodic GhostExchangePlan
     */
    static GhostExchangePlan<double> periodicPlan(const Slab& global_domain)
    {
        DirectionalArray<BoundaryConditions> bcs;
        bcs.fill(BoundaryConditions::PERIODIC);
        DirectionalArray<int> neighbours;
        neighbours.fill(0);
        return GhostExchangePlan<double>(global_domain, bcs, neighbours, MPI_COMM_SELF);
    }
};

#endif  // GhostExchangePlanTest_H
