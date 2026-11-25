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
    static GhostExchangePlan arbitraryPlan()
    {
        // The plan shouldn't matter here
        DirectionalArray<BoundaryConditions> bcs;
        bcs.fill(BoundaryConditions::INNER);
        DirectionalArray<int> neighbours;
        neighbours.fill(0);
        GhostExchangePlan plan(bcs, neighbours, MPI_COMM_WORLD);
        return plan;
    }

    /**
     * @brief Create an periodic plan for tests which require one to exist.
     *
     * @returns The periodic GhostExchangePlan
     */
    static GhostExchangePlan periodicPlan()
    {
        DirectionalArray<BoundaryConditions> bcs;
        bcs.fill(BoundaryConditions::PERIODIC);
        DirectionalArray<int> neighbours;
        neighbours.fill(0);
        return GhostExchangePlan(bcs, neighbours, MPI_COMM_SELF);
    }
};

#endif  // GhostExchangePlanTest_H
