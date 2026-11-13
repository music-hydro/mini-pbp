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
    static GhostExchangePlan<double> arbitraryPlan()
    {
        // The plan shouldn't matter here
        DirectionalArray<BoundaryConditions> bcs;
        bcs.fill(BoundaryConditions::INNER);
        DirectionalArray<int> neighbours;
        neighbours.fill(0);
        GhostExchangePlan<double> plan(bcs, neighbours, MPI_COMM_WORLD);
        return plan;
    }
};

#endif  // GhostExchangePlanTest_H
