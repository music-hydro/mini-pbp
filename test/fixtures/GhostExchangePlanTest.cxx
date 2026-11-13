/**
 * @file Functions and classes required for test which utilise the class GhostExchangePlan
 */

#ifndef GhostExchangePlanTest_H
#define GhostExchangePlanTest_H

#include "GhostExchangePlan.h"

#include "DirectionalArray.h"

namespace
{
    /**
     * @brief Create an arbitrary plan for tests which require one to exist but don't care about its state.
     */
    GhostExchangePlan<double> arbitraryPlan()
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
