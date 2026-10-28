#include "fixtures/PhysicsBasedPreconditionerBenchmark.cxx"
#include "PhysicsBasedPreconditioner.h"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE_METHOD(PhysicsBasedPreconditionerBenchmark, "PhysicsBasedPreconditioner benchmarks")
{
    BENCHMARK("reconstructDensityDelta")
    {
        REQUIRE_NOTHROW(pbp->reconstructDensityDelta(*delta_pressure, *delta_energy, *delta_rho));
        return *delta_rho;
        // clang-format off
    };
    // clang-format on

    BENCHMARK("reconstructEnergyDelta")
    {
        REQUIRE_NOTHROW(pbp->reconstructEnergyDelta(
            *preconditioned_delta_pressure, *preconditioned_delta_energy, *rhs_pressure, *delta_energy));
        return *delta_energy;
        // clang-format off
    };
    // clang-format on

    BENCHMARK("setPressureRHS")
    {
        REQUIRE_NOTHROW(pbp->setPressureRHS(*delta_rho, *delta_energy, *rhs_pressure, *rhs_energy));
        return *rhs_energy;
        // clang-format off
    };
    // clang-format on
}
