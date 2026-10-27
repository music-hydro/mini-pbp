#include "fixtures/PhysicsBasedPreconditionerBenchmark.cxx"
#include "PhysicsBasedPreconditioner.h"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("PhysicsBasedPreconditioner::PhysicsBasedPreconditioner throws if solution is not a valid size")
{
    int n    = 2;
    int size = n * n * n;
    std::vector<double> solution_raw(size);             // invalid
    std::vector<double> solution_dPd_raw(size * 2);     // valid
    std::vector<double> squared_sound_speed_raw(size);  // valid
    PackedVariables solution(solution_raw, n, n, n);
    PackedVariables solution_dPd(solution_dPd_raw, n, n, n);
    Array3D<double> squared_sound_speed(&squared_sound_speed_raw[0], n, n, n);

    REQUIRE_THROWS_WITH(PhysicsBasedPreconditioner(solution, solution_dPd, squared_sound_speed),
                        Catch::Matchers::ContainsSubstring("Solution should contain exactly 3 variables"));
}

TEST_CASE("PhysicsBasedPreconditioner::PhysicsBasedPreconditioner throws if solution_dPd is not a valid size")
{
    int n    = 2;
    int size = n * n * n;
    std::vector<double> solution_raw(size * 3);         // valid
    std::vector<double> solution_dPd_raw(size);         // invalid
    std::vector<double> squared_sound_speed_raw(size);  // valid
    PackedVariables solution(solution_raw, n, n, n);
    PackedVariables solution_dPd(solution_dPd_raw, n, n, n);
    Array3D<double> squared_sound_speed(&squared_sound_speed_raw[0], n, n, n);

    REQUIRE_THROWS_WITH(PhysicsBasedPreconditioner(solution, solution_dPd, squared_sound_speed),
                        Catch::Matchers::ContainsSubstring("Solution gradients should contain exactly 2 variables"));
}

TEST_CASE_METHOD(PhysicsBasedPreconditionerBenchmark, "PhysicsBasedPreconditioner benchmarks")
{
    BENCHMARK("reconstructDensityDelta")
    {
        return pbp->reconstructDensityDelta(*delta_pressure, *delta_energy, *delta_rho);
        // clang-format off
    };
    // clang-format on

    BENCHMARK("reconstructEnergyDelta")
    {
        return pbp->reconstructEnergyDelta(
            *preconditioned_delta_pressure, *preconditioned_delta_energy, *rhs_pressure, *delta_energy);
        // clang-format off
    };
    // clang-format on

    BENCHMARK("setPressureRHS")
    {
        return pbp->setPressureRHS(*delta_rho, *delta_energy, *rhs_pressure, *rhs_energy);
        // clang-format off
    };
    // clang-format on
}
