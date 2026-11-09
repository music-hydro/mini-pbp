
#include "fixtures/GhostExchangePlanTest.cxx"
#include "PhysicsBasedPreconditioner.h"
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("PhysicsBasedPreconditioner::PhysicsBasedPreconditioner throws if solution is not a valid size")
{
    int n    = 2;
    int size = n * n * n;
    std::vector<double> solution_raw(size);                                                       // invalid
    std::vector<double> solution_dPd_raw(size * PhysicsBasedPreconditioner::num_sol_dPd_vars());  // valid
    std::vector<double> squared_sound_speed_raw(size);                                            // valid
    PackedVariablesWithGhosts solution(solution_raw, n, n, n, GhostExchangePlanTest::arbitraryPlan());
    PackedVariables solution_dPd(solution_dPd_raw, n, n, n);
    Array3D<double> squared_sound_speed(n, n, n, squared_sound_speed_raw);

    REQUIRE_THROWS_WITH(PhysicsBasedPreconditioner(3, 3, false, solution, solution_dPd, squared_sound_speed),
                        Catch::Matchers::ContainsSubstring("Solution should contain exactly " +
                                                           std::to_string(PhysicsBasedPreconditioner::num_sol_vars())));
}

TEST_CASE("PhysicsBasedPreconditioner::PhysicsBasedPreconditioner throws if solution_dPd is not a valid size")
{
    int n    = 2;
    int size = n * n * n;
    std::vector<double> solution_raw(size * PhysicsBasedPreconditioner::num_sol_vars());  // valid
    std::vector<double> solution_dPd_raw(size);                                           // invalid
    std::vector<double> squared_sound_speed_raw(size);                                    // valid
    PackedVariablesWithGhosts solution(solution_raw, n, n, n, GhostExchangePlanTest::arbitraryPlan());
    PackedVariables solution_dPd(solution_dPd_raw, n, n, n);
    Array3D<double> squared_sound_speed(n, n, n, squared_sound_speed_raw);

    REQUIRE_THROWS_WITH(PhysicsBasedPreconditioner(3, 3, false, solution, solution_dPd, squared_sound_speed),
                        Catch::Matchers::ContainsSubstring("Solution gradients should contain exactly 2 variables"));
}

TEST_CASE("PhysicsBasedPreconditioner::PhysicsBasedPreconditioner rejects invalid input ndim and/or nmom")
{
    const std::array<int, 3> nc     = {3, 3, 3};
    const size_t size               = static_cast<size_t>(nc[0]) * nc[1] * nc[2];
    Slab global_domain({0,0,0}, nc);
    GhostExchangePlan plan  = GhostExchangePlanTest::periodicPlan(global_domain);
    const unsigned int num_sol_vars = PhysicsBasedPreconditioner::num_sol_vars();
    std::vector<double> solution_raw(num_sol_vars * size);
    std::vector<double> solution_dPd_raw(PhysicsBasedPreconditioner::num_sol_dPd_vars() * size);
    std::vector<double> squared_sound_speed_raw(size);
    PackedVariablesWithGhosts solution(solution_raw, nc[0], nc[1], nc[2], plan);
    PackedVariables solution_dPd(solution_dPd_raw, nc[0], nc[1], nc[2]);
    Array3D<double> squared_sound_speed(nc[0], nc[1], nc[2], squared_sound_speed_raw);

    for (const auto [ndim, nmom, expected_err] : std::array<std::tuple<unsigned int, unsigned int, std::string>, 5> {
             {{1, 3, "ndim must be 2 or 3"},
              {4, 3, "ndim must be 2 or 3"},
              {3, 1, "nmom must be 2 or 3"},
              {3, 4, "nmom must be 2 or 3"},
              {3, 2, "nmom must be at least ndim"}}
    })
    {
        REQUIRE_THROWS_WITH(PhysicsBasedPreconditioner(ndim, nmom, false, solution, solution_dPd, squared_sound_speed),
                            Catch::Matchers::ContainsSubstring(expected_err));
    }
}
