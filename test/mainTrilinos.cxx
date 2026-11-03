/**
 * @brief Catch2 entrypoint for tests which need MPI and Kokkos to be initialised (Trilinos based tests).
 */

#include <catch2/catch_session.hpp>
#include <cstdint>
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetraScope(&argc, &argv);

    Catch::Session session;
    const int status = session.applyCommandLine(argc, argv);
    if (status != 0) { return status; }

    // The test cases call MPI collectives, so every rank must run them in the same order. Catch2 shuffles the test
    // cases with a random seed drawn on each rank, so use the seed of rank 0 on all ranks.
    std::uint32_t seed = session.configData().rngSeed;
    Teuchos::broadcast(*Tpetra::getDefaultComm(), 0, &seed);
    session.configData().rngSeed = seed;

    return session.run();
}
