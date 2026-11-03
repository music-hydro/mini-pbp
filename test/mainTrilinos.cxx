/**
 * @brief Catch2 entrypoint for tests which need MPI and Kokkos to be initialised (Trilinos based tests).
 */

#include <catch2/catch_session.hpp>
#include <Tpetra_Core.hpp>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetraScope(&argc, &argv);
    return Catch::Session().run(argc, argv);
}
