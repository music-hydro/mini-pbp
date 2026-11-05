#include <catch2/catch_session.hpp>
#include <mpi.h>
#include <MpiType.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    Catch::Session session;
    const int status = session.applyCommandLine(argc, argv);
    if (status != 0) { return status; }

    // The test cases call MPI collectives, so every rank must run them in the same order. Catch2 shuffles the test
    // cases with a random seed drawn on each rank, so use the seed of rank 0 on all ranks.
    std::uint32_t seed = session.configData().rngSeed;
    MPI_Bcast(&seed, 1, MpiType<int>::value, 0, MPI_COMM_WORLD);
    session.configData().rngSeed = seed;

    int result = session.run(argc, argv);
    MPI_Finalize();
    return result;
}
