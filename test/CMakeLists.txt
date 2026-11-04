find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()

set(PROJ_TEST_DIR "${PROJECT_SOURCE_DIR}/test")

# Serial tests
set(TEST_FILES
    "${PROJ_TEST_DIR}/fixtures/PhysicsBasedPreconditionerTest.cxx"
    "${PROJ_TEST_DIR}/fixtures/MpiTest.cxx"
    "${PROJ_TEST_DIR}/fixtures/Array3DWithGhostsTest.cxx"
    "${PROJ_TEST_DIR}/testArray3D.cxx"
    "${PROJ_TEST_DIR}/testArray2D.cxx"
    # "${PROJ_TEST_DIR}/testArray3DWithGhosts.cxx"
    # "${PROJ_TEST_DIR}/testGhostExchangePlan.cxx"
    "${PROJ_TEST_DIR}/testPackedVariables.cxx"
    "${PROJ_TEST_DIR}/testPhysicsBasedPreconditioner.cxx"
    "${PROJ_TEST_DIR}/testReconstructDensityDelta.cxx"
    "${PROJ_TEST_DIR}/testReconstructEnergyDelta.cxx"
    "${PROJ_TEST_DIR}/testSetPressureRhs.cxx"
    "${PROJ_TEST_DIR}/testSlab.cxx"
)

set(BENCHMARK_FILES
    "${PROJ_TEST_DIR}/fixtures/PhysicsBasedPreconditionerBenchmark.cxx"
    "${PROJ_TEST_DIR}/benchmarkPhysicsBasedPreconditioner.cxx"
)

# These tests can use the Catch2-provided main
add_executable(mini_pbp_unit_tests ${TEST_FILES} ${HEADER_FILES} ${IMPLEMENTATION_FILES})
target_link_libraries(mini_pbp_unit_tests PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(mini_pbp_unit_tests PRIVATE
    TEST_DATA_DIR="${PROJ_TEST_DIR}/data"
)
add_test(NAME mini_pbp_unit_tests COMMAND mini_pbp_unit_tests)

add_executable(mini_pbp_benchmarks ${BENCHMARK_FILES} ${HEADER_FILES} ${IMPLEMENTATION_FILES})
target_link_libraries(mini_pbp_benchmarks PRIVATE Catch2::Catch2WithMain)

# MPI tests
set(MPI_TEST_FILES
    "${PROJ_TEST_DIR}/mainMPITests.cxx"
    "${PROJ_TEST_DIR}/testArray3DWithGhosts.cxx"
    "${PROJ_TEST_DIR}/testGhostExchangePlan.cxx"
)

add_executable(mini_pbp_mpi_unit_tests ${MPI_TEST_FILES})
target_link_libraries(mini_pbp_mpi_unit_tests PRIVATE "${PROJECT_NAME}_lib" Catch2::Catch2 )
target_compile_definitions(mini_pbp_mpi_unit_tests PRIVATE
    TEST_DATA_DIR="${PROJ_TEST_DIR}/data"
)

foreach(num_ranks 1 4 8)
    add_test(NAME mini_pbp_mpi_unit_tests_${num_ranks}_ranks
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${num_ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:mini_pbp_mpi_unit_tests> ${MPIEXEC_POSTFLAGS})
endforeach()
