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

set(TEST_FILES
    "${PROJ_TEST_DIR}/testArray3D.cxx"
    "${PROJ_TEST_DIR}/testPackedVariables.cxx"
    "${PROJ_TEST_DIR}/testSlab.cxx"
    "${PROJ_TEST_DIR}/testPhysicsBasedPreconditioner.cxx"
)

# These tests can use the Catch2-provided main
add_executable(tests ${TEST_FILES} ${HEADER_FILES} ${IMPLEMENTATION_FILES})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(tests PRIVATE
    TEST_DATA_DIR="${PROJ_TEST_DIR}/data"
)
add_test(NAME tests COMMAND tests)
