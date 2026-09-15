find_package(Catch2 3 REQUIRED)

SET(PROJ_TEST_DIR "${PROJECT_SOURCE_DIR}/test")

set(TEST_FILES
    "${PROJ_TEST_DIR}/testArray3D.cxx"
    "${PROJ_TEST_DIR}/testPackedVariables.cxx"
)

# These tests can use the Catch2-provided main
add_executable(tests ${TEST_FILES} ${HEADER_FILES} ${IMPLEMENTATION_FILES})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
add_test(NAME tests COMMAND tests)
