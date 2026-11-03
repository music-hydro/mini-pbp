cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(
    "mini-pbp"
    LANGUAGES "CXX"
    VERSION "0.0.1"
    DESCRIPTION "Standalone Physics Based Preconditioner for music"
)

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_DIRS "${SRC_DIR}/include")
include_directories(${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Define src files
set(HEADER_FILES
    "${INCLUDE_DIRS}/Array3D.h"
    "${INCLUDE_DIRS}/Array2D.h"
    "${INCLUDE_DIRS}/Array3DWithGhosts.h"
    "${INCLUDE_DIRS}/PackedVariables.h"
    "${INCLUDE_DIRS}/Slab.h"
    "${INCLUDE_DIRS}/PhysicsBasedPreconditioner.h"
)
set(IMPLEMENTATION_FILES
    "${SRC_DIR}/PackedVariables.cxx"
    "${SRC_DIR}/PhysicsBasedPreconditioner.cxx"
)

add_executable(mini-pbp "${SRC_DIR}/main.cxx" ${HEADER_FILES} ${IMPLEMENTATION_FILES})

# Pressure solve with Trilinos (Tpetra, Belos, MueLu on Kokkos). Built only when Trilinos is found, e.g. by adding
# the Trilinos install prefix to CMAKE_PREFIX_PATH.
find_package(Trilinos QUIET COMPONENTS Tpetra Belos MueLu)
if(Trilinos_FOUND)
    message(STATUS "Found Trilinos ${Trilinos_VERSION}: building the pressure solver")
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_library(mini_pbp_pressure_solver
        "${INCLUDE_DIRS}/HostArrayVector.h"
        "${INCLUDE_DIRS}/PressureSolver.h"
        "${SRC_DIR}/PressureSolver.cxx"
    )
    target_include_directories(mini_pbp_pressure_solver PUBLIC ${INCLUDE_DIRS})
    target_link_libraries(mini_pbp_pressure_solver PUBLIC Tpetra::all_libs Belos::all_libs MueLu::all_libs MPI::MPI_CXX)
else()
    message(STATUS "Trilinos not found: the pressure solver and its tests are not built")
endif()

enable_testing()
add_subdirectory("test")
