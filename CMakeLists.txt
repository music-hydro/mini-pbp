cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(
    "mini-pbp"
    LANGUAGES "CXX"
    VERSION "0.0.1"
    DESCRIPTION "Standalone Physic Based Preconditioner for music"
)

find_package(MPI COMPONENTS CXX REQUIRED)
set(TO_LINK_TO_SRC MPI::MPI_CXX)

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_DIRS "${SRC_DIR}/include")
include_directories(${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Define src files
set(HEADER_FILES
    "${INCLUDE_DIRS}/Array3D.h"
    "${INCLUDE_DIRS}/Array2D.h"
    "${INCLUDE_DIRS}/Array3DWithGhosts.h"
    "${INCLUDE_DIRS}/GhostExchangePlan.h"
    "${INCLUDE_DIRS}/MpiType.h"
    "${INCLUDE_DIRS}/PackedVariables.h"
    "${INCLUDE_DIRS}/Slab.h"
    "${INCLUDE_DIRS}/PhysicsBasedPreconditioner.h"
)
set(IMPLEMENTATION_FILES
    "${SRC_DIR}/PackedVariables.cxx"
    "${SRC_DIR}/PhysicsBasedPreconditioner.cxx"
)

add_library("${PROJECT_NAME}_lib" ${HEADER_FILES} ${IMPLEMENTATION_FILES})
target_link_libraries("${PROJECT_NAME}_lib" "${TO_LINK_TO_SRC}")

add_executable("${PROJECT_NAME}" "${SRC_DIR}/main.cxx")
target_link_libraries("${PROJECT_NAME}" PRIVATE "${PROJECT_NAME}_lib")

enable_testing()
add_subdirectory("test")
