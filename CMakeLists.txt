cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(
    "mini-pbp"
    LANGUAGES "CXX"
    VERSION "0.0.1"
    DESCRIPTION "Standalone Physic Based Preconditioner for music"
)

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(INCLUDE_DIRS "${SRC_DIR}/include")
include_directories(${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Define src files
set(HEADER_FILES
    "${INCLUDE_DIRS}/Array3D.h"
    "${INCLUDE_DIRS}/PackedVariables.h"
    "${INCLUDE_DIRS}/Slab.h"
    "${INCLUDE_DIRS}/PhysicsBasedPreconditioner.h"
)
set(IMPLEMENTATION_FILES
    "${SRC_DIR}/PackedVariables.cxx"
    "${SRC_DIR}/PhysicsBasedPreconditioner.cxx"
)

add_executable(mini-pbp "${SRC_DIR}/main.cxx" ${HEADER_FILES} ${IMPLEMENTATION_FILES})

enable_testing()
add_subdirectory("test")
