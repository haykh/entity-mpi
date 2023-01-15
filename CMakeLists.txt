cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME entity-mpi)

project(${PROJECT_NAME}
  LANGUAGES CXX
)

find_package(Git QUIET)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  option(GIT_SUBMODULE "Check submodules during build" ON)

  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)

    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()

# -------------------------- Compilation settings -------------------------- #
set(DEBUG "OFF" CACHE BOOL "Debug mode")

set(BUILD_TESTING OFF CACHE BOOL "Build tests")

if(${BUILD_TESTING} STREQUAL "OFF")
  set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Kokkos tests")
else()
  set(Kokkos_ENABLE_TESTS ON CACHE BOOL "Kokkos tests")
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${DEBUG} STREQUAL "OFF")
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "Kokkos aggressive vectorization")
  set(Kokkos_ENABLE_COMPILER_WARNINGS OFF CACHE BOOL "Kokkos compiler warnings")
  set(Kokkos_ENABLE_DEBUG OFF CACHE BOOL "Kokkos debug")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK OFF CACHE BOOL "Kokkos debug bounds check")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
  message(STATUS "Release mode")
else()
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION OFF CACHE BOOL "Kokkos aggressive vectorization")
  set(Kokkos_ENABLE_COMPILER_WARNINGS ON CACHE BOOL "Kokkos compiler warnings")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Kokkos debug")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "Kokkos debug bounds check")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "CMake build type")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  message(STATUS "Debug mode")
endif()

if(${Kokkos_ENABLE_CUDA})
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable CUDA lambda")
  set(ADIOS2_USE_CUDA ON CACHE BOOL "Enable ADIOS2 CUDA support")
endif()

set(output OFF CACHE BOOL "Enable output")

if(${output} STREQUAL "ON")
  set(ADIOS2_USE_Python OFF CACHE BOOL "Use Python for ADIOS2")
  set(ADIOS2_USE_Fortran OFF CACHE BOOL "Use Fortran for ADIOS2")
  set(ADIOS2_BUILD_EXAMPLES OFF CACHE BOOL "Build ADIOS2 examples")
  add_compile_options("-D OUTPUT_ENABLED")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/adios2 extern/adios2)
  if(${mpi} STREQUAL "ON")
  set(ADIOS2_USE_MPI ON CACHE BOOL "Use MPI for ADIOS2")
else()
  set(ADIOS2_USE_MPI OFF CACHE BOOL "Use MPI for ADIOS2")
endif()
endif()

# set(mpi OFF CACHE BOOL "Enable MPI")

if(${mpi} STREQUAL "ON")
  add_compile_options("-D MPI_ENABLED")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/kokkos extern/kokkos)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src src)
