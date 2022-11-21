cmake_minimum_required(VERSION 3.21)

project(
  NESO-Particles
  VERSION 0.0.1
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

# set build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
  message(STATUS "Set CMAKE_BUILD_TYPE=RelWithDebInfo")
endif()

# find MPI
find_package(MPI REQUIRED)

# find NESO-Particles config
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(NESO-PARTICLES REQUIRED)

# Include hipSYCL
find_package(hipSYCL 0.9.2 QUIET)
if(NOT hipSYCL_FOUND)
  find_package(IntelDPCPP QUIET)
  if(NOT IntelDPCPP_FOUND)
    message(
      WARNING
        "Proceeding on the assumption that the CXX compiler is a SYCL2020 compiler."
    )
  endif()
else()
  message(STATUS "hipsycl found")
  set(HIPSYCL_TARGETS "omp")
endif(NOT hipSYCL_FOUND)

# hipsycl, trisycl and computecpp all define an "add_sycl_to_target" for the
# compilation of a target
if(NOT COMMAND add_sycl_to_target)
  # Note from hipsycl: "Do not call target_sources after add_sycl_to_target or
  # dependency tracking on compiler flags will break in subtle ways"
  function(add_sycl_to_target)

  endfunction()
endif()

if(NESO_PARTICLES_DEVICE_TYPE STREQUAL GPU)
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_GPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_GPU")
else()
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_CPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_CPU")
endif()

enable_testing()

# put all targets in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# put all libraries in lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# install the header files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX})
# install the cmake files
install(DIRECTORY ${CMAKE_SOURCE_DIR}/cmake DESTINATION ${CMAKE_INSTALL_PREFIX})

option(ENABLE_NESO_PARTICLES_TESTS "Build unit tests for this project." ON)

# Include test directory
if(ENABLE_NESO_PARTICLES_TESTS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()
