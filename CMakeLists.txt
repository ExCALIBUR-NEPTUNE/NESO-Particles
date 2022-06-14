cmake_minimum_required(VERSION 3.14)

project(
  NESO-Particles
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# find MPI
find_package(MPI REQUIRED)

# Include hipSYCL
find_package(hipSYCL 0.9.2 QUIET)
if(NOT hipSYCL_FOUND)
  message(WARNING "hipsycl not found")
  find_package(IntelDPCPP REQUIRED)
else()
  message(STATUS "hipsycl found")
  set(HIPSYCL_TARGETS "omp")
endif(NOT hipSYCL_FOUND)

enable_testing()

# put all targets in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
# put all libraries in lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Include test directory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
