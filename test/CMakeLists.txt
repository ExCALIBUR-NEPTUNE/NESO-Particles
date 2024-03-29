cmake_minimum_required(VERSION 3.14)
project(NESO-Particles VERSION 0.0.1)

# GoogleTest requires at least C++11 DataParallel C++ requires at least C++17
set(CMAKE_CXX_STANDARD 17)

add_definitions(-DGPU_SELECTOR=0)

find_package(GTest QUIET)

if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

set(EXECUTABLE testNESOParticles)
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# List test source files
file(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)

# options common to both fftw/mkl
set(LINK_LIBRARIES_COMMON
    ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${NESO_PARTICLES_LIBRARIES}
    ${NESO_PARTICLES_LINK_FLAGS} GTest::gtest)

set(INCLUDE_DIRECTORIES_COMMON
    ${CMAKE_SOURCE_DIR}/include ${MPI_CXX_INCLUDE_PATH}
    ${NESO_PARTICLES_INCLUDE_PATH})

# Build the tests individually
include(GoogleTest)
foreach(TEST ${TEST_SRCS})
  get_filename_component(TEST_NAME ${TEST} NAME_WLE)
  message(STATUS "Found test - ${TEST_NAME}")
  set(TEST_LIST ${TEST_LIST} ${TEST})

  set(TEST_SOURCES ${TEST_MAIN} ${TEST})
  add_executable(${TEST_NAME} ${TEST_SOURCES})

  target_link_libraries(${TEST_NAME} PRIVATE ${LINK_LIBRARIES_COMMON})
  target_include_directories(${TEST_NAME} PRIVATE ${INCLUDE_DIRECTORIES_COMMON})

  add_sycl_to_target(TARGET ${TEST_NAME} SOURCES ${TEST_SOURCES})
  gtest_add_tests(TARGET ${TEST_NAME})
endforeach()

# Build a global test suite
add_executable(${EXECUTABLE} ${TEST_MAIN} ${TEST_LIST})

target_include_directories(${EXECUTABLE} PRIVATE ${INCLUDE_DIRECTORIES_COMMON})
target_link_libraries(${EXECUTABLE} PRIVATE ${LINK_LIBRARIES_COMMON})

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN} ${TEST_LIST})
gtest_add_tests(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN})

# install the combined test binary
install(TARGETS ${EXECUTABLE} DESTINATION bin)
