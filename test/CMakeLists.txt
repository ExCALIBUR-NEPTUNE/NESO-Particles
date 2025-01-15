# No need should be inherited from neso-particles set(CMAKE_CXX_STANDARD 17)

# add_definitions(-DGPU_SELECTOR=0)

find_package(GTest QUIET)

if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
        DOWNLOAD_EXTRACT_TIMESTAMP
        FALSE)
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_SRCS
    ${TEST_DIR}/test_boundary_pbc.cpp
    ${TEST_DIR}/test_buffers.cpp
    ${TEST_DIR}/test_cell_dat.cpp
    ${TEST_DIR}/test_communication.cpp
    ${TEST_DIR}/test_device_functions.cpp
    ${TEST_DIR}/test_error_propagate.cpp
    ${TEST_DIR}/test_examples.cpp
    ${TEST_DIR}/test_external_common.cpp
    ${TEST_DIR}/test_global_array.cpp
    ${TEST_DIR}/test_int_key_value_map.cpp
    ${TEST_DIR}/test_local_array.cpp
    ${TEST_DIR}/test_local_decomp_mesh.cpp
    ${TEST_DIR}/test_mesh_hierarchy.cpp
    ${TEST_DIR}/test_parallel_initialisation.cpp
    ${TEST_DIR}/test_petsc_interface.cpp
    ${TEST_DIR}/test_petsc_boundary_interface.cpp
    ${TEST_DIR}/test_petsc_project_evaluate.cpp
    ${TEST_DIR}/test_petsc_utility.cpp
    ${TEST_DIR}/test_particle_dat.cpp
    ${TEST_DIR}/test_particle_group_cell_move.cpp
    ${TEST_DIR}/test_particle_group.cpp
    ${TEST_DIR}/test_particle_group_global_move.cpp
    ${TEST_DIR}/test_particle_group_hybrid_move.cpp
    ${TEST_DIR}/test_particle_group_local_move.cpp
    ${TEST_DIR}/test_particle_group_stencil_move.cpp
    ${TEST_DIR}/test_particle_h5part.cpp
    ${TEST_DIR}/test_particle_loop.cpp
    ${TEST_DIR}/test_particle_loop_iteration_set.cpp
    ${TEST_DIR}/test_particle_loop_local_array.cpp
    ${TEST_DIR}/test_particle_loop_local_memory.cpp
    ${TEST_DIR}/test_particle_loop_rng.cpp
    ${TEST_DIR}/test_particle_loop_sym_vector.cpp
    ${TEST_DIR}/test_particle_remover.cpp
    ${TEST_DIR}/test_particle_set.cpp
    ${TEST_DIR}/test_particle_sub_group.cpp
    ${TEST_DIR}/test_product_matrix.cpp
    ${TEST_DIR}/test_serial_container.cpp
    ${TEST_DIR}/test_sycl_target.cpp
    ${TEST_DIR}/test_tuple.cpp
    ${TEST_DIR}/test_utility.cpp)

# Check that the files added above are not missing any files in the test
# directory.
set(TEST_SRCS_IGNORE ${TEST_DIR}/main.cpp)
check_added_file_list(${CMAKE_CURRENT_SOURCE_DIR} cpp "${TEST_SRCS}"
                      "${TEST_SRCS_IGNORE}")

set(EXECUTABLE testNESOParticles)
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
# Build the tests individually
include(GoogleTest)
# Add a a target so we can build all the tests "make tests"
add_custom_target(tests)
foreach(TEST ${TEST_SRCS})
  get_filename_component(TEST_NAME ${TEST} NAME_WLE)
  message(STATUS "Found test - ${TEST_NAME}")
  set(TEST_LIST ${TEST_LIST} ${TEST})

  set(TEST_SOURCES ${TEST_MAIN} ${TEST})
  add_executable(${TEST_NAME} ${TEST_SOURCES})
  target_link_libraries(${TEST_NAME} PRIVATE NESO-Particles GTest::gtest)
  target_compile_definitions(${TEST_NAME} PRIVATE GPU_SELECTOR=0)
  target_compile_definitions(
    ${TEST_NAME} PRIVATE CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})
  target_compile_options(${TEST_NAME} PRIVATE "-Wpedantic;-Wall;-Wextra")
  # exclude from all so indiviual tests aren't built for install
  set_target_properties(${TEST_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  # Make usre the dummy target depends on each tests
  add_dependencies(tests ${TEST_NAME})
  add_sycl_to_target(TARGET ${TEST_NAME} SOURCES ${TEST_SOURCES})
  gtest_add_tests(TARGET ${TEST_NAME})

  set_target_properties(
    ${TEST_NAME}
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)

endforeach()

# Build a global test suite
add_executable(${EXECUTABLE} ${TEST_MAIN} ${TEST_LIST})

target_link_libraries(${EXECUTABLE} PRIVATE NESO-Particles GTest::gtest)
target_compile_definitions(${EXECUTABLE} PRIVATE GPU_SELECTOR=0)
target_compile_definitions(${EXECUTABLE}
                           PRIVATE CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX})
target_compile_options(${EXECUTABLE} PRIVATE "-Wpedantic;-Wall;-Wextra")

set_target_properties(
  ${EXECUTABLE}
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)

# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN} ${TEST_LIST})
gtest_add_tests(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN})

# install the combined test binary
install(TARGETS ${EXECUTABLE} DESTINATION bin)
