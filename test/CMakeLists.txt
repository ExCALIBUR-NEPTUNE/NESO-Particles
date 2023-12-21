#No need should be inherited from neso-particles
#set(CMAKE_CXX_STANDARD 17)

#add_definitions(-DGPU_SELECTOR=0)

find_package(GTest QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/054a986a8513149e8374fc669a5fe40117ca6b41.zip
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
        )
    # For Windows: Prevent overriding the parent project's compiler/linker
    # settings
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()


#Find SYCL
if (COMMAND add_sycl_to_target) 
    #Not perfect method but if in a submodule dont want to use a different
    #sycl
    message("add_sycl_to_target macro exists")
else()
    message("add_sycl_to_target macro NOT FOUND looking for SYCL package...")
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
endif()

set(TEST_SRCS 
    test_boundary_pbc.cpp
    test_buffers.cpp
    test_cell_dat.cpp
    test_error_propagate.cpp
    test_examples.cpp
    test_global_array.cpp
    test_int_key_value_map.cpp
    test_local_array.cpp
    test_local_decomp_mesh.cpp
    test_mesh_hierarchy.cpp
    test_parallel_initialisation.cpp
    test_particle_dat.cpp
    test_particle_group_cell_move.cpp
    test_particle_group.cpp
    test_particle_group_global_move.cpp
    test_particle_group_hybrid_move.cpp
    test_particle_group_local_move.cpp
    test_particle_group_stencil_move.cpp
    test_particle_h5part.cpp
    test_particle_loop.cpp
    test_particle_remover.cpp
    test_particle_set.cpp
    test_particle_sub_group.cpp
    test_sycl_target.cpp
    test_tuple.cpp
    test_utility.cpp)

set(EXECUTABLE testNESOParticles)
set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
# Build the tests individually
include(GoogleTest)
foreach(TEST ${TEST_SRCS})
    get_filename_component(TEST_NAME ${TEST} NAME_WLE)
    message(STATUS "Found test - ${TEST_NAME}")
    set(TEST_LIST ${TEST_LIST} ${TEST})

    set(TEST_SOURCES ${TEST_MAIN} ${TEST})
    add_executable(${TEST_NAME} ${TEST_SOURCES})
    target_link_libraries(${TEST_NAME} PRIVATE NesoParticles GTest::gtest)
    target_compile_definitions(${TEST_NAME} PRIVATE GPU_SELECTOR=0)

    add_sycl_to_target(TARGET ${TEST_NAME} SOURCES ${TEST_SOURCES})
    gtest_add_tests(TARGET ${TEST_NAME})
endforeach()

# Build a global test suite
add_executable(${EXECUTABLE} ${TEST_MAIN} ${TEST_LIST})

target_link_libraries(${EXECUTABLE} PRIVATE NesoParticles GTest::gtest)
target_compile_definitions(${EXECUTABLE} PRIVATE GPU_SELECTOR=0)


# define the test executable as a sycl target
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN} ${TEST_LIST})
gtest_add_tests(TARGET ${EXECUTABLE} SOURCES ${TEST_MAIN})

# install the combined test binary
#install(TARGETS ${EXECUTABLE} DESTINATION bin)
