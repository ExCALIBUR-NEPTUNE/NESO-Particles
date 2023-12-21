cmake_minimum_required(VERSION 3.24)

project(
  NESO-Particles
  VERSION 0.0.1
  LANGUAGES CXX C)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/restrict-keyword.cmake)

option(ENABLE_NESO_PARTICLES_TESTS "Build unit tests for this project." ON)

#Create interface/Header onlu library
add_library(NesoParticles INTERFACE)
#Alias the name to the namespaces name.
#Can use in subdirectory or via Confiig files with namespace
add_library(${PROJECT_NAME}::NesoParticles ALIAS NesoParticles)

#Set standard
set_property(TARGET NesoParticles PROPERTY CXX_STANDARD 17)
#Makes it easy to install + addds the files to the INCLUDE property of the lib
target_sources(NesoParticles
    PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cell_dat.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/access.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/boundary_conditions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cartesian_mesh.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cell_binning.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cell_dat_compression.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cell_dat.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cell_dat_move.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cell_dat_move_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/communication.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/compute_target.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/containers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/departing_particle_identification.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/departing_particle_identification_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/domain.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/global_mapping.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/global_mapping_impl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/global_move_exchange.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/global_move.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/local_mapping.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/local_move.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/loop
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mesh_hierarchy.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mesh_interface.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mesh_interface_local_decomp.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/neso_particles.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/packing_unpacking.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/parallel_initialisation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/particle_dat.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/particle_group.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/particle_io.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/particle_remover.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/particle_set.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/particle_spec.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/particle_sub_group.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/profiling.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/typedefs.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utility.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utility_mesh_hierarchy_plotting.hpp
    )


if(NESO_PARTICLES_DEVICE_TYPE STREQUAL GPU)
    target_compile_definitions(NesoParticles INTERFACE NESO_PARTICLES_DEVICE_TYPE_GPU)
    add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_GPU)
    message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_GPU")
else()
    target_compile_definitions(NesoParticles INTERFACE NESO_PARTICLES_DEVICE_TYPE_CPU)
    message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_CPU")
endif()



#find a sensible SYCL (assuming adaptivecpp or intelsycl 
#(need a newer intel oneapi suite so can use the the intesycl package that
#behaves the same way as hip|openSycl|AdaptiveCPP re. add_sycl_to_target)
if (NOT DEFINED NESO_PARTICLES_SYCL)
    set(NESO_PARTICLES_SYCL "ACPP" CACHE STRING "INTEL|ACPP")
else()
    set(NESO_PARTICLES_SYCL ${NESO_PARTICLES_SYCL} CACHE STRING "INTEL|ACPP")
endif()

if (${NESO_PARTICLES_SYCL_VENDOR} MATCHES "INTEL")
    find_package(IntelSYCL REQUIRED)
    #    add_compile_options("$<$<CONFIG:Release>:-O3;-xHost;-ffast-math;-g>")
    #Intel compiler can complains about comapring to NAN in ffast-math 
    #add_compile_options(-Wno-tautological-constant-compare)
else()
    #add_compile_options("$<$<CONFIG:Release>:-O3;-march=native;-ffast-math;-g>")
    if(NESO_PARTICLES_DEVICE_TYPE STREQUAL GPU)
        set(ACPP_TARGETS "omp;cuda:sm_80")
    else()
        set(ACPP_TARGETS "omp")
    endif()
    find_package(AdaptiveCPP CONFIG REQUIRED)

    if(NOT ACPP_DEBUG_LEVEL)
        if(CMAKE_BUILD_TYPE MATCHES "Debug")
            set(ACPP_DEBUG_LEVEL 3 CACHE STRING
                "Choose the debug level, options are: 
                0 (no debug), 
                1 (print errors), 
                2 (also print warnings), 
                3 (also print general information)"
                FORCE)
        else()
            set(ACPP_DEBUG_LEVEL 0 CACHE STRING
                "Choose the debug level, options are: 
                0 (no debug), 
                1 (print errors), 
                2 (also print warning s), 
                3 (also print general information)"
                FORCE)
        endif()
    endif()
endif()

#Get MPI 
find_package(MPI REQUIRED)
target_link_libraries(NesoParticles INTERFACE MPI::MPI_CXX)
#Get HDF5
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 QUIET)
if(HDF5_FOUND AND HDF5_IS_PARALLEL)
    message(STATUS "HDF5 found")
    target_link_libraries(NesoParticles INTERFACE HDF5::HDF5)
    target_compile_definitions(NesoParticles INTERFACE NESO_PARTICLES_HDF5)
else()
    message(STATUS "HDF5 NOT found")
endif()
message(STATUS "HDF5_IS_PARALLEL " ${HDF5_IS_PARALLEL})

if(ENABLE_NESO_PARTICLES_TESTS)
  # set build type
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    message(STATUS "Set CMAKE_BUILD_TYPE=RelWithDebInfo")
  endif()
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

#install the (header lib) and export the dependancies
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion)
#
install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/cmake)
#
install(EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Config.cmake
    NAMESPACE
    ${PROJECT_NAME}::
    DESTINATION
    lib/cmake)
export(TARGETS NesoParticles FILE ${PROJECT_NAME}Config.cmake)

install(TARGETS NesoParticles 
    EXPORT ${PROJECT_NAME}_Targets
    FILE_SET public_headers
    DESTINATION include)

