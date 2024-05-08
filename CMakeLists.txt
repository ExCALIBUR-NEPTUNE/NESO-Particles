cmake_minimum_required(VERSION 3.24)

project(
  NESO-Particles
  VERSION 0.0.1
  LANGUAGES CXX C)

include(GNUInstallDirs)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/restrict-keyword.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check-file-list.cmake)

option(NESO_PARTICLES_ENABLE_HDF5 "Add HDF5 to targets." ON)
option(
  NESO_PARTICLES_ENABLE_FIND_SYCL
  "Enabling search for a SYCL implementation if add_sycl_to_target is not found."
  ON)

# Create interface/Header only library
add_library(NESO-Particles INTERFACE)
# Alias the name to the namespaces name. Can use in subdirectory or via Confiig
# files with namespace
add_library(NESO-Particles::NESO-Particles ALIAS NESO-Particles)

# Set standard
set_property(TARGET NESO-Particles PROPERTY CXX_STANDARD 17)

# Create a list of the header files.
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INCLUDE_DIR_NESO_PARTICLES ${INCLUDE_DIR}/neso_particles)
set(HEADER_FILES
    ${INCLUDE_DIR}/neso_particles.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/access.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/boundary_conditions.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cartesian_mesh.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cell_binning.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cell_dat.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cell_dat_compression.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cell_dat_move.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cell_dat_move_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication/comm_pair.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication/communication_edges_counter.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication/communication_typedefs.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/compute_target.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/blocked_binary_tree.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/cell_dat.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/cell_dat_const.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/cell_data.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/descendant_products.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/global_array.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/local_array.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/lookup_table.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/product_matrix.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/sym_vector.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/tuple.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/departing_particle_identification.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/departing_particle_identification_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/domain.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/error_propagate.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/bounding_box.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/local_claim.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/overlay_cartesian_mesh.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_cell_serialise.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_helper.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_local_mapper.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/particle_cell_mapping/dmplex_2d_mapper.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/particle_cell_mapping/dmplex_host_mapper.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/petsc_common.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/petsc_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/global_mapping.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/global_mapping_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/global_move.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/global_move_exchange.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/local_mapping.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/local_move.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/access_descriptors.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_base.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_index.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/pli_particle_dat.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy_data/mesh_hierarchy_container.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy_data/mesh_hierarchy_data.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy_data/serial_container.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy_data/serial_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_interface_local_decomp.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/packing_unpacking.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/parallel_initialisation.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_dat.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_group.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_group_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_io.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_remover.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_set.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_spec.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/profiling.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/sycl_typedefs.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/typedefs.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/utility.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/utility_mesh_hierarchy_plotting.hpp)

# Check that the files added above are not missing any files in the include
# directory.
set(HEADER_FILES_IGNORE "")
check_added_file_list(${INCLUDE_DIR} hpp "${HEADER_FILES}"
                      "${HEADER_FILES_IGNORE}")

# Makes it easy to install + adds the files to the INCLUDE property of the lib
# i.e. don't need target_include_dir.. also no GLOBS
target_sources(
  NESO-Particles
  PUBLIC FILE_SET
         public_headers
         TYPE
         HEADERS
         BASE_DIRS
         ${INCLUDE_DIR}
         FILES
         ${HEADER_FILES})

# target_include_directories( NESO-Particles INTERFACE
# $<BUILD_INTERFACE:${INCLUDE_DIR}> $<INSTALL_INTERFACE:include> )

# Don't like this .... TODO: FIXME: Should be a runtime thing?
if(NESO_PARTICLES_DEVICE_TYPE STREQUAL GPU)
  target_compile_definitions(NESO-Particles
                             INTERFACE NESO_PARTICLES_DEVICE_TYPE_GPU)
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_GPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_GPU")
else()
  target_compile_definitions(NESO-Particles
                             INTERFACE NESO_PARTICLES_DEVICE_TYPE_CPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_CPU")
endif()

# Get MPI
find_package(MPI REQUIRED)
target_link_libraries(NESO-Particles INTERFACE MPI::MPI_CXX)

# Get HDF5 if its around
set(NESO_PARTICLES_USING_HDF5 FALSE)
if(NESO_PARTICLES_ENABLE_HDF5)
  set(HDF5_PREFER_PARALLEL TRUE)
  # HDF5 requires MPI::MPI_C, which is only loaded if C is enabled. See
  # https://github.com/HDFGroup/hdf5/issues/3844.  Note that this issue only
  # arises for certain versions of HDF5.
  enable_language(C)
  find_package(HDF5 REQUIRED)
  if(HDF5_FOUND AND HDF5_IS_PARALLEL)
    message(STATUS "Parallel HDF5 found")
    target_link_libraries(NESO-Particles INTERFACE HDF5::HDF5 NESO-Particles)
    target_compile_definitions(NESO-Particles INTERFACE NESO_PARTICLES_HDF5)
    set(NESO_PARTICLES_USING_HDF5 TRUE)
  else()
    message("HDF5 NOT found")
  endif()
endif()

# now look for PETSc
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PETSC PETSc)
endif()

if(PETSC_FOUND)
  message(STATUS "PETSc found")
  # TODO CHECK AND CLEANUP
  target_compile_definitions(NESO-Particles INTERFACE NESO_PARTICLES_PETSC
                                                      ${PETSC_DEFINITIONS})
  add_definitions(-DNESO_PARTICLES_PETSC ${PETSC_DEFINITIONS})
  target_link_libraries(NESO-Particles INTERFACE ${PETSC_LINK_LIBRARIES})
  target_include_directories(NESO-Particles INTERFACE ${PETSC_INCLUDE_DIRS})
  message(STATUS PETSC_LINK_LIBRARIES ${PETSC_LINK_LIBRARIES})
  message(STATUS PETSC_LDFLAGS ${PETSC_LDFLAGS})
  message(STATUS PETSC_INCLUDE_DIRS ${PETSC_INCLUDE_DIRS})
  message(STATUS PETSC_LIBRARY_DIRS ${PETSC_LIBRARY_DIRS})
  message(STATUS PETSC_CFLAGS ${PETSC_CFLAGS})
else()
  message(STATUS "PETSc NOT found")
endif()

# Find SYCL
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SYCL.cmake)
if(NESO_PARTICLES_ENABLE_FIND_SYCL)
  find_sycl_if_required()
endif()

option(NESO_PARTICLES_ENABLE_TESTS "Build unit tests for this project." ON)

# Include the tests
if(NESO_PARTICLES_ENABLE_TESTS)
  # set build type
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    message(STATUS "Set CMAKE_BUILD_TYPE=RelWithDebInfo")
  endif()
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
endif()

# install the headers
install(
  TARGETS NESO-Particles
  EXPORT ${PROJECT_NAME}_Targets
  FILE_SET public_headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
# create the config.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)
# install the config scripts
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SYCL.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/restrict-keyword.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check-file-list.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
export(
  EXPORT ${PROJECT_NAME}_Targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake"
  NAMESPACE ${PROJECT_NAME}::)
