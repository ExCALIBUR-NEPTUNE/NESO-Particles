cmake_minimum_required(VERSION 3.24)

project(
  NESO-Particles
  VERSION 0.10.0
  LANGUAGES CXX C)

include(GNUInstallDirs)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/restrict-keyword.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check-file-list.cmake)

option(NESO_PARTICLES_ENABLE_HDF5 "Add HDF5 to targets." ON)
option(
  NESO_PARTICLES_ENABLE_FIND_SYCL
  "Enabling search for a SYCL implementation if add_sycl_to_target is not found."
  ON)
option(NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
       "Option to use SYCL 1.2 selectors if 2020 ones are not supported." OFF)
option(NESO_PARTICLES_ENABLE_PETSC "Look for PETSc." OFF)
option(
  NESO_PARTICLES_DEVICE_AWARE_MPI
  "Enable device aware MPI by default, i.e. pass device pointers to MPI rather
  than copying to the host." OFF)
option(
  NESO_PARTICLES_SINGLE_COMPILED_LOOP
  "By default, for most backends, NESO-Particles will compile two parallel
  loops for each ParticleLoop, one with and one without conditionals which mask
  off the end of the parallel iteration set. This means that the compiler takes
  longer and consumes more RAM in exhchange for hopefully faster loops. This
  flag explicitly disables building two loops and produces a single parallel
  loop for each ParticleLoop."
  OFF)

# This means that when the tests and lib are installed they have rpath set for
# the installed lib/binary.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Find SYCL
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SYCL.cmake)
if(NESO_PARTICLES_ENABLE_FIND_SYCL)
  find_sycl_if_required()
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SRC_FILES
    ${SRC_DIR}/algorithms/cellwise_methods.cpp
    ${SRC_DIR}/algorithms/dsmc/pair_sampler_ntc.cpp
    ${SRC_DIR}/algorithms/particle_data_movement.cpp
    ${SRC_DIR}/algorithms/reduce_dat_cellwise.cpp
    ${SRC_DIR}/algorithms/unseen_value_extractor.cpp
    ${SRC_DIR}/external_interfaces/petsc/boundary_interaction/boundary_interaction_2d.cpp
    ${SRC_DIR}/external_interfaces/petsc/boundary_interaction/boundary_interaction_common.cpp
    ${SRC_DIR}/external_interfaces/petsc/dmplex_cell_serialise.cpp
    ${SRC_DIR}/external_interfaces/petsc/dmplex_interface.cpp
    ${SRC_DIR}/external_interfaces/petsc/dmplex_helper.cpp
    ${SRC_DIR}/external_interfaces/petsc/dmplex_mesh_coupler_dg0.cpp
    ${SRC_DIR}/external_interfaces/petsc/dmplex_utility.cpp
    ${SRC_DIR}/external_interfaces/petsc/particle_cell_mapping/dmplex_2d_mapper.cpp
    ${SRC_DIR}/external_interfaces/petsc/particle_cell_mapping/dmplex_host_mapper.cpp
    ${SRC_DIR}/external_interfaces/petsc/project_evaluate/dmplex_project_evaluate_barycentric.cpp
    ${SRC_DIR}/external_interfaces/petsc/project_evaluate/dmplex_project_evaluate_dg.cpp
    ${SRC_DIR}/external_interfaces/common/dof_mapper_dg.cpp
    ${SRC_DIR}/external_interfaces/common/local_claim.cpp
    ${SRC_DIR}/external_interfaces/common/overlay_cartesian_mesh.cpp
    ${SRC_DIR}/external_interfaces/common/quadrature_point_mapper.cpp
    ${SRC_DIR}/external_interfaces/vtk/vtk.cpp
    ${SRC_DIR}/boundary/boundary_interaction_specification.cpp
    ${SRC_DIR}/boundary/boundary_mesh_interface.cpp
    ${SRC_DIR}/boundary/boundary_reflection.cpp
    ${SRC_DIR}/boundary/boundary_truncation.cpp
    ${SRC_DIR}/particle_io.cpp
    ${SRC_DIR}/cartesian_mesh.cpp
    ${SRC_DIR}/cartesian_mesh/cartesian_cell_bin.cpp
    ${SRC_DIR}/cartesian_mesh/cartesian_h_mesh.cpp
    ${SRC_DIR}/cartesian_mesh/cartesian_h_mesh_function.cpp
    ${SRC_DIR}/cartesian_mesh/cartesian_periodic.cpp
    ${SRC_DIR}/cartesian_mesh/cartesian_trajectory_intersection.cpp
    ${SRC_DIR}/communication/communication_edges_counter.cpp
    ${SRC_DIR}/communication/communication_utility.cpp
    ${SRC_DIR}/compute_target.cpp
    ${SRC_DIR}/containers/cell_dat.cpp
    ${SRC_DIR}/containers/cell_dat_const.cpp
    ${SRC_DIR}/containers/descendant_products.cpp
    ${SRC_DIR}/containers/ephemeral_dats.cpp
    ${SRC_DIR}/containers/global_array.cpp
    ${SRC_DIR}/containers/local_array.cpp
    ${SRC_DIR}/containers/particle_set_device.cpp
    ${SRC_DIR}/containers/sym_vector.cpp
    ${SRC_DIR}/containers/sym_vector_pointer_cache.cpp
    ${SRC_DIR}/containers/sym_vector_pointer_cache_dispatch.cpp
    ${SRC_DIR}/cell_dat_compression.cpp
    ${SRC_DIR}/cell_dat_move.cpp
    ${SRC_DIR}/device_limits.cpp
    ${SRC_DIR}/departing_particle_identification.cpp
    ${SRC_DIR}/mesh_hierarchy.cpp
    ${SRC_DIR}/global_mapping.cpp
    ${SRC_DIR}/parallel_initialisation.cpp
    ${SRC_DIR}/packing_unpacking.cpp
    ${SRC_DIR}/pair_loop/cellwise_pair_list.cpp
    ${SRC_DIR}/pair_loop/cellwise_pair_list_host.cpp
    ${SRC_DIR}/pair_loop/cellwise_pair_list_block.cpp
    ${SRC_DIR}/particle_dat.cpp
    ${SRC_DIR}/particle_group.cpp
    ${SRC_DIR}/particle_set.cpp
    ${SRC_DIR}/loop/particle_loop_args.cpp
    ${SRC_DIR}/loop/particle_loop_base.cpp
    ${SRC_DIR}/loop/particle_loop_iteration_set.cpp
    ${SRC_DIR}/loop/particle_loop_index.cpp
    ${SRC_DIR}/loop/particle_loop_utility.cpp
    ${SRC_DIR}/loop/pli_particle_dat.cpp
    ${SRC_DIR}/particle_sub_group/cell_sub_group_selector.cpp
    ${SRC_DIR}/particle_sub_group/truncate_sub_group_selector.cpp
    ${SRC_DIR}/particle_sub_group/disjoint_union_sub_group_selector.cpp
    ${SRC_DIR}/particle_sub_group/discard_sub_group_selector.cpp
    ${SRC_DIR}/particle_sub_group/copy_selector.cpp
    ${SRC_DIR}/particle_sub_group/particle_group_partition.cpp
    ${SRC_DIR}/particle_sub_group/particle_sub_group_base.cpp
    ${SRC_DIR}/particle_sub_group/sub_group_particle_map.cpp
    ${SRC_DIR}/particle_sub_group/sub_group_selector.cpp
    ${SRC_DIR}/particle_sub_group/sub_group_selector_base.cpp
    ${SRC_DIR}/particle_sub_group/sub_group_selector_exclusive_scan.cpp
    ${SRC_DIR}/particle_sub_group/sub_group_selector_utility.cpp
    ${SRC_DIR}/particle_group_temporary.cpp
    ${SRC_DIR}/particle_group_pointer_map.cpp
    ${SRC_DIR}/utility.cpp
    ${SRC_DIR}/utility_mesh_hierarchy_plotting.cpp)
set(SRC_FILES_IGNORE "")
check_added_file_list(${SRC_DIR} cpp "${SRC_FILES}" "${SRC_FILES_IGNORE}")

# Create a list of the header files.
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INCLUDE_DIR_NESO_PARTICLES ${INCLUDE_DIR}/neso_particles)
set(HEADER_FILES
    ${INCLUDE_DIR}/neso_particles.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/access.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/algorithms/algorithms.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/algorithms/dsmc/dsmc.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/algorithms/dsmc/cellwise_all_to_all.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/algorithms/dsmc/pair_sampler_ntc.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/algorithms/cellwise_methods.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/algorithms/particle_data_movement.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/algorithms/reduce_dat_cellwise.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/algorithms/unseen_value_extractor.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cartesian_mesh/cartesian_h_mesh.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cartesian_mesh/cartesian_h_mesh_function.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cartesian_mesh/cartesian_cell_bin.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cartesian_mesh/cartesian_periodic.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cartesian_mesh/cartesian_trajectory_intersection.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/boundary/boundary_interaction_specification.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/boundary/boundary_mesh_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/boundary/boundary_reflection.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/boundary/boundary_truncation.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cartesian_mesh.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cell_dat.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cell_dat_compression.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/cell_dat_move.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/common_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication/comm_pair.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication/communication_edges_counter.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication/communication_typedefs.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication/communication_utility.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/communication/global_move_communication.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/compute_target.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/blocked_binary_tree.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/cell_dat.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/cell_dat_const.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/cell_data.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/ephemeral_dats.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/nd_index.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/rng/rng.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/rng/kernel_rng.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/rng/host_rng_common.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/rng/host_per_particle_block_rng.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/rng/host_atomic_block_kernel_rng.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/rng/null_kernel_rng.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/rng/tuple_rng.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/descendant_products.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/global_array.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/local_array.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/local_memory_block.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/local_memory_interlaced.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/lookup_table.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/nd_local_array.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/particle_set_device.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/product_matrix.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/resource_stack.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/resource_stack_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/resource_stack_map.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/sym_vector.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/sym_vector_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/sym_vector_utility.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/sym_vector_pointer_cache.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/sym_vector_pointer_cache_dispatch.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/sym_vector_pointer_cache_dispatch_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/containers/tuple.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/departing_particle_identification.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/device_atomic_sanity_check.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/device_buffers.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/device_functions.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/device_limits.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/domain.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/error_propagate.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/bounding_box.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/common.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/coordinate_mapping.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/local_claim.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/overlay_cartesian_mesh.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/dof_mapper_dg.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/common/quadrature_point_mapper.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/boundary_interaction/boundary_interaction.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/boundary_interaction/boundary_interaction_2d.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/boundary_interaction/boundary_interaction_common.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_cell_serialise.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_helper.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_local_mapper.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_mesh_coupler_dg0.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_project_evaluate.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/dmplex_utility.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/petsc_api_redirection.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/particle_cell_mapping/dmplex_2d_mapper.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/particle_cell_mapping/dmplex_host_mapper.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/petsc_common.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/petsc_utility.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/petsc_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/project_evaluate/dmplex_project_evaluate_base.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/project_evaluate/dmplex_project_evaluate_barycentric.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/petsc/project_evaluate/dmplex_project_evaluate_dg.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/external_interfaces/vtk/vtk.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/global_mapping.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/global_move.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/global_move_exchange.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/local_mapping.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/local_move.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/access_descriptors.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/cell_info_npart.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/kernel.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_args.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_base.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_functions.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_utility.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_index.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_iteration_set.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_args_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/particle_loop_reduction.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/loop/pli_particle_dat.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy_data/mesh_hierarchy_container.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy_data/mesh_hierarchy_data.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy_data/serial_container.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_hierarchy_data/serial_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/mesh_interface_local_decomp.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/packing_unpacking.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/cellwise_pair_list.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/cellwise_pair_list_absolute.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/cellwise_pair_list_block.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/cellwise_pair_list_host.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/pair_loop.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/particle_pair_loop_args.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/particle_pair_loop_base.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/particle_pair_loop_index.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/particle_pair_loop_cellwise_pair_list.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/pair_loop/particle_pair_loop_cellwise_pair_list_block.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/parallel_initialisation.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/parameters.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_dat.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_dat_direct_access.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_group.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_group_impl.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_group_pointer_map.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_group_temporary.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_io.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_remover.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_set.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_spec.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/copy_selector.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/particle_group_partition.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/particle_sub_group.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/particle_sub_group_base.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/particle_sub_group_utility.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/sub_group_particle_map.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/sub_group_selector.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/sub_group_selector_exclusive_scan.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/sub_group_selector_whole_group.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/sub_group_selector_base.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/sub_group_selector_utility.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/sub_group_selector_resource_stack_interface.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/particle_loop_sub_group.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/particle_loop_sub_group_functions.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/particle_loop_sub_group_reduction.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/cell_sub_group_selector.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/discard_sub_group_selector.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/truncate_sub_group_selector.hpp
    ${INCLUDE_DIR_NESO_PARTICLES}/particle_sub_group/disjoint_union_sub_group_selector.hpp
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

# Create shared library
add_library(NESO-Particles SHARED ${SRC_FILES} ${HEADER_FILES})
# Alias the name to the namespaces name. Can use in subdirectory or via Confiig
# files with namespace
add_library(NESO-Particles::NESO-Particles ALIAS NESO-Particles)

# Set standard
set_property(TARGET NESO-Particles PROPERTY CXX_STANDARD 17)

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

# Don't like this .... TODO: FIXME: Should be a runtime thing?
if(NESO_PARTICLES_DEVICE_TYPE STREQUAL GPU)
  target_compile_definitions(NESO-Particles
                             PUBLIC NESO_PARTICLES_DEVICE_TYPE_GPU)
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_GPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_GPU")
else()
  target_compile_definitions(NESO-Particles
                             PUBLIC NESO_PARTICLES_DEVICE_TYPE_CPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_CPU")
endif()

if(NESO_PARTICLES_LEGACY_DEVICE_SELECTORS)
  target_compile_definitions(NESO-Particles
                             PUBLIC NESO_PARTICLES_LEGACY_DEVICE_SELECTORS)
  message(STATUS "Using legacy SYCL device selectors")
endif()

if(NESO_PARTICLES_DEVICE_AWARE_MPI)
  target_compile_definitions(NESO-Particles
                             PUBLIC NESO_PARTICLES_DEVICE_AWARE_MPI)
  message(STATUS "Using device aware MPI.")
endif()

if(NESO_PARTICLES_SINGLE_COMPILED_LOOP)
  target_compile_definitions(NESO-Particles
                             PUBLIC NESO_PARTICLES_SINGLE_COMPILED_LOOP)
  message(STATUS "Using single compiled loop")
endif()

# Get MPI
find_package(MPI REQUIRED)
target_link_libraries(NESO-Particles PUBLIC MPI::MPI_CXX)

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
    target_link_libraries(NESO-Particles PUBLIC HDF5::HDF5)
    target_compile_definitions(NESO-Particles PUBLIC NESO_PARTICLES_HDF5)
    set(NESO_PARTICLES_USING_HDF5 TRUE)
  else()
    message("HDF5 NOT found")
  endif()
endif()

# set the restrict keyword
target_compile_definitions(NESO-Particles PUBLIC RESTRICT=${RESTRICT})
# set the version infomation
target_compile_definitions(
  NESO-Particles
  PUBLIC NESO_PARTICLES_VERSION_MAJOR=${CMAKE_PROJECT_VERSION_MAJOR})
target_compile_definitions(
  NESO-Particles
  PUBLIC NESO_PARTICLES_VERSION_MINOR=${CMAKE_PROJECT_VERSION_MINOR})
target_compile_definitions(
  NESO-Particles
  PUBLIC NESO_PARTICLES_VERSION_PATCH=${CMAKE_PROJECT_VERSION_PATCH})

# now look for PETSc
if(NESO_PARTICLES_ENABLE_PETSC)
  find_package(PkgConfig)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PETSC PETSc)
  endif()

  if(PETSC_FOUND)
    message(STATUS "PETSc found")
    # TODO CHECK AND CLEANUP
    target_compile_definitions(NESO-Particles PUBLIC NESO_PARTICLES_PETSC
                                                     ${PETSC_DEFINITIONS})
    add_definitions(-DNESO_PARTICLES_PETSC ${PETSC_DEFINITIONS})
    target_link_libraries(NESO-Particles PUBLIC ${PETSC_LINK_LIBRARIES})
    target_include_directories(NESO-Particles PUBLIC ${PETSC_INCLUDE_DIRS})
    message(STATUS PETSC_LINK_LIBRARIES ${PETSC_LINK_LIBRARIES})
    message(STATUS PETSC_LDFLAGS ${PETSC_LDFLAGS})
    message(STATUS PETSC_INCLUDE_DIRS ${PETSC_INCLUDE_DIRS})
    message(STATUS PETSC_LIBRARY_DIRS ${PETSC_LIBRARY_DIRS})
    message(STATUS PETSC_CFLAGS ${PETSC_CFLAGS})
  else()
    message(STATUS "PETSc NOT found")
  endif()
endif()

option(NESO_PARTICLES_ENABLE_TESTS "Build unit tests for this project." ON)

add_sycl_to_target(TARGET NESO-Particles SOURCES ${SRC_FILES})

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
