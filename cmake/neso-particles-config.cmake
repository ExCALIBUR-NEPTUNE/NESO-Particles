include(${CMAKE_CURRENT_LIST_DIR}/restrict-keyword.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/check-file-list.cmake)

option(NESO_PARTICLES_ENABLE_HDF5 "Add HDF5 to targets." ON)
option(NESO_PARTICLES_ENABLE_FIND_SYCL "Enabling search for a SYCL implementation if add_sycl_to_target is not found." ON)

#Create interface/Header only library
add_library(NESO-Particles INTERFACE)
#Alias the name to the namespaces name.
#Can use in subdirectory or via Confiig files with namespace
add_library(NESO-Particles::NESO-Particles ALIAS NESO-Particles)

#Set standard
set_property(TARGET NESO-Particles PROPERTY CXX_STANDARD 17)

# Create a list of the header files.
set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include)
set(HEADER_FILES
    ${INCLUDE_DIR}/access.hpp
    ${INCLUDE_DIR}/boundary_conditions.hpp
    ${INCLUDE_DIR}/cartesian_mesh.hpp
    ${INCLUDE_DIR}/cell_binning.hpp
    ${INCLUDE_DIR}/cell_dat.hpp
    ${INCLUDE_DIR}/cell_dat_compression.hpp
    ${INCLUDE_DIR}/cell_dat_move.hpp
    ${INCLUDE_DIR}/cell_dat_move_impl.hpp
    ${INCLUDE_DIR}/communication.hpp
    ${INCLUDE_DIR}/compute_target.hpp
    ${INCLUDE_DIR}/containers/blocked_binary_tree.hpp
    ${INCLUDE_DIR}/containers/cell_dat.hpp
    ${INCLUDE_DIR}/containers/cell_dat_const.hpp
    ${INCLUDE_DIR}/containers/cell_data.hpp
    ${INCLUDE_DIR}/containers/descendant_products.hpp
    ${INCLUDE_DIR}/containers/global_array.hpp
    ${INCLUDE_DIR}/containers/local_array.hpp
    ${INCLUDE_DIR}/containers/product_matrix.hpp
    ${INCLUDE_DIR}/containers/sym_vector.hpp
    ${INCLUDE_DIR}/containers/tuple.hpp
    ${INCLUDE_DIR}/departing_particle_identification.hpp
    ${INCLUDE_DIR}/departing_particle_identification_impl.hpp
    ${INCLUDE_DIR}/domain.hpp
    ${INCLUDE_DIR}/error_propagate.hpp
    ${INCLUDE_DIR}/global_mapping.hpp
    ${INCLUDE_DIR}/global_mapping_impl.hpp
    ${INCLUDE_DIR}/global_move.hpp
    ${INCLUDE_DIR}/global_move_exchange.hpp
    ${INCLUDE_DIR}/local_mapping.hpp
    ${INCLUDE_DIR}/local_move.hpp
    ${INCLUDE_DIR}/loop/access_descriptors.hpp
    ${INCLUDE_DIR}/loop/particle_loop.hpp
    ${INCLUDE_DIR}/loop/particle_loop_base.hpp
    ${INCLUDE_DIR}/loop/particle_loop_index.hpp
    ${INCLUDE_DIR}/loop/pli_particle_dat.hpp
    ${INCLUDE_DIR}/mesh_hierarchy.hpp
    ${INCLUDE_DIR}/mesh_interface.hpp
    ${INCLUDE_DIR}/mesh_interface_local_decomp.hpp
    ${INCLUDE_DIR}/neso_particles.hpp
    ${INCLUDE_DIR}/packing_unpacking.hpp
    ${INCLUDE_DIR}/parallel_initialisation.hpp
    ${INCLUDE_DIR}/particle_dat.hpp
    ${INCLUDE_DIR}/particle_group.hpp
    ${INCLUDE_DIR}/particle_group_impl.hpp
    ${INCLUDE_DIR}/particle_io.hpp
    ${INCLUDE_DIR}/particle_remover.hpp
    ${INCLUDE_DIR}/particle_set.hpp
    ${INCLUDE_DIR}/particle_spec.hpp
    ${INCLUDE_DIR}/particle_sub_group.hpp
    ${INCLUDE_DIR}/profiling.hpp
    ${INCLUDE_DIR}/typedefs.hpp
    ${INCLUDE_DIR}/utility.hpp
    ${INCLUDE_DIR}/utility_mesh_hierarchy_plotting.hpp
)

# Check that the files added above are not missing any files in the include
# directory.
set(HEADER_FILES_IGNORE "")
check_added_file_list(${INCLUDE_DIR} hpp "${HEADER_FILES}" "${HEADER_FILES_IGNORE}")

#Makes it easy to install + adds the files to the INCLUDE property of the lib
#i.e. don't need target_include_dir.. also no GLOBS
target_sources(NESO-Particles
    PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../include
    FILES
    ${HEADER_FILES}
)

target_include_directories(
    NESO-Particles
    INTERFACE
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

#Don't like this .... TODO: FIXME:
#Should be a runtime thing?
if(NESO_PARTICLES_DEVICE_TYPE STREQUAL GPU)
    target_compile_definitions(NESO-Particles INTERFACE NESO_PARTICLES_DEVICE_TYPE_GPU)
    add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_GPU)
    message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_GPU")
else()
    target_compile_definitions(NESO-Particles INTERFACE NESO_PARTICLES_DEVICE_TYPE_CPU)
    message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_CPU")
endif()


#Get MPI 
find_package(MPI REQUIRED)
target_link_libraries(NESO-Particles INTERFACE MPI::MPI_CXX)

#Get HDF5 if its around
set(NESO_PARTICLES_USING_HDF5 FALSE)
if (NESO_PARTICLES_ENABLE_HDF5) 
    set(HDF5_PREFER_PARALLEL TRUE)
    find_package(HDF5 REQUIRED)
    if(HDF5_FOUND AND HDF5_IS_PARALLEL)
        message(STATUS "Parallel HDF5 found")
        target_link_libraries(NESO-Particles INTERFACE HDF5::HDF5 NESO-Particles)
        target_compile_definitions(NESO-Particles INTERFACE NESO_PARTICLES_HDF5)
        set(NESO_PARTICLES_USING_HDF5 TRUE)
    else()
        message( "HDF5 NOT found")
    endif()
endif()

# Find SYCL
include(${CMAKE_CURRENT_LIST_DIR}/SYCL.cmake)
if(NESO_PARTICLES_ENABLE_FIND_SYCL)
    find_sycl_if_required()
endif()
