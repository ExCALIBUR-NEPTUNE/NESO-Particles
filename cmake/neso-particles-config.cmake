message(STATUS "NESO PARTICLES INTERFACE START")
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
include(${CMAKE_CURRENT_LIST_DIR}/restrict-keyword.cmake)

option(NESO_PARTICLES_ENABLE_HDF5 "Add HDF5 to targets" ON)

#Create interface/Header onlu library
add_library(NESO-Particles INTERFACE)
#Alias the name to the namespaces name.
#Can use in subdirectory or via Confiig files with namespace
add_library(NESO-Particles::NESO-Particles ALIAS NESO-Particles)

#Set standard
set_property(TARGET NESO-Particles PROPERTY CXX_STANDARD 17)

#Makes it easy to install + adds the files to the INCLUDE property of the lib
#i.e. don't need target_include_dir.. also no GLOBS
target_sources(NESO-Particles
    PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../include
    FILES
    ${CMAKE_CURRENT_LIST_DIR}/../include/access.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/boundary_conditions.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/cartesian_mesh.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/cell_binning.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/cell_dat.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/cell_dat_compression.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/cell_dat_move.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/cell_dat_move_impl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/communication.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/compute_target.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/blocked_binary_tree.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/cell_dat.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/cell_dat_const.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/cell_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/descendant_products.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/global_array.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/local_array.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/product_matrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/sym_vector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/containers/tuple.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/departing_particle_identification.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/departing_particle_identification_impl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/domain.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/error_propagate.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/global_mapping.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/global_mapping_impl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/global_move.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/global_move_exchange.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/local_mapping.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/local_move.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/loop/access_descriptors.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/loop/particle_loop.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/loop/particle_loop_base.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/loop/particle_loop_index.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/loop/pli_particle_dat.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/mesh_hierarchy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/mesh_interface.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/mesh_interface_local_decomp.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/neso_particles.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/packing_unpacking.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/parallel_initialisation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/particle_dat.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/particle_group.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/particle_group_impl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/particle_io.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/particle_remover.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/particle_set.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/particle_spec.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/particle_sub_group.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/profiling.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/typedefs.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/utility.hpp
    ${CMAKE_CURRENT_LIST_DIR}/../include/utility_mesh_hierarchy_plotting.hpp
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
find_sycl_if_required()

message(STATUS "NESO PARTICLES INTERFACE END")
