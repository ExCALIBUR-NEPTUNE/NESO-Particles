# Get the absolute path of NESO-Particles root
get_filename_component(NESO_PARTICLES_ROOT "${CMAKE_CURRENT_LIST_DIR}/../"
                       ABSOLUTE)

# set the variable for projects to use to find header files
set(NESO_PARTICLES_INCLUDE_PATH ${NESO_PARTICLES_ROOT}/include)

# set link libraries and flags
set(NESO_PARTICLES_LIBRARIES "")
set(NESO_PARTICLES_LINK_FLAGS "")

# set the neso particles device type
if(NESO_PARTICLES_DEVICE_TYPE STREQUAL GPU)
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_GPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_GPU")
else()
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_CPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_CPU")
endif()

set(HDF5_PREFER_PARALLEL TRUE)
enable_language(C)
find_package(HDF5 QUIET)

if(HDF5_FOUND AND HDF5_IS_PARALLEL)
    message(STATUS "HDF5 found")
    add_definitions(-DNESO_PARTICLES_HDF5)
    add_definitions(${HDF5_DEFINITIONS})
    set(NESO_PARTICLES_LIBRARIES ${NESO_PARTICLES_LIBRARIES} ${HDF5_LIBRARIES})
    set(NESO_PARTICLES_INCLUDE_PATH ${NESO_PARTICLES_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS})
else()
    message(STATUS "HDF5 NOT found")
endif()
message(STATUS "HDF5_IS_PARALLEL " ${HDF5_IS_PARALLEL})

