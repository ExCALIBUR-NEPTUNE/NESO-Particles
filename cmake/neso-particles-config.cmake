# Get the absolute path of NESO-Particles root
get_filename_component(NESO_PARTICLES_ROOT "${CMAKE_CURRENT_LIST_DIR}/../"
                       ABSOLUTE)

# set the variable for projects to use to find header files
set(NESO_PARTICLES_INCLUDE_PATH ${NESO_PARTICLES_ROOT}/include)

# set link libraries and flags
set(NESO_PARTICLES_LIBRARIES "")
set(NESO_PARTICLES_LINK_FLAGS "")

# set the neso particles device type
if(NESO_PARTICLES_DEVICE_TYPE STREQUAL CPU)
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_CPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_CPU")
else()
  add_definitions(-DNESO_PARTICLES_DEVICE_TYPE_GPU)
  message(STATUS "Using NESO_PARTICLES_DEVICE_TYPE_GPU")
endif()
