cmake_minimum_required(VERSION 3.24)

project(
  NESO-Particles
  VERSION 0.0.1
  LANGUAGES CXX C)

include(GNUInstallDirs)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/neso-particles-config.cmake)

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
  FILES ${CMAKE_SOURCE_DIR}/cmake/SYCL.cmake
        ${CMAKE_SOURCE_DIR}/cmake/restrict-keyword.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
export(
  EXPORT ${PROJECT_NAME}_Targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake"
  NAMESPACE ${PROJECT_NAME}::)
