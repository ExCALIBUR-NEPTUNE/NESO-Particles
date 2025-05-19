include(CheckCXXSourceCompiles)

# Does oneDPL exist and enabled?
option(NESO_PARTICLES_ENABLE_ONEDPL "Enable using oneDPL if found." ON)
option(NESO_PARTICLES_REQUIRE_ONEDPL "Force using oneDPL." OFF)

if(NESO_PARTICLES_REQUIRE_ONEDPL)
  find_package(oneDPL REQUIRED)
elseif(NESO_PARTICLES_ENABLE_ONEDPL)
  find_package(oneDPL QUIET)
endif()

if(oneDPL_FOUND)
  message(STATUS "oneDPL Found")

  set(ORIG_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
  set(CMAKE_REQUIRED_LIBRARIES oneDPL)
  check_cxx_source_compiles(
    "
      #include <sycl/sycl.hpp>
      #include <oneapi/dpl/random>
      int main(int argc, char ** argv) {return 0;}
      "
    COMPILES_ONEDPL)
  set(CMAKE_REQUIRED_LIBRARIES ${ORIG_CMAKE_REQUIRED_LIBRARIES})

  if(COMPILES_ONEDPL)
    target_compile_definitions(NESO-Particles PUBLIC NESO_PARTICLES_ONEDPL)
    target_link_libraries(NESO-Particles PUBLIC oneDPL)
  endif()
else()
  message(STATUS "oneDPL NOT Found")
endif()

# Does the CUDA toolkit exist and is enabled?
option(NESO_PARTICLES_ENABLE_CUDA_TOOLKIT
       "Enable using the CUDA toolkit, e.g. for curand" ON)
option(NESO_PARTICLES_REQUIRE_CUDA_TOOLKIT
       "Force using the CUDA toolkit, e.g. for curand" OFF)

if(NESO_PARTICLES_REQUIRE_CUDA_TOOLKIT)
  find_package(CUDAToolkit REQUIRED)
elseif(NESO_PARTICLES_ENABLE_CUDA_TOOLKIT)
  find_package(CUDAToolkit QUIET)
endif()

if(CUDAToolkit_FOUND)
  message(STATUS "CUDA Tookit Found")
  if(TARGET CUDA::curand)
    message(STATUS "CUDA::curand Found")

    set(ORIG_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    set(CMAKE_REQUIRED_LIBRARIES CUDA::curand)
    check_cxx_source_compiles(
      "
        #include <curand.h>
        int main(int argc, char ** argv) {return 0;}
        "
      COMPILES_CURAND)
    set(CMAKE_REQUIRED_LIBRARIES ${ORIG_CMAKE_REQUIRED_LIBRARIES})

    if(COMPILES_CURAND)
      target_compile_definitions(NESO-Particles PUBLIC NESO_PARTICLES_CURAND)
      target_link_libraries(NESO-Particles PUBLIC CUDA::curand)
    endif()
  else()
    message(STATUS "CUDA::curand NOT Found")
  endif()
else()
  message(STATUS "CUDA Tookit NOT Found")
endif()
