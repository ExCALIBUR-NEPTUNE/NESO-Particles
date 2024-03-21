************
Installation
************

Dependencies
============

* CMake 3.24+.
* SYCL 2020: Tested with hipsycl 0.9.4 or Intel DPCPP 2022.1.0) with USM support.
* MPI 3.0: Tested with MPICH 4.0 or IntelMPI 2021.6. See known issues for Ubuntu 22.03 mpich.
* HDF5 (parallel): (optional) If particle trajectories are required - will execute without.

Using with CMake 
================

We provide a CMake interface for projects to access as follows:
::

    find_package(NESO-Particles REQUIRED)
    ...
    target_link_libraries(${EXECUTABLE} PUBLIC NESO-Particles::NESO-Particles)
    ...
    add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE_SOURCE})

Note that we use the existence of the `add_sycl_to_target` function when `find_package(NESO-Particles REQUIRED)` is called to determine if a SYCL implementation has been found already.
If a SYCL implementation is not already configured NESO-Particles will attempt to find one.
This detection is relevant at build time, i.e. when the implementation using NESO-Particles is configured.
These downstream implementations should pass CMake variables or compilers at configure time like:
::
    # Intel DPCPP
    cmake -DCMAKE_CXX_COMPILER=dpcpp .
    # Hipsycl cpu via omp
    cmake -DHIPSYCL_TARGETS=omp . 
    # Hipsycl cuda using nvcxx
    cmake -DHIPSYCL_TARGETS=cuda-nvcxx .

When a SYCL implementation is passed to NESO-Particles itself then it is used to build the tests.

Installing
==========

To build NESO-Particles with tests:
::
    
    # Choose a SYCL implementation to build tests with
    cmake -DHIPSYCL_TARGETS=omp.library-only -DCMAKE_INSTALL_PREFIX=<install location> .
    make
    make install


Known Issues and Workarounds
============================

Ubuntu 22.04 MPICH
------------------






