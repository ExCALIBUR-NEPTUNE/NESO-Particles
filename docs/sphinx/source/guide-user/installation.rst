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
    
    git clone https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles.git
    cd NESO-Particles
    mkdir build
    cd  build
    # Choose a SYCL implementation to build tests with
    cmake -DHIPSYCL_TARGETS=omp.library-only -DCMAKE_INSTALL_PREFIX=<install location> .
    make
    make install

The tests can then be run with:
::

    # Intel DPCPP
    SYCL_DEVICE_FILTER=host mpirun -n <nproc> bin/testNESOParticles

    # hipsycl omp
    OMP_NUM_THREADS=<nthreads> mpirun -n <nproc> bin/testNESOParticles

Note that typically the OpenMP default is to run a number of threads equal to the number of logical CPU cores (this is implementation defined). 
Hence with OpenMP SYCL implementations the number of threads should, almost always, be set when using MPI.
If the number of threads is not set then each MPI rank will launch a thread per CPU core resulting in over-subscription of cores and slowdown.
Intel SYCL implementations currently choose the number of workers (threads) based on the process affinity of each MPI rank which can be configured through environment variables and runtime options passed to the MPI launcher.


Known Issues and Workarounds
============================

Ubuntu 22.04 MPICH
------------------

The build of MPICH found in the Ubuntu 22.04 APT repositories includes additional compiler flags, e.g. ``-flto=auto``, relating to link time optimisation which break compilation workflows that involve compilers different to the system GCC.
The symptoms of this issue are errors at compile time due to the unknown flags, e.g.
::

    nvc++-Error-Unknown switch: -flto=auto
    nvc++-Error-Unknown switch: -ffat-lto-objects

Or issues at link time due to the intermediate objects having types incompatible with the linker, e.g.
::
    
    CMakeFiles/test_buffers.dir/main.cpp.o: file not recognized: file format not recognized

Two possible solutions are as follows

1. Build a separate installation of MPICH (ideally with the same compiler as used by the SYCL implementation).
2. Clear the offending variables from the CMake cache by first running CMake as normal then rerunning CMake as ``cmake -DMPI_CXX_COMPILE_OPTIONS="" -DMPI_CXX_COMPILE_OPTIONS="" ..``.

