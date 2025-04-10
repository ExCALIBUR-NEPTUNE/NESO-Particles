************
Installation
************

Dependencies
============

* CMake 3.24+.
* SYCL 2020 with USM support: e.g. AdaptiveCpp 24.10.0 or Intel DPCPP 2024.2.1. Please read the section on known issues and workarounds before choosing a SYCL implementation.
* MPI 3.0: e.g. MPICH 4.0 or IntelMPI 2021.6. See known issues for Ubuntu 22.03 mpich.
* HDF5 (parallel): (optional) See CMake variable ``NESO_PARTICLES_ENABLE_HDF5`` if particle trajectories are required - will execute without.
* PETSc: (optional) See CMake variable ``NESO_PARTICLES_ENABLE_PETSC`` if PETSc DMPlex support is required.

Using with CMake 
================

We provide a CMake interface for projects to access NESO-Particles as a package or as a submodule.
To use NESO-Particles in package form projects should involve CMake implementation like:
::

    find_package(NESO-Particles REQUIRED)
    ...
    target_link_libraries(${EXECUTABLE} PUBLIC NESO-Particles::NESO-Particles)
    ...
    add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE_SOURCE})

 
To use NESO-Particles from a submodule within your project replace the ``find_package`` call as follows:
::
    
    # Enable/disable options for NESO-Particles, e.g. HDF5
    #option(NESO_PARTICLES_ENABLE_HDF5 OFF)
    add_subdirectory(neso-particles)

where ``neso-particles`` is the directory within your project that contains NESO-Particles.

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
Please read the section :ref:`device-aware-mpi` for more information relating to device aware MPI.

CXX Standard
------------

Although we explicitly set C++17 as the required C++ version on the NESO-Particles interface target, CMake may not pass this requirement down onto targets using NESO-Particles.
Downstream projects may need to explicitly set the required C++ standard to C++17.

Installing
==========

To build NESO-Particles with tests:
::
    
    git clone https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles.git
    cd NESO-Particles
    mkdir build
    cd  build
    # Choose a SYCL implementation to build tests with
    cmake -DHIPSYCL_TARGETS=omp.library-only -DCMAKE_INSTALL_PREFIX=<install location> ..
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

Notes:

#. Searching for and building against HDF5 can be disabled by passing ``-DNESO_PARTICLES_ENABLE_HDF5=OFF`` to CMake.
#. If an installation is required without tests built or SYCL configured then CMake can be called with ``-DNESO_PARTICLES_ENABLE_TESTS=OFF -DNESO_PARTICLES_ENABLE_FIND_SYCL=OFF``. **Tests should always be ran before any notion of trust is formed of outputs**.


.. _device-aware-mpi:

Device Aware MPI
================

By default device aware MPI, i.e. passing device pointers to MPI, is disabled for compatibility. 
Device aware MPI functionality can be enabled by default by passing ``-DNESO_PARTICLES_DEVICE_AWARE_MPI=ON`` to cmake.
At runtime device aware MPI can be enabled by setting the environment variable ``NESO_PARTICLES_DEVICE_AWARE_MPI`` to ``ON`` alternatively at runtime this functionality can be disabled by setting the same environment variable to ``OFF``.
The environment variable takes precedence over the cmake variable and in particular it is not required to enable device aware MPI at cmake time to later enable this functionality by using the environment variable.
Note that when compiling for a CPU device it should always be safe to enable device aware MPI. 

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

#. Build a separate installation of MPICH (ideally with the same compiler as used by the SYCL implementation).
#. Clear the offending variables from the CMake cache by first running CMake as normal then rerunning CMake as ``cmake -DMPI_CXX_COMPILE_OPTIONS="" -DMPI_CXX_COMPILE_OPTIONS="" ..``.

AdaptiveCpp
-----------

As the AdaptiveCpp ``generic`` implementation uses just-in-time (JIT) compilation, initial runs and time steps may be slower than desired as the parallel loops are compiled.
Subsequent execution of the same parallel loops should be much faster.

When running the tests setting ``ACPP_PERSISTENT_RUNTIME=1`` will prevent issues and errors relating to cached JIT compiled objects.
For MPI execution setting ``ACPP_APPDB_DIR`` to a directory on the parallel storage or node local and ``ACPP_RT_NO_JIT_CACHE_POPULATION=1`` may also be beneficial.
Please visit the AdaptiveCpp documentation for more information and latest guidance.
The tests may take a long time to run with the default AdaptiveCpp adaptivity settings.
Setting ``ACPP_ADAPTIVITY_LEVEL=0`` will speed up the test execution (note that the AdaptiveCpp authors do not recommend changing this value).

Intel SYCL
----------

With the Intel SYCL implementation, currently branded as oneAPI, users may observe illegal instructions. Typically these are a ``vgatherdpd`` like instruction.
Currently the known workaround is to set the environment variable ``CL_CONFIG_CPU_TARGET_ARCH=corei7-avx``.
