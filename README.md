# NESO Particles

This is the particle component of NESO and is designed as a header only library.

## Documentation

[Documentation should be available here.](https://excalibur-neptune.github.io/NESO-Particles/main/sphinx/html/index.html)

## Dependencies

* CMake 
* SYCL 2020 (tested with hipsycl 0.9.4 or Intel DPCPP 2022.1.0).
* MPI 3.0 (tested with MPICH 4.0 or IntelMPI 2021.6)
* HDF5: (optional) If particle trajectories are required - will execute without.

## Installing

NESO-Particles should be placed on the `CMAKE_PREFIX_PATH` such that cmake `find_package` works, e.g.

```
git clone https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles.git
export CMAKE_PREFIX_PATH=<absolute path to NESO-Particles>:$CMAKE_PREFIX_PATH
```

## Testing

Configuring cmake depends on which SYCL implementation/target you wish to use:

```
# Intel DPCPP
cmake -DCMAKE_CXX_COMPILER=dpcpp -DNESO_PARTICLES_DEVICE_TYPE=CPU .
# Hipsycl cpu via omp and host compiler
cmake -DNESO_PARTICLES_DEVICE_TYPE=CPU -DHIPSYCL_TARGETS=omp . 
# Hipsycl cuda using nvcxx
cmake -DNESO_PARTICLES_DEVICE_TYPE=GPU -DHIPSYCL_TARGETS=cuda-nvcxx .
```

Finally executing can be done as follows

```
# Intel DPCPP
SYCL_DEVICE_FILTER=host mpirun -n <nproc> bin/testNESOParticles

# hipsycl omp
OMP_NUM_THREADS=<nthreads> mpirun -n <nproc> bin/testNESOParticles
```

