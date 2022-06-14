# NESO Particles

This is the particle component of NESO and is designed as a header only library.

## Dependencies

* CMake 
* SYCL 2020 (e.g. hipsycl 0.9.2 or Intel DPCPP 2022.1.0).
* MPI 3.0 (e.g. MPICH or IntelMPI)

## Testing

Build and run the tests with:

### Hipsycl
```
# choose target, e.g.
# 1)
cmake -DHIPSYCL_TARGETS=omp .
# or 2)
cmake -DHIPSYCL_TARGETS=cuda-nvcxx .
# build tests
make -j 12
# run tests
bin/testNESOParticles
```

### Intel DPCPP
```
# configure
cmake -DCMAKE_CXX_COMPILER=dpcpp .
# build tests
make -j 12
# run tests
bin/testNESOParticles
```


