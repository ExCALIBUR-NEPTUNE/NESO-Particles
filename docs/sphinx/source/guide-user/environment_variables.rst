*********************
Environment Variables
*********************

The behaviour of NESO-Particles can be configured at runtime by various environment variables. Note that each SYCL implementation typically has a set of environment variables which are complementary to those we list here.

.. list-table:: Environment variables for runtime configuration.
   :header-rows: 1

   * - Variable Name
     - Description
   * - ``NESO_PARTICLES_LOOP_LOCAL_SIZE``
     - Set a maximum local size for parallel loop execution. Local memory requirements may force the actual local size to be smaller than this specified value.
   * - ``NESO_PARTICLES_LOOP_NBIN``
     - NESO-Particles attempts to create parallel iteration sets which are as large as possible whilst being reasonably efficient in the case where there are large differences in the number of particles in each cell. Internally we create a large iteration set that covers all cells up to the smallest occupancy over all cells.Then we create nbin peel loops that iterate over the remaining cells. The default value for this variable is 4, i.e. 4+1 parallel loops are launched. Simulations with significant differences in cell occupancy counts may see a performance improvement by increasing this number. Simulations with very uniform cell occupancy counts may see a performance improvement by decreasing this number. 
   * - ``NESO_PARTICLES_IN_ORDER_QUEUE``
     - When enabled (by setting to a non-zero value) attempts to create a SYCL queue which is in-order.
   * - ``NESO_PARTICLES_VERBOSE_DEVICE``
     - When enabled (by setting to a non-zero value) the information held for the SYCLTarget will be printed on construction.
   * - ``NESO_PARTICLES_DEVICE_LIMIT_WORK_GROUP_SIZE``
     - When enabled (by setting to a non-zero value) the device limits for parallel workgroup sizes will be overridden. Typically this is only useful if you suspect that the SYCL implementation has miss-reported the valid workgroup size.
   * - ``NESO_PARTICLES_DEBUG_SUB_GROUPS``
     - When enabled (by setting to a non-zero value) the runtime will print information relating to when and why ``ParticleSubGroups`` have been reconstructed.
   * - ``NESO_PARTICLES_DEVICE_AWARE_MPI``
     - When enabled (by setting to a non-zero value) the runtime will assume that the MPI implementation natively operates with device pointers. Enabling device aware MPI can be done at CMake time in addition to this environment variable. Note that this environment variable is default off but can almost certainly be enabled in cases where the compute device is a CPU backend. For GPU backend users should build NESO-Particles with a device aware MPI implementation then enable this option at CMake time or runtime.
   * - ``NESO_PARTICLES_DEBUG_LEVEL``
     - When enabled (by setting to a non-zero value) the runtime may print more debugging information at runtime.
   * - ``NESO_PARTICLES_TEST_RESOURCES_DIR``
     - Allows the tests to be ran from a different directory. By default NESO-Particles is built and installed with the test binaries. These binaries are installed in a different location to the test resources and hence cannot find the test resources. This variable should be set to an absolute path to a copy of `test/test_resources`.
   * - ``NESO_PARTICLES_TEST_TMP_DIR``
     - Configures which directory should be used as a temporary directory for running tests. This directory should be visible from all MPI ranks and hence on a distributed memory system will probably need to be set to run the tests in a multinode configuration. The default setting is the value returned from `std::filesystem::current_path()`.
   * - ``NESO_PARTICLES_DMPLEX_CHECK_FACES``
     - Default 1. When set to a non-zero value enables self checking of the vertex ordering in DMPlex meshes passed to the DMPlexInterface and the generated halo DMPlex.
   * - ``NESO_PARTICLES_AUTO_PROFILE``
     - Default empty string. When set to a file name prefix, e.g, foo, profiling will be enabled and events written to foo.<rank>json on call to free of the SYCLTarget.
   * - ``NESO_PARTICLES_MAX_COMPUTE_UNITS``
     - Default empty string. When set to a positive integer this will override how many compute units NESO-Particles considers each rank to have.
