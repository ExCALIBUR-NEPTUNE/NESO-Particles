#include "include/test_neso_particles.hpp"

TEST(SubGroupSelector, process_npart) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_SELF);

  const int cell_count = 107;
  auto dh_npart_cell = std::make_shared<BufferDeviceHost<int>>(
      sycl_target, cell_count + cell_count * NESO_PARTICLES_CACHELINE_NUM_int);

  for (int ix = 0; ix < cell_count; ix++) {
    dh_npart_cell->h_buffer.ptr[ix] = ix + 1;
  }

  dh_npart_cell->host_to_device();

  ParticleSubGroupImplementation::SubGroupSelector::pre_process_npart_cell(
      sycl_target, cell_count, dh_npart_cell->d_buffer.ptr);

  dh_npart_cell->device_to_host();

  for (int ix = 0; ix < cell_count; ix++) {
    dh_npart_cell->h_buffer.ptr[ix] = -1;
    ASSERT_EQ(dh_npart_cell->h_buffer
                  .ptr[cell_count + ix * NESO_PARTICLES_CACHELINE_NUM_int],
              ix + 1);
    dh_npart_cell->h_buffer
        .ptr[cell_count + ix * NESO_PARTICLES_CACHELINE_NUM_int] *= 2;
  }

  dh_npart_cell->host_to_device();
  ParticleSubGroupImplementation::SubGroupSelector::post_process_npart_cell(
      sycl_target, cell_count, dh_npart_cell->d_buffer.ptr);

  dh_npart_cell->device_to_host();

  for (int ix = 0; ix < cell_count; ix++) {
    ASSERT_EQ(dh_npart_cell->h_buffer.ptr[ix], 2 * (ix + 1));
  }

  sycl_target->free();
}
