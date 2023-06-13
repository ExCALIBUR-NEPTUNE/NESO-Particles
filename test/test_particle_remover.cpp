#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

// test that int atomics are functional
TEST(ParticleRemover, atomics) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  const int N = 1024;
  BufferDeviceHost<int> dh_a(sycl_target, 1);
  dh_a.h_buffer.ptr[0] = 0;
  dh_a.host_to_device();
  auto k_ptr = dh_a.d_buffer.ptr;

  sycl_target->queue
      .submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::item<1> id) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              remove_count_atomic{k_ptr[0]};
          remove_count_atomic.fetch_add(1);
        });
      })
      .wait_and_throw();

  dh_a.device_to_host();

  ASSERT_EQ(dh_a.h_buffer.ptr[0], 1024);

  mesh->free();
}

TEST(ParticleRemover, remove) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto domain = std::make_shared<Domain>(mesh);
  const int cell_count = domain->mesh->get_cell_count();

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<INT>("REMOVE_FLAG"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  const int rank = sycl_target->comm_pair.rank_parent;

  const int N = 2048;
  const int last_keep_index = N / 2 - 1;

  // Add some particles and set flag such that roughly half should be removed
  if (rank == 0) {
    std::mt19937 rng_pos(52234234);
    auto positions =
        uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);

    ParticleSet initial_distribution(N, particle_spec);

    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;

      const int remove_flag = (px > last_keep_index) ? 1 : 0;
      initial_distribution[Sym<INT>("REMOVE_FLAG")][px][0] = remove_flag;
    }
    A->add_particles_local(initial_distribution);
  }

  A->hybrid_move();

  // create a remover and remove particles
  auto particle_remover = std::make_shared<ParticleRemover>(sycl_target);
  particle_remover->remove(A, (*A)[Sym<INT>("REMOVE_FLAG")], 1);

  // check particles were removed
  int npart_found = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto id = (*A)[Sym<INT>("ID")]->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < id->nrow; rowx++) {
      const INT px = (*id)[0][rowx];
      ASSERT_TRUE(px <= last_keep_index);
      npart_found++;
    }
  }
  // check number of remaining particles
  int npart_total;
  MPICHK(MPI_Allreduce(&npart_found, &npart_total, 1, MPI_INT, MPI_SUM,
                       sycl_target->comm_pair.comm_parent));

  ASSERT_EQ(npart_total, (last_keep_index + 1));

  mesh->free();
  A->free();
}
