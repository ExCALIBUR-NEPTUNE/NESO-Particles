#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

TEST(ParticleGroup, cell_move) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 8;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 0;
  CartesianHMesh mesh(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order);

  SYCLTarget sycl_target{GPU_SELECTOR, mesh.get_comm()};

  Domain domain(mesh);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  ParticleGroup A(domain, particle_spec, sycl_target);

  A.add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3),
                                 domain.mesh.get_cell_count()));

  const int size = sycl_target.comm_pair.size_parent;
  const int rank = sycl_target.comm_pair.rank_parent;

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_cell(18241 + rank);
  std::mt19937 rng_rank(112348241);

  const int N = 1024;
  const int Ntest = 20;

  auto positions =
      uniform_within_extents(N, ndim, mesh.global_extents, rng_pos);

  const int cell_count = domain.mesh.get_cell_count();
  std::uniform_int_distribution<int> dist_cell(0, cell_count - 1);
  std::uniform_int_distribution<int> dist_rank(0, size - 1);

  ParticleSet initial_distribution(N, A.get_particle_spec());

  // determine which particles should end up on which rank
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = (px * 3) + dimx;
    }

    const auto px_cell = dist_cell(rng_cell);
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = px_cell;
    const auto px_cell_new = dist_cell(rng_cell);

    initial_distribution[Sym<INT>("ID")][px][0] = px;

    const auto px_rank = dist_rank(rng_rank);
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
  }

  A.add_particles_local(initial_distribution);

  BufferShared<INT> new_cell_ids(sycl_target, N);

  for (int testx = 0; testx < Ntest; testx++) {
    // set the new cells
    for (int px = 0; px < N; px++) {
      new_cell_ids.ptr[px] = dist_cell(rng_cell);
    }
    auto pl_iter_range = A.position_dat->get_particle_loop_iter_range();
    auto pl_stride = A.position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = A.position_dat->get_particle_loop_npart_cell();

    auto k_id_dat = A[Sym<INT>("ID")]->cell_dat.device_ptr();
    auto k_cell_id_dat = A[Sym<INT>("CELL_ID")]->cell_dat.device_ptr();
    auto k_new_cell_ids = new_cell_ids.ptr;

    sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                const INT px = k_id_dat[cellx][0][layerx];
                k_cell_id_dat[cellx][0][layerx] = k_new_cell_ids[px];
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    A.cell_move();

    int npart_found = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto cell_id = A[Sym<INT>("CELL_ID")]->cell_dat.get_cell(cellx);
      auto id = A[Sym<INT>("ID")]->cell_dat.get_cell(cellx);
      auto v = A[Sym<REAL>("V")]->cell_dat.get_cell(cellx);

      for (int rowx = 0; rowx < v->nrow; rowx++) {

        ASSERT_EQ((*cell_id)[0][rowx], cellx);

        const INT px = (*id)[0][rowx];
        for (int dimx = 0; dimx < 3; dimx++) {
          ASSERT_TRUE(ABS((px * 3 + dimx) - (*v)[dimx][rowx]) < 1.0e-10);
        }

        npart_found++;
      }
    }
    ASSERT_EQ(npart_found, N);
  }

  mesh.free();
}
