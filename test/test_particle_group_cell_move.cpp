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
                             ParticleProp(Sym<INT>("CELL_ID_NEW"), 1),
                             ParticleProp(Sym<INT>("ID"), 1)};

  ParticleGroup A(domain, particle_spec, sycl_target);

  A.add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3),
                                 domain.mesh.get_cell_count()));

  const int size = sycl_target.comm_pair.size_parent;
  const int rank = sycl_target.comm_pair.rank_parent;

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);
  std::mt19937 rng_cell(18241);
  std::mt19937 rng_rank(112348241);

  const int N = 10;

  auto positions =
      uniform_within_extents(N, ndim, mesh.global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

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
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }

    const auto px_cell = dist_cell(rng_cell);
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = px_cell;
    const auto px_cell_new = dist_cell(rng_cell);
    initial_distribution[Sym<INT>("CELL_ID_NEW")][px][0] = px_cell_new;

    initial_distribution[Sym<INT>("ID")][px][0] = px;

    const auto px_rank = dist_rank(rng_rank);
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
  }

  A.add_particles_local(initial_distribution);

  // set the new cells

  auto pl_iter_range = A.position_dat->get_particle_loop_iter_range();
  auto pl_stride = A.position_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell = A.position_dat->get_particle_loop_npart_cell();

  auto k_cell_id_dat = A[Sym<INT>("CELL_ID")]->cell_dat.device_ptr();
  auto k_cell_id_new_dat = A[Sym<INT>("CELL_ID_NEW")]->cell_dat.device_ptr();

  sycl_target.queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          const INT cellx = ((INT)idx) / pl_stride;
          const INT layerx = ((INT)idx) % pl_stride;
          if (layerx < pl_npart_cell[cellx]) {
            k_cell_id_dat[cellx][0][layerx] =
                k_cell_id_new_dat[cellx][0][layerx];
          }
        });
      })
      .wait_and_throw();

  A.cell_move();

  mesh.free();
}
