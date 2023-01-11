#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>
#include <memory>

using namespace NESO::Particles;

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

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<INT>("REMOVE_FLAG"), 1)
  };

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3),
                                 domain->mesh->get_cell_count()));
  
  const int rank = sycl_target->comm_pair.rank_parent;

  const int N = 10;
  const int last_keep_index = N / 2 - 1;

  if (rank == 0){
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
    A->add_particles(initial_distribution);
  } else {
    A->add_particles();
  }

  mesh->free();
  A->free();
}

