#include <gtest/gtest.h>

// #define NESO_PARTICLES_PROFILING_REGION
#include <neso_particles.hpp>
#include <random>
#include <type_traits>

using namespace NESO::Particles;

namespace {

const int ndim = 2;

ParticleGroupSharedPtr
particle_loop_common(const int N = 10930, const int sx = 4, const int sy = 8) {
  std::vector<int> dims(ndim);
  dims[0] = sx;
  dims[1] = sy;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;

  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 2),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int rank = sycl_target->comm_pair.rank_parent;
  const INT id_offset = rank * N;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px + id_offset;
    initial_distribution[Sym<REAL>("Q")][px][0] = 1.0;
  }

  A->add_particles_local(initial_distribution);
  parallel_advection_initialisation(A, 16);

  auto ccb = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, A->position_dat, A->cell_id_dat);

  ccb->execute();
  A->cell_move();

  return A;
}

} // namespace

#include "example_sources/example_particle_descendant_products.hpp"
#include "example_sources/example_particle_loop_0.hpp"
#include "example_sources/example_particle_loop_0_nc.hpp"
#include "example_sources/example_particle_loop_cell_dat_const.hpp"
#include "example_sources/example_particle_loop_cell_info_npart.hpp"
#include "example_sources/example_particle_loop_global_array.hpp"
#include "example_sources/example_particle_loop_index.hpp"
#include "example_sources/example_particle_loop_local_array.hpp"
#include "example_sources/example_particle_loop_nd_local_array.hpp"
#include "example_sources/example_particle_loop_rng.hpp"
#include "example_sources/example_particle_loop_sym_vector.hpp"
#include "example_sources/example_particle_sub_group_creation.hpp"
#include "example_sources/example_particle_sub_group_loop.hpp"

TEST(Examples, particle_loop_base) {
  auto A = particle_loop_common();

  advection_example(A);
  advection_example_no_comments(A);
  local_array_example(A);
  nd_local_array_example(A);
  global_array_example(A);
  cell_dat_const_example(A);
  particle_sub_group_creation(A);
  particle_sub_group_loop(A);
  sym_vector_example(A);
  advection_example_loop_index(A);
  particle_loop_rng(A);
  particle_loop_example_cell_info_npart(A);

  auto B = particle_loop_common(5);
  descendant_products_example(B);

  A->free();
  A->sycl_target->free();
  A->domain->mesh->free();
  B->free();
  B->sycl_target->free();
  B->domain->mesh->free();
}

#include "example_sources/example_profile_regions.hpp"

TEST(Examples, profile_region) {
  auto A = particle_loop_common(1000, 8, 8);
  profile_regions_example(A);
  A->free();
  A->sycl_target->free();
  A->domain->mesh->free();
}
