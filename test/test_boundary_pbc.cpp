#include <CL/sycl.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

TEST(BoundaryConditions, pbc_apply) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 8;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 0;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto domain = std::make_shared<Domain>(mesh);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int rank = sycl_target->comm_pair.rank_parent;
  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241 + rank);
  std::mt19937 rng_cell(3258241 + rank);

  const int N = 1024;

  std::uniform_real_distribution<double> uniform_rng(
      -100.0 * (dims[0] * cell_extent), 100.0 * (dims[0] * cell_extent));

  const int cell_count = mesh->get_cell_count();
  std::uniform_int_distribution<int> uniform_dist(
      0, sycl_target->comm_pair.size_parent - 1);
  std::uniform_int_distribution<int> cell_dist(0, cell_count - 1);

  ParticleSet initial_distribution(N, A->get_particle_spec());

  // determine which particles should end up on which rank
  std::map<int, std::vector<int>> mapping;
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const REAL pos = uniform_rng(rng_pos);
      initial_distribution[Sym<REAL>("P")][px][dimx] = pos;
      initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = pos;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cell_dist(rng_cell);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
    const auto px_rank = uniform_dist(rng_rank);
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
    mapping[px_rank].push_back(px);
  }
  A->add_particles_local(initial_distribution);

  CartesianPeriodic pbc(sycl_target, mesh, A->position_dat);

  pbc.execute();

  // ParticleDat P should contain the perodically mapped positions in P_ORIG

  // for each local cell
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto pos = (*A)[Sym<REAL>("P")]->cell_dat.get_cell(cellx);
    auto pos_orig = (*A)[Sym<REAL>("P_ORIG")]->cell_dat.get_cell(cellx);

    ASSERT_EQ(pos->nrow, pos_orig->nrow);
    const int nrow = pos->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {

      // for each dimension
      for (int dimx = 0; dimx < ndim; dimx++) {
        const REAL correct_pos = std::fmod(
            (*pos_orig)[dimx][rowx] + (1000.0 * cell_extent * dims[dimx]),
            (REAL)(cell_extent * dims[dimx]));
        const REAL to_test_pos = (*pos)[dimx][rowx];
        ASSERT_TRUE(ABS(to_test_pos - correct_pos) < 1.0e-10);
        ASSERT_TRUE(correct_pos >= 0.0);
        ASSERT_TRUE(correct_pos <= (cell_extent * dims[dimx]));
      }
    }
  }

  mesh->free();
}
