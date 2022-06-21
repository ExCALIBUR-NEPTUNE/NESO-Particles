#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

TEST(ParticleGroup, test_particle_group_creation) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
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

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);

  const int N = 10;

  auto positions =
      uniform_within_extents(N, ndim, mesh.global_extents, rng_pos);
  auto velocities = normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }

  A.add_particles_local(initial_distribution);

  for (int cellx = 0; cellx < mesh.get_cell_count(); cellx++) {
    auto P = A.get_cell(Sym<REAL>("P"), cellx);
    auto V = A.get_cell(Sym<REAL>("V"), cellx);
    auto FOO = A.get_cell(Sym<REAL>("FOO"), cellx);
    auto ID = A.get_cell(Sym<INT>("ID"), cellx);
    auto CELL_ID = A.get_cell(Sym<INT>("CELL_ID"), cellx);

    for (int rowx = 0; rowx < P->nrow; rowx++) {
      int row = -1;
      for (int rx = 0; rx < N; rx++) {
        if (initial_distribution[Sym<INT>("ID")][rx][0] == (*ID)[0][rowx]) {
          row = rx;
          break;
        }
      }
      ASSERT_TRUE(row >= 0);

      for (int cx = 0; cx < ndim; cx++) {
        ASSERT_EQ((*P)[cx][rowx],
                  initial_distribution[Sym<REAL>("P")][row][cx]);
      }
      for (int cx = 0; cx < 3; cx++) {
        ASSERT_EQ((*V)[cx][rowx],
                  initial_distribution[Sym<REAL>("V")][row][cx]);
      }
      for (int cx = 0; cx < 1; cx++) {
        ASSERT_EQ((*CELL_ID)[cx][rowx],
                  initial_distribution[Sym<INT>("CELL_ID")][row][cx]);
      }
      for (int cx = 0; cx < 3; cx++) {
        ASSERT_EQ((*FOO)[cx][rowx], 0.0);
      }
    }
  }

  mesh.free();
}
