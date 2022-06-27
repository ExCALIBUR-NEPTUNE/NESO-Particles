#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

TEST(ParticleGroup, creation) {

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

TEST(ParticleGroup, compression_removal_all) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  CartesianHMesh mesh(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order);

  SYCLTarget sycl_target{GPU_SELECTOR, mesh.get_comm()};

  Domain domain(mesh);

  const int cell_count = domain.mesh.get_cell_count();
  ASSERT_TRUE(cell_count > 4);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  ParticleGroup A(domain, particle_spec, sycl_target);

  A.add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3),
                                 domain.mesh.get_cell_count()));

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);
  std::mt19937 rng_cell(5223524);

  const int N = 32;

  auto positions =
      uniform_within_extents(N, ndim, mesh.global_extents, rng_pos);
  auto velocities = normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  std::uniform_int_distribution<int> uniform_dist(0, 8);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("ID")][px][0] = px;

    int cellid = uniform_dist(rng_cell);
    // ensure there are some particles in the cells we test
    if (px == 0) {
      cellid = 0;
    } else if ((px > 0) && (px < 4)) {
      cellid = 1;
    }

    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cellid;
  }

  A.add_particles_local(initial_distribution);

  std::vector<int> orig_occupancies(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    orig_occupancies[cx] = A.get_npart_cell(cx);
    ASSERT_EQ(orig_occupancies[cx], A[Sym<REAL>("P")]->s_npart_cell[cx]);
  }

  std::vector<INT> cells;
  std::vector<INT> layers;
  cells.reserve(N);
  layers.reserve(N);

  // in cell 0 remove all the particles
  const auto npart_cell_0 = A[Sym<INT>("ID")]->s_npart_cell[0];
  for (int layerx = 0; layerx < npart_cell_0; layerx++) {
    cells.push_back(0);
    layers.push_back(layerx);
  }

  A.remove_particles(cells.size(), cells, layers);

  // cell 0 should have no particles
  ASSERT_EQ(A.get_npart_cell(0), 0);
  ASSERT_EQ(A[Sym<REAL>("P")]->s_npart_cell[0], 0);
  ASSERT_EQ(A[Sym<REAL>("V")]->s_npart_cell[0], 0);
  ASSERT_EQ(A[Sym<INT>("CELL_ID")]->s_npart_cell[0], 0);
  ASSERT_EQ(A[Sym<INT>("ID")]->s_npart_cell[0], 0);
  ASSERT_EQ(A[Sym<REAL>("FOO")]->s_npart_cell[0], 0);
  // cells 1, ..., cell_count - 1 should be unchanged
  for (int cx = 1; cx < cell_count; cx++) {
    const int cx_count = orig_occupancies[cx];
    ASSERT_EQ(A.get_npart_cell(cx), cx_count);
    ASSERT_EQ(A[Sym<REAL>("P")]->s_npart_cell[cx], cx_count);
    ASSERT_EQ(A[Sym<REAL>("V")]->s_npart_cell[cx], cx_count);
    ASSERT_EQ(A[Sym<INT>("CELL_ID")]->s_npart_cell[cx], cx_count);
    ASSERT_EQ(A[Sym<INT>("ID")]->s_npart_cell[cx], cx_count);
    ASSERT_EQ(A[Sym<REAL>("FOO")]->s_npart_cell[cx], cx_count);
  }
  for (int cellx = 1; cellx < cell_count; cellx++) {
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

  // remove first and last particle in cell 1
  cells.clear();
  layers.clear();
  cells.reserve(N);
  layers.reserve(N);

  const auto npart_cell_1 = A[Sym<INT>("ID")]->s_npart_cell[1];

  cells.push_back(1);
  layers.push_back(0);
  cells.push_back(1);
  layers.push_back(npart_cell_1 - 1);

  const auto cell1 = A[Sym<INT>("ID")]->cell_dat.get_cell(1);

  const INT rm_id_0 = (*cell1)[0][0];
  const INT rm_id_1 = (*cell1)[0][npart_cell_1 - 1];

  ASSERT_EQ(cells.size(), 2);
  A.remove_particles(cells.size(), cells, layers);

  // cell 1 should have npart_cell_1 - 2 particles
  ASSERT_EQ(A.get_npart_cell(1), npart_cell_1 - 2);
  ASSERT_EQ(A[Sym<REAL>("P")]->s_npart_cell[1], npart_cell_1 - 2);
  ASSERT_EQ(A[Sym<REAL>("V")]->s_npart_cell[1], npart_cell_1 - 2);
  ASSERT_EQ(A[Sym<INT>("CELL_ID")]->s_npart_cell[1], npart_cell_1 - 2);
  ASSERT_EQ(A[Sym<INT>("ID")]->s_npart_cell[1], npart_cell_1 - 2);
  ASSERT_EQ(A[Sym<REAL>("FOO")]->s_npart_cell[1], npart_cell_1 - 2);
  // cells 2, ..., cell_count - 1 should be unchanged
  for (int cx = 2; cx < cell_count; cx++) {
    const int cx_count = orig_occupancies[cx];
    ASSERT_EQ(A.get_npart_cell(cx), cx_count);
    ASSERT_EQ(A[Sym<REAL>("P")]->s_npart_cell[cx], cx_count);
    ASSERT_EQ(A[Sym<REAL>("V")]->s_npart_cell[cx], cx_count);
    ASSERT_EQ(A[Sym<INT>("CELL_ID")]->s_npart_cell[cx], cx_count);
    ASSERT_EQ(A[Sym<INT>("ID")]->s_npart_cell[cx], cx_count);
    ASSERT_EQ(A[Sym<REAL>("FOO")]->s_npart_cell[cx], cx_count);
  }
  for (int cellx = 1; cellx < cell_count; cellx++) {
    auto P = A.get_cell(Sym<REAL>("P"), cellx);
    auto V = A.get_cell(Sym<REAL>("V"), cellx);
    auto FOO = A.get_cell(Sym<REAL>("FOO"), cellx);
    auto ID = A.get_cell(Sym<INT>("ID"), cellx);
    auto CELL_ID = A.get_cell(Sym<INT>("CELL_ID"), cellx);

    for (int rowx = 0; rowx < P->nrow; rowx++) {
      const INT pid = (*ID)[0][rowx];
      int row = -1;
      for (int rx = 0; rx < N; rx++) {
        if (initial_distribution[Sym<INT>("ID")][rx][0] == pid) {
          row = rx;
          break;
        }
      }

      // these ids were rm'd and should not be found
      if ((pid == rm_id_0) || (pid == rm_id_0)) {
        ASSERT_EQ(row, -1);
      } else {
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
  }
  // cell 1 should be missing the first and last elements

  mesh.free();
}

TEST(ParticleGroup, add_particle_dat) {

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

  // dats added after add_particles_* must have space allocated
  A.add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<INT>("BAR"), 1),
                                 domain.mesh.get_cell_count()));

  for (int cellx = 0; cellx < mesh.get_cell_count(); cellx++) {
    ASSERT_EQ(A[Sym<REAL>("P")]->cell_dat.nrow_alloc[cellx],
              A[Sym<INT>("BAR")]->cell_dat.nrow_alloc[cellx]);
  }

  mesh.free();
}
