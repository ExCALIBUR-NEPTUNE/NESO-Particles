#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, get_particles) {

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
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  const int cell_count = domain->mesh->get_cell_count();
  auto A = make_test_obj<ParticleGroup>(domain, particle_spec, sycl_target);

  A->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3), cell_count));

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);

  const int N = cell_count * 2 + 7;

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
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % (cell_count - 1);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }

  A->add_particles_local(initial_distribution);
  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int npart_cell = aa->get_npart_cell(cellx);
    std::vector<INT> cells;
    std::vector<INT> layers;
    cells.reserve(npart_cell);
    layers.reserve(npart_cell);
    for (int layerx = 0; layerx < npart_cell; layerx++) {
      cells.push_back(cellx);
      layers.push_back(layerx);
    }
    auto particles = aa->get_particles(cells, layers);
    ASSERT_EQ(particles->npart, npart_cell);

    for (int layerx = 0; layerx < npart_cell; layerx++) {
      const INT px = particles->at(Sym<INT>("ID"), layerx, 0);
      EXPECT_EQ(px % (cell_count - 1), cellx);
      EXPECT_TRUE(px % 2 == 0);
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("P"), px, 0),
                particles->at(Sym<REAL>("P"), layerx, 0));
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("P"), px, 1),
                particles->at(Sym<REAL>("P"), layerx, 1));
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("V"), px, 0),
                particles->at(Sym<REAL>("V"), layerx, 0));
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("V"), px, 1),
                particles->at(Sym<REAL>("V"), layerx, 1));
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("V"), px, 2),
                particles->at(Sym<REAL>("V"), layerx, 2));

      EXPECT_EQ(particles->at(Sym<INT>("CELL_ID"), layerx, 0), cellx);
      EXPECT_EQ(particles->at(Sym<REAL>("FOO"), layerx, 0), 0.0);
      EXPECT_EQ(particles->at(Sym<REAL>("FOO"), layerx, 1), 0.0);
      EXPECT_EQ(particles->at(Sym<REAL>("FOO"), layerx, 2), 0.0);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
