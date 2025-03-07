#include "include/test_neso_particles.hpp"

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common(const int N = 4093) {
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

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

TEST(ParticleLoop, cell_info_npart) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();
  std::vector<INT> correct(cell_count);
  auto la_to_test = std::make_shared<LocalArray<INT>>(sycl_target, cell_count);

  {
    for (int cx = 0; cx < cell_count; cx++) {
      correct.at(cx) = static_cast<INT>(A->get_npart_cell(cx));
    }
    la_to_test->fill(0);
    particle_loop(
        A,
        [=](auto INDEX, auto CELL_INFO_NPART, auto LA) {
          // Is this the first particle in the cell?
          if (INDEX.loop_layer == 0) {
            LA.at(INDEX.cell) = CELL_INFO_NPART.get();
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(CellInfoNPart{}),
        Access::write(la_to_test))
        ->execute();

    auto h_to_test = la_to_test->get();
    for (int cx = 0; cx < cell_count; cx++) {
      EXPECT_EQ(h_to_test.at(cx), correct.at(cx));
    }
  }

  {
    auto aa = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));
    for (int cx = 0; cx < cell_count; cx++) {
      correct.at(cx) = static_cast<INT>(aa->get_npart_cell(cx));
    }
    la_to_test->fill(0);
    particle_loop(
        aa,
        [=](auto INDEX, auto CELL_INFO_NPART, auto LA) {
          // Is this the first particle in the cell?
          if (INDEX.loop_layer == 0) {
            LA.at(INDEX.cell) = CELL_INFO_NPART.get();
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(CellInfoNPart{}),
        Access::write(la_to_test))
        ->execute();

    auto h_to_test = la_to_test->get();
    for (int cx = 0; cx < cell_count; cx++) {
      EXPECT_EQ(h_to_test.at(cx), correct.at(cx));
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
