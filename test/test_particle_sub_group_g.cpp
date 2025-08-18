#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, single_cell_base) {

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

  const int N = cell_count * 3 + 7;

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

  const int cm1 = cell_count - 1;
  auto aam1 = particle_sub_group(A, cm1);

  ASSERT_TRUE(!aam1->is_entire_particle_group());
  ASSERT_EQ(aam1->get_particle_group(), A);
  for (int cx = 0; cx < cell_count; cx++) {
    if (cx != cm1) {
      ASSERT_EQ(aam1->get_npart_cell(cx), 0);
    } else {
      ASSERT_EQ(aam1->get_npart_cell(cx), A->get_npart_cell(cx));
    }
  }
  ASSERT_EQ(aam1->get_npart_local(), A->get_npart_cell(cm1));
  ASSERT_TRUE(aam1->is_valid());
  ASSERT_TRUE(!aam1->static_status());

  // sub group from sub group selecting different cells.
  if (0 != cm1) {
    auto aam10 = particle_sub_group(aam1, 0);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(aam10->get_npart_cell(cx), 0);
    }
    ASSERT_EQ(aam10->get_npart_local(), 0);
  }
  // sub group from sub group selecting same cell.
  auto aam1m1 = particle_sub_group(aam1, cm1);
  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(aam1m1->get_npart_cell(cx), aam1->get_npart_cell(cx));
  }
  ASSERT_EQ(aam1m1->get_npart_local(), A->get_npart_cell(cm1));

  auto ep = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = ep->device_ptr();

  particle_loop(
      aam1, [=](auto INDEX) { NESO_KERNEL_ASSERT(INDEX.cell == cm1, k_ep); },
      Access::read(ParticleLoopIndex{}))
      ->execute();
  ASSERT_TRUE(!ep->get_flag());

  particle_loop(
      aam1m1, [=](auto /*INDEX*/) { NESO_KERNEL_ASSERT(false, k_ep); },
      Access::read(ParticleLoopIndex{}))
      ->execute();
  ASSERT_TRUE(!ep->get_flag());

  if (cell_count >= 2) {
    // Get all the particles in the last cell with even ID and move them to
    // cell 0
    auto aae = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));
    auto aaem1 = particle_sub_group(aae, cm1);

    auto la = std::make_shared<LocalArray<int>>(sycl_target,
                                                aaem1->get_npart_cell(cm1));
    la->fill(0);
    particle_loop(
        aaem1,
        [=](auto ID, auto INDEX, auto /*LA*/) {
          NESO_KERNEL_ASSERT(INDEX.cell == cm1, k_ep);
          NESO_KERNEL_ASSERT(ID.at(0) % 2 == 0, k_ep);
          NESO_KERNEL_ASSERT(INDEX.get_loop_linear_index() ==
                                 INDEX.get_sub_linear_index(),
                             k_ep);
        },
        Access::read(Sym<INT>("ID")), Access::read(ParticleLoopIndex{}),
        Access::add(la))
        ->execute();
    ASSERT_TRUE(!ep->get_flag());

    auto aa0 = particle_sub_group(A, 0);
    A->remove_particles(aa0);
    ASSERT_TRUE(aa0->create_if_required());
    ASSERT_EQ(A->get_npart_cell(0), 0);

    particle_loop(
        aaem1, [=](auto CELL) { CELL.at(0) = 0; },
        Access::write(Sym<INT>("CELL_ID")))
        ->execute();
    A->cell_move();

    ASSERT_TRUE(aa0->create_if_required());
    ASSERT_TRUE(aaem1->create_if_required());

    particle_loop(
        aa0, [=](auto ID) { NESO_KERNEL_ASSERT(ID.at(0) % 2 == 0, k_ep); },
        Access::read(Sym<INT>("ID")))
        ->execute();
    ASSERT_TRUE(!ep->get_flag());

    particle_loop(
        aam1, [=](auto ID) { NESO_KERNEL_ASSERT(ID.at(0) % 2 == 1, k_ep); },
        Access::read(Sym<INT>("ID")))
        ->execute();
    ASSERT_TRUE(!ep->get_flag());
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
