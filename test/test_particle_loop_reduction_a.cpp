#include "include/test_neso_particles.hpp"

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common(const int N = 10083) {
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
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 4),
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

template <typename T, typename GROUP_TYPE>
void wrapper_cdc_reduction_base(GROUP_TYPE A, SYCLTargetSharedPtr sycl_target,
                                int cell_count, int ncomp) {
  auto Vcorrect =
      std::make_shared<CellDatConst<T>>(sycl_target, cell_count, ncomp, 2);
  auto Vto_test =
      std::make_shared<CellDatConst<T>>(sycl_target, cell_count, ncomp, 2);

  Vcorrect->fill(3);
  Vto_test->fill(3);

  particle_loop(
      A,
      [=](auto V, auto CDC, auto LOOP_INDEX) {
        for (int dx = 0; dx < ncomp; dx++) {
          CDC.combine(dx, 0, V.at(dx % 3));
        }
        for (int dx = 0; dx < ncomp; dx++) {
          CDC.combine(dx, 1, V.at(dx % 3) + 0.1 * dx);
        }
        LOOP_INDEX.at(0) = 4;
      },
      Access::read(Sym<REAL>("V")), Access::reduce(Vto_test, Kernel::plus<T>{}),
      Access::write(Sym<INT>("LOOP_INDEX")))
      ->execute();

  particle_loop(
      A,
      [=](auto V, auto CDC, auto LOOP_INDEX) {
        for (int dx = 0; dx < ncomp; dx++) {
          CDC.fetch_add(dx, 0, V.at(dx % 3));
        }
        for (int dx = 0; dx < ncomp; dx++) {
          CDC.fetch_add(dx, 1, V.at(dx % 3) + 0.1 * dx);
        }
        LOOP_INDEX.at(1) = 4;
      },
      Access::read(Sym<REAL>("V")), Access::add(Vcorrect),
      Access::write(Sym<INT>("LOOP_INDEX")))
      ->execute();

  auto ep = ErrorPropagate(sycl_target);
  auto k_ep = ep.device_ptr();
  particle_loop(
      A,
      [=](auto LOOP_INDEX) {
        NESO_KERNEL_ASSERT(LOOP_INDEX.at(1) == LOOP_INDEX.at(0), k_ep);
      },
      Access::read(Sym<INT>("LOOP_INDEX")))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  auto correct = Vcorrect->get_all_cells();
  auto to_test = Vto_test->get_all_cells();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int dx = 0; dx < ncomp; dx++) {
      auto err = relative_error(correct.at(cellx)->at(dx, 0),
                                to_test.at(cellx)->at(dx, 0));
      ASSERT_TRUE(err < 1.0e-8);
      err = relative_error(correct.at(cellx)->at(dx, 1),
                           to_test.at(cellx)->at(dx, 1));
      ASSERT_TRUE(err < 1.0e-8);
    }
  }
}

} // namespace

TEST(ParticleLoopReduction, base) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;
  for (int dx = 1; dx < 8; dx++) {
    wrapper_cdc_reduction_base<REAL>(A, sycl_target, cell_count, dx);
  }
  for (int dx = 1; dx < 8; dx++) {
    wrapper_cdc_reduction_base<INT>(A, sycl_target, cell_count, dx);
  }
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopReduction, base_sub_group) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto AA = particle_sub_group(A);

  for (int dx = 1; dx < 8; dx++) {
    wrapper_cdc_reduction_base<REAL>(AA, sycl_target, cell_count, dx);
  }
  for (int dx = 1; dx < 8; dx++) {
    wrapper_cdc_reduction_base<INT>(AA, sycl_target, cell_count, dx);
  }

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  for (int dx = 1; dx < 8; dx++) {
    wrapper_cdc_reduction_base<REAL>(aa, sycl_target, cell_count, dx);
  }
  for (int dx = 1; dx < 8; dx++) {
    wrapper_cdc_reduction_base<INT>(aa, sycl_target, cell_count, dx);
  }

  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopReduction, minimum) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  {
    auto cdc_real =
        std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 2, 1);
    cdc_real->fill(1.0);

    auto cdc_real_correct =
        std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 2, 1);
    cdc_real_correct->fill(1.0);

    particle_loop(
        A,
        [=](auto P, auto CDC) {
          CDC.combine(0, 0, P.at(0));
          CDC.combine(1, 0, P.at(1));
        },
        Access::read(Sym<REAL>("P")),
        Access::reduce(cdc_real, Kernel::minimum<REAL>()))
        ->execute();

    particle_loop(
        A,
        [=](auto P, auto CDC) {
          CDC.fetch_min(0, 0, P.at(0));
          CDC.fetch_min(1, 0, P.at(1));
        },
        Access::read(Sym<REAL>("P")), Access::min(cdc_real_correct))
        ->execute();

    auto correct = cdc_real_correct->get_all_cells();
    auto to_test = cdc_real->get_all_cells();
    for (int cellx = 0; cellx < cell_count; cellx++) {
      for (int dx = 0; dx < 2; dx++) {
        // These should be bitwise equivalent.
        ASSERT_EQ(correct.at(cellx)->at(dx, 0), to_test.at(cellx)->at(dx, 0));
      }
    }
  }

  {
    auto cdc_int =
        std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, 2, 1);
    cdc_int->fill(1);

    auto cdc_int_correct =
        std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, 2, 1);
    cdc_int_correct->fill(1);

    particle_loop(
        A,
        [=](auto ID, auto CDC) {
          CDC.combine(0, 0, ID.at(0));
          CDC.combine(1, 0, -ID.at(0));
        },
        Access::read(Sym<INT>("ID")),
        Access::reduce(cdc_int, Kernel::minimum<INT>()))
        ->execute();

    particle_loop(
        A,
        [=](auto ID, auto CDC) {
          CDC.fetch_min(0, 0, ID.at(0));
          CDC.fetch_min(1, 0, -ID.at(0));
        },
        Access::read(Sym<INT>("ID")), Access::min(cdc_int_correct))
        ->execute();

    auto correct_int = cdc_int_correct->get_all_cells();
    auto to_test_int = cdc_int->get_all_cells();
    for (int cellx = 0; cellx < cell_count; cellx++) {
      for (int dx = 0; dx < 2; dx++) {
        // These should be bitwise equivalent.
        ASSERT_EQ(correct_int.at(cellx)->at(dx, 0),
                  to_test_int.at(cellx)->at(dx, 0));
      }
    }
  }

  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopReduction, maximum) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  {
    auto cdc_real =
        std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 2, 1);
    cdc_real->fill(1.0);

    auto cdc_real_correct =
        std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 2, 1);
    cdc_real_correct->fill(1.0);

    particle_loop(
        A,
        [=](auto P, auto CDC) {
          CDC.combine(0, 0, P.at(0));
          CDC.combine(1, 0, P.at(1));
        },
        Access::read(Sym<REAL>("P")),
        Access::reduce(cdc_real, Kernel::maximum<REAL>()))
        ->execute();

    particle_loop(
        A,
        [=](auto P, auto CDC) {
          CDC.fetch_max(0, 0, P.at(0));
          CDC.fetch_max(1, 0, P.at(1));
        },
        Access::read(Sym<REAL>("P")), Access::max(cdc_real_correct))
        ->execute();

    auto correct = cdc_real_correct->get_all_cells();
    auto to_test = cdc_real->get_all_cells();
    for (int cellx = 0; cellx < cell_count; cellx++) {
      for (int dx = 0; dx < 2; dx++) {
        // These should be bitwise equivalent.
        ASSERT_EQ(correct.at(cellx)->at(dx, 0), to_test.at(cellx)->at(dx, 0));
      }
    }
  }

  {
    auto cdc_int =
        std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, 2, 1);
    cdc_int->fill(1);

    auto cdc_int_correct =
        std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, 2, 1);
    cdc_int_correct->fill(1);

    particle_loop(
        A,
        [=](auto ID, auto CDC) {
          CDC.combine(0, 0, ID.at(0));
          CDC.combine(1, 0, -ID.at(0));
        },
        Access::read(Sym<INT>("ID")),
        Access::reduce(cdc_int, Kernel::maximum<INT>()))
        ->execute();

    particle_loop(
        A,
        [=](auto ID, auto CDC) {
          CDC.fetch_max(0, 0, ID.at(0));
          CDC.fetch_max(1, 0, -ID.at(0));
        },
        Access::read(Sym<INT>("ID")), Access::max(cdc_int_correct))
        ->execute();

    auto correct_int = cdc_int_correct->get_all_cells();
    auto to_test_int = cdc_int->get_all_cells();
    for (int cellx = 0; cellx < cell_count; cellx++) {
      for (int dx = 0; dx < 2; dx++) {
        // These should be bitwise equivalent.
        ASSERT_EQ(correct_int.at(cellx)->at(dx, 0),
                  to_test_int.at(cellx)->at(dx, 0));
      }
    }
  }

  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopReduction, min_max_same_loop) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto cdc_max =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, 1, 1);
  cdc_max->fill(1);

  auto cdc_max_correct =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, 1, 1);
  cdc_max_correct->fill(1);

  auto cdc_min =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, 1, 1);
  cdc_min->fill(10000000);

  auto cdc_min_correct =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, 1, 1);
  cdc_min_correct->fill(10000000);

  particle_loop(
      A,
      [=](auto ID, auto MIN, auto MAX) {
        MIN.combine(0, 0, ID.at(0));
        MAX.combine(0, 0, ID.at(0));
      },
      Access::read(Sym<INT>("ID")),
      Access::reduce(cdc_min, Kernel::minimum<INT>()),
      Access::reduce(cdc_max, Kernel::maximum<INT>()))
      ->execute();

  particle_loop(
      A,
      [=](auto ID, auto MIN, auto MAX) {
        MIN.fetch_min(0, 0, ID.at(0));
        MAX.fetch_max(0, 0, ID.at(0));
      },
      Access::read(Sym<INT>("ID")), Access::min(cdc_min_correct),
      Access::max(cdc_max_correct))
      ->execute();

  auto correct_min = cdc_min_correct->get_all_cells();
  auto to_test_min = cdc_min->get_all_cells();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    // These should be bitwise equivalent.
    ASSERT_EQ(correct_min.at(cellx)->at(0, 0), to_test_min.at(cellx)->at(0, 0));
  }

  auto correct_max = cdc_max_correct->get_all_cells();
  auto to_test_max = cdc_max->get_all_cells();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    // These should be bitwise equivalent.
    ASSERT_EQ(correct_max.at(cellx)->at(0, 0), to_test_max.at(cellx)->at(0, 0));
  }

  sycl_target->free();
  mesh->free();
}
