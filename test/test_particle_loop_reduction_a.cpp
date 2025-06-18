#include "include/test_neso_particles.hpp"

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common(const int N = 20) {
  std::vector<int> dims(ndim);
  dims[0] = 2;
  dims[1] = 2;

  const double cell_extent = 1.0;
  const int subdivision_order = 0;

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
      [=](auto V, auto CDC) {
        for (int dx = 0; dx < ncomp; dx++) {
          CDC.combine(dx, 0, V.at(dx % 3));
        }
        for (int dx = 0; dx < ncomp; dx++) {
          CDC.combine(dx, 1, V.at(dx % 3) + 0.1 * dx);
        }
      },
      Access::read(Sym<REAL>("V")), Access::reduce(Vto_test, Kernel::plus<T>{}))
      ->execute();

  particle_loop(
      A,
      [=](auto V, auto CDC) {
        for (int dx = 0; dx < ncomp; dx++) {
          CDC.fetch_add(dx, 0, V.at(dx % 3));
        }
        for (int dx = 0; dx < ncomp; dx++) {
          CDC.fetch_add(dx, 1, V.at(dx % 3) + 0.1 * dx);
        }
      },
      Access::read(Sym<REAL>("V")), Access::add(Vcorrect))
      ->execute();

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
