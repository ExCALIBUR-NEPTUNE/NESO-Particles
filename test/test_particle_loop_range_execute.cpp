#include "include/test_neso_particles.hpp"

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common(const int N = 1093) {
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
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 7),
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

TEST(ParticleLoop, range_execute_loop_index) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto lambda_test = [&](auto iteration_set, auto mask_func) {
    auto particle_group = get_particle_group(iteration_set);

    for (int range_size : {1, 3, 4, 7, 8}) {
      particle_loop(
          iteration_set,
          [=](auto loop_index) {
            loop_index.at(0) = -1;
            loop_index.at(1) = -1;
          },
          Access::write(Sym<INT>("LOOP_INDEX")))
          ->execute();

      auto pl = particle_loop(
          iteration_set,
          [=](Access::LoopIndex::Read cell_layer,
              Access::ParticleDat::Write<INT> loop_index) {
            loop_index.at(0) = cell_layer.cell;
            loop_index.at(1) = cell_layer.layer;
            loop_index.at(2) = cell_layer.loop_layer;
            loop_index.at(3) = cell_layer.starting_cell;
            loop_index.at(4) = cell_layer.get_local_linear_index();
            loop_index.at(5) = cell_layer.get_loop_linear_index();
            loop_index.at(6) = cell_layer.get_sub_linear_index();
          },
          Access::read(ParticleLoopIndex{}),
          Access::write(Sym<INT>("LOOP_INDEX")));

      for (int cellx = 0; cellx < cell_count; cellx += range_size) {
        pl->execute(cellx, std::min(cellx + range_size, cell_count));

        if (cellx < cell_count - (range_size + 1)) {
          auto loop_index = particle_group->get_dat(Sym<INT>("LOOP_INDEX"))
                                ->cell_dat.get_cell(cellx + range_size);
          auto id = particle_group->get_dat(Sym<INT>("ID"))
                        ->cell_dat.get_cell(cellx + range_size);
          const int nrow = loop_index->nrow;

          // for each particle in the cell
          for (int rowx = 0; rowx < nrow; rowx++) {
            if (mask_func(id->at(rowx, 0))) {
              // for each dimension
              ASSERT_EQ(loop_index->at(rowx, 0), -1);
              ASSERT_EQ(loop_index->at(rowx, 1), -1);
            }
          }
        }
      }

      INT local_linear_index = 0;
      INT sub_linear_index = 0;
      INT loop_linear_index = 0;

      for (int cellx = 0; cellx < cell_count; cellx++) {
        if (cellx % range_size == 0) {
          loop_linear_index = 0;
        }

        auto loop_index = particle_group->get_dat(Sym<INT>("LOOP_INDEX"))
                              ->cell_dat.get_cell(cellx);
        auto id =
            particle_group->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);

        const int nrow = loop_index->nrow;

        INT row_index = 0;
        // for each particle in the cell
        for (int rowx = 0; rowx < nrow; rowx++) {
          if (mask_func(id->at(rowx, 0))) {

            // for each dimension
            ASSERT_EQ(loop_index->at(rowx, 0), cellx);
            ASSERT_EQ(loop_index->at(rowx, 1), rowx);
            ASSERT_EQ(loop_index->at(rowx, 2), row_index);
            ASSERT_EQ(loop_index->at(rowx, 3),
                      (cellx / range_size) * range_size);
            ASSERT_EQ(loop_index->at(rowx, 4), local_linear_index);
            ASSERT_EQ(loop_index->at(rowx, 5), loop_linear_index);
            ASSERT_EQ(loop_index->at(rowx, 6), sub_linear_index);

            row_index++;
            sub_linear_index++;
            loop_linear_index++;
          }

          local_linear_index++;
        }
      }
    }
  };

  lambda_test(A, []([[maybe_unused]] auto id) { return true; });
  lambda_test(particle_sub_group(
                  A, [=](auto ID) { return ID.at(0) % 2 == 0; },
                  Access::read(Sym<INT>("ID"))),
              []([[maybe_unused]] auto id) { return id % 2 == 0; });

  A->free();
  sycl_target->free();
  mesh->free();
}
