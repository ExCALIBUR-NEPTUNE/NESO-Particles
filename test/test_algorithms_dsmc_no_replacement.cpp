#include "include/test_neso_particles.hpp"

TEST(DSMCCollisionCells, collision_cell_partition) {

  int npart_cell = 511;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("SPECIES_ID"), 1);
  A->add_particle_dat(Sym<INT>("COLLISION_CELL"), 1);

  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng_state(52234234 + rank);
  std::uniform_real_distribution<> rng_dist(0.0, 1.0);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng_state); };

  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, 2);

  particle_loop(
      A,
      [=](auto INDEX, auto SPECIES_ID, auto RNG) {
        if (RNG.at(INDEX, 0) < 0.8) {
          SPECIES_ID.at(0) = 0;
        } else {
          SPECIES_ID.at(0) = 1;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("SPECIES_ID")),
      Access::read(rng_kernel))
      ->execute();

  A->remove_particles(particle_sub_group(
      A, [=](auto S) { return S.at(0) == 1; },
      Access::read(Sym<INT>("SPECIES_ID"))));

  auto aa = particle_sub_group(A, []() { return true; });

  const int num_species = 2;
  const int species_id_offset = 3;
  const int num_collision_cells = 7;

  particle_loop(
      A,
      [=](auto INDEX, auto SPECIES_ID, auto COLLISION_CELL, auto RNG) {
        SPECIES_ID.at(0) =
            RNG.at(INDEX, 0) < 0.8 ? species_id_offset : species_id_offset + 1;
        COLLISION_CELL.at(0) = INDEX.layer % num_collision_cells;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("SPECIES_ID")),
      Access::write(Sym<INT>("COLLISION_CELL")), Access::read(rng_kernel))
      ->execute();

  std::vector<INT> species_ids(num_species);
  std::iota(species_ids.begin(), species_ids.end(), species_id_offset);

  std::shared_ptr<DSMC::CollisionCellPartition> collision_cell_partition =
      std::make_shared<DSMC::CollisionCellPartition>(sycl_target, cell_count,
                                                     species_ids);

  std::vector<int> collision_cell_counts(cell_count);
  std::fill(collision_cell_counts.begin(), collision_cell_counts.end(),
            num_collision_cells);

  collision_cell_partition->construct(aa, collision_cell_counts,
                                      Sym<INT>("SPECIES_ID"), 0,
                                      Sym<INT>("COLLISION_CELL"), 0);

  auto d_collision_cell_partition = collision_cell_partition->get_device();

  ASSERT_EQ(d_collision_cell_partition.mesh_cell_count, cell_count);
  ASSERT_EQ(d_collision_cell_partition.max_num_collision_cells, 7);
  ASSERT_EQ(d_collision_cell_partition.max_num_species, 2);

  auto cdc_counts = std::make_shared<CellDatConst<int>>(
      sycl_target, cell_count, num_collision_cells, num_species);
  cdc_counts->fill(0);

  particle_loop(
      aa,
      [=](auto SPECIES_ID, auto COLLISION_CELL, auto CDC_COUNTS) {
        CDC_COUNTS.combine(COLLISION_CELL.at(0),
                           SPECIES_ID.at(0) - species_id_offset, 1);
      },
      Access::read(Sym<INT>("SPECIES_ID")),
      Access::read(Sym<INT>("COLLISION_CELL")),
      Access::reduce(cdc_counts, Kernel::plus<int>()))
      ->execute();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      aa,
      [=](auto INDEX, auto SPECIES_ID, auto COLLISION_CELL, auto CDC_COUNTS) {
        INT species_linear = 0;

        NESO_KERNEL_ASSERT(d_collision_cell_partition.get_linear_species_index(
                               SPECIES_ID.at(0), &species_linear),
                           k_ep);

        NESO_KERNEL_ASSERT(
            species_linear == SPECIES_ID.at(0) - species_id_offset, k_ep);

        const INT correct_count =
            CDC_COUNTS.at(COLLISION_CELL.at(0), species_linear);
        const INT to_test_count =
            d_collision_cell_partition.get_num_particles_cell_species(
                INDEX.cell, COLLISION_CELL.at(0), species_linear);

        NESO_KERNEL_ASSERT(correct_count == to_test_count, k_ep);

        int found = 0;
        for (INT cx = 0; cx < to_test_count; cx++) {
          const INT l = d_collision_cell_partition.get_particle_layer(
              INDEX.cell, COLLISION_CELL.at(0), species_linear, cx);
          if (l == INDEX.layer) {
            found++;
          }
        }
        NESO_KERNEL_ASSERT(found == 1, k_ep);
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<INT>("SPECIES_ID")),
      Access::read(Sym<INT>("COLLISION_CELL")), Access::read(cdc_counts))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  auto h_counts = cdc_counts->get_all_cells();

  int correct_max = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rx = 0; rx < num_collision_cells; rx++) {
      for (int cx = 0; cx < num_species; cx++) {
        correct_max = std::max(correct_max, h_counts.at(cellx)->at(rx, cx));
      }
    }
  }

  ASSERT_EQ(correct_max,
            collision_cell_partition->max_collision_cell_occupancy);

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(DSMCCollisionCells, pair_sampler_no_replacement) {

  int npart_cell = 511;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("SPECIES_ID"), 1);
  A->add_particle_dat(Sym<INT>("COLLISION_CELL"), 1);

  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng_state(52234234 + rank);
  std::uniform_real_distribution<> rng_dist(0.0, 1.0);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng_state); };

  auto rng_function =
      std::make_shared<HostRNGGenerationFunction<REAL>>(rng_lambda);
  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, 2);

  particle_loop(
      A,
      [=](auto INDEX, auto SPECIES_ID, auto RNG) {
        if (RNG.at(INDEX, 0) < 0.8) {
          SPECIES_ID.at(0) = 0;
        } else {
          SPECIES_ID.at(0) = 1;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("SPECIES_ID")),
      Access::read(rng_kernel))
      ->execute();

  A->remove_particles(particle_sub_group(
      A, [=](auto S) { return S.at(0) == 1; },
      Access::read(Sym<INT>("SPECIES_ID"))));

  auto aa = particle_sub_group(A, []() { return true; });

  const int num_species = 2;
  const int species_id_offset = 3;
  const int num_collision_cells = 7;

  particle_loop(
      A,
      [=](auto INDEX, auto SPECIES_ID, auto COLLISION_CELL, auto RNG) {
        SPECIES_ID.at(0) =
            RNG.at(INDEX, 0) < 0.8 ? species_id_offset : species_id_offset + 1;
        COLLISION_CELL.at(0) = INDEX.layer % num_collision_cells;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("SPECIES_ID")),
      Access::write(Sym<INT>("COLLISION_CELL")), Access::read(rng_kernel))
      ->execute();

  std::vector<INT> species_ids(num_species);
  std::iota(species_ids.begin(), species_ids.end(), species_id_offset);

  std::shared_ptr<DSMC::CollisionCellPartition> collision_cell_partition =
      std::make_shared<DSMC::CollisionCellPartition>(sycl_target, cell_count,
                                                     species_ids);

  std::vector<int> collision_cell_counts(cell_count);
  std::fill(collision_cell_counts.begin(), collision_cell_counts.end(),
            num_collision_cells);

  collision_cell_partition->construct(aa, collision_cell_counts,
                                      Sym<INT>("SPECIES_ID"), 0,
                                      Sym<INT>("COLLISION_CELL"), 0);

  auto pair_sampler_no_replacement =
      std::make_shared<DSMC::PairSamplerNoReplacement>(sycl_target, cell_count,
                                                       rng_function);

  // Sample no pairs and check output
  std::vector<std::vector<int>> map_cells_to_counts(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    const auto collision_cell_count = collision_cell_counts.at(cx);
    map_cells_to_counts.at(cx).resize(collision_cell_count);
    std::fill(map_cells_to_counts.at(cx).begin(),
              map_cells_to_counts.at(cx).end(), 0);
  }

  pair_sampler_no_replacement->sample(collision_cell_partition,
                                      species_id_offset, species_id_offset + 1,
                                      map_cells_to_counts);

  auto d_no_pairs = pair_sampler_no_replacement->get_pair_list();
  ASSERT_EQ(d_no_pairs.max_pair_count, 0);
  ASSERT_EQ(d_no_pairs.max_wave_count, 1);
  ASSERT_EQ(d_no_pairs.pair_count, 0);
  ASSERT_EQ(d_no_pairs.cell_count, cell_count);

  std::vector<int> h_int_cell_count(cell_count);
  std::vector<INT> h_INT_cell_count(cell_count);
  sycl_target->queue
      .memcpy(h_int_cell_count.data(), d_no_pairs.d_wave_count,
              cell_count * sizeof(int))
      .wait_and_throw();

  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(h_int_cell_count.at(cx), 1);
  }

  sycl_target->queue
      .memcpy(h_int_cell_count.data(), d_no_pairs.d_wave_offsets,
              cell_count * sizeof(int))
      .wait_and_throw();

  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(h_int_cell_count.at(cx), 0);
  }

  sycl_target->queue
      .memcpy(h_int_cell_count.data(), d_no_pairs.d_pair_counts,
              cell_count * sizeof(int))
      .wait_and_throw();

  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(h_int_cell_count.at(cx), 0);
  }

  sycl_target->queue
      .memcpy(h_INT_cell_count.data(), d_no_pairs.d_pair_counts_es,
              cell_count * sizeof(INT))
      .wait_and_throw();

  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(h_INT_cell_count.at(cx), 0);
  }

  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(d_no_pairs.h_wave_count[cx], 1);
    ASSERT_EQ(d_no_pairs.h_pair_counts[cx], 0);
  }

  auto h_no_pairs = pair_sampler_no_replacement->get_host_pair_list();
  ASSERT_EQ(h_no_pairs.size(), 0);

  sycl_target->free();
  A->domain->mesh->free();
}
