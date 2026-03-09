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

  std::map<int, std::map<int, int>> correct_num_collisions_no_replacement00;
  std::map<int, std::map<int, int>> correct_num_collisions_no_replacement01;
  std::map<int, std::map<int, int>> correct_num_collisions_no_replacement11;
  std::map<int, std::map<int, int>> correct_num_collisions_replacement00;
  std::map<int, std::map<int, int>> correct_num_collisions_replacement01;
  std::map<int, std::map<int, int>> correct_num_collisions_replacement11;

  int correct_max = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rx = 0; rx < num_collision_cells; rx++) {
      for (int cx = 0; cx < num_species; cx++) {
        correct_max = std::max(correct_max, h_counts.at(cellx)->at(rx, cx));
      }

      const int npart0 = h_counts.at(cellx)->at(rx, 0);
      const int npart1 = h_counts.at(cellx)->at(rx, 1);

      correct_num_collisions_no_replacement00[cellx][rx] = npart0 / 2;
      correct_num_collisions_no_replacement01[cellx][rx] =
          std::min(npart0, npart1);
      correct_num_collisions_no_replacement11[cellx][rx] = npart1 / 2;

      constexpr int max_int = std::numeric_limits<int>::max();
      correct_num_collisions_replacement00[cellx][rx] =
          (npart0 > 1) ? max_int : 0;
      correct_num_collisions_replacement01[cellx][rx] =
          (npart0 > 0) && (npart1 > 0) ? max_int : 0;
      correct_num_collisions_replacement11[cellx][rx] =
          (npart1 > 1) ? max_int : 0;
    }
  }

  std::vector<std::vector<int>> to_test_num_collisions;
  collision_cell_partition->get_max_num_pairs(0 + species_id_offset,
                                              0 + species_id_offset, false,
                                              to_test_num_collisions);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rx = 0; rx < num_collision_cells; rx++) {
      ASSERT_EQ(to_test_num_collisions.at(cellx).at(rx),
                correct_num_collisions_no_replacement00.at(cellx).at(rx));
    }
  }
  collision_cell_partition->get_max_num_pairs(0 + species_id_offset,
                                              1 + species_id_offset, false,
                                              to_test_num_collisions);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rx = 0; rx < num_collision_cells; rx++) {
      ASSERT_EQ(to_test_num_collisions.at(cellx).at(rx),
                correct_num_collisions_no_replacement01.at(cellx).at(rx));
    }
  }
  collision_cell_partition->get_max_num_pairs(1 + species_id_offset,
                                              1 + species_id_offset, false,
                                              to_test_num_collisions);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rx = 0; rx < num_collision_cells; rx++) {
      ASSERT_EQ(to_test_num_collisions.at(cellx).at(rx),
                correct_num_collisions_no_replacement11.at(cellx).at(rx));
    }
  }

  collision_cell_partition->get_max_num_pairs(0 + species_id_offset,
                                              0 + species_id_offset, true,
                                              to_test_num_collisions);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rx = 0; rx < num_collision_cells; rx++) {
      ASSERT_EQ(to_test_num_collisions.at(cellx).at(rx),
                correct_num_collisions_replacement00.at(cellx).at(rx));
    }
  }
  collision_cell_partition->get_max_num_pairs(0 + species_id_offset,
                                              1 + species_id_offset, true,
                                              to_test_num_collisions);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rx = 0; rx < num_collision_cells; rx++) {
      ASSERT_EQ(to_test_num_collisions.at(cellx).at(rx),
                correct_num_collisions_replacement01.at(cellx).at(rx));
    }
  }
  collision_cell_partition->get_max_num_pairs(1 + species_id_offset,
                                              1 + species_id_offset, true,
                                              to_test_num_collisions);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rx = 0; rx < num_collision_cells; rx++) {
      ASSERT_EQ(to_test_num_collisions.at(cellx).at(rx),
                correct_num_collisions_replacement11.at(cellx).at(rx));
    }
  }

  ASSERT_EQ(correct_max,
            collision_cell_partition->max_collision_cell_occupancy);

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(DSMCCollisionCells, pair_sampler_no_replacement) {

  int npart_cell = 257;
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

  const auto cell_count_t = cell_count;
  auto A_t = A;
  auto lambda_check = [&](const INT species_id_a, const INT species_id_b) {
    std::vector<std::vector<int>> max_num_pairs_per_cell;
    collision_cell_partition->get_max_num_pairs(species_id_a, species_id_b,
                                                false, max_num_pairs_per_cell);

    std::vector<int> num_pairs_per_mesh_cell(cell_count_t);
    std::fill(num_pairs_per_mesh_cell.begin(), num_pairs_per_mesh_cell.end(),
              0);

    for (int cx = 0; cx < cell_count_t; cx++) {
      const auto collision_cell_count = collision_cell_counts.at(cx);
      map_cells_to_counts.at(cx).resize(collision_cell_count);
      std::fill(map_cells_to_counts.at(cx).begin(),
                map_cells_to_counts.at(cx).end(), 0);

      int npair = 0;
      for (int rx = 0; rx < collision_cell_count; rx++) {
        const int max_num_pairs = max_num_pairs_per_cell.at(cx).at(rx);
        if (max_num_pairs > 0) {
          std::uniform_int_distribution<int> rng_int(0, max_num_pairs);
          const int num_pairs_to_sample = rng_int(rng_state);
          map_cells_to_counts.at(cx).at(rx) = num_pairs_to_sample;
          npair += num_pairs_to_sample;
        }
      }
      num_pairs_per_mesh_cell.at(cx) = npair;
    }

    pair_sampler_no_replacement->sample(collision_cell_partition, species_id_a,
                                        species_id_b, map_cells_to_counts);

    auto h_pair_list = pair_sampler_no_replacement->get_host_pair_list();

    std::set<int> seen_layers;
    for (int cellx = 0; cellx < cell_count_t; cellx++) {
      const int expected_npair = num_pairs_per_mesh_cell.at(cellx);
      seen_layers.clear();
      if (expected_npair) {
        ASSERT_TRUE(h_pair_list.count(cellx));

        const auto &waves = h_pair_list.at(cellx);
        ASSERT_EQ(waves.size(), 1);

        const auto &pairs_a = waves.at(0).first;
        const auto &pairs_b = waves.at(0).second;

        const auto npairs = pairs_a.size();

        ASSERT_EQ(pairs_b.size(), npairs);
        ASSERT_EQ(pairs_b.size(), expected_npair);

        auto SPECIES_ID = A_t->get_cell(Sym<INT>("SPECIES_ID"), cellx);
        auto COLLISION_CELL = A_t->get_cell(Sym<INT>("COLLISION_CELL"), cellx);

        for (std::size_t pairx = 0; pairx < npairs; pairx++) {
          const int layer_a = pairs_a.at(pairx);
          const int layer_b = pairs_b.at(pairx);

          ASSERT_NE(layer_a, layer_b);
          ASSERT_EQ(seen_layers.count(layer_a), 0);
          seen_layers.insert(layer_a);
          ASSERT_EQ(seen_layers.count(layer_b), 0);
          seen_layers.insert(layer_b);

          auto SA = SPECIES_ID->at(layer_a, 0);
          auto SB = SPECIES_ID->at(layer_b, 0);
          ASSERT_EQ(SA, species_id_a);
          ASSERT_EQ(SB, species_id_b);

          auto CA = COLLISION_CELL->at(layer_a, 0);
          auto CB = COLLISION_CELL->at(layer_b, 0);
          ASSERT_EQ(CA, CB);
        }
      }
    }
  };

  lambda_check(species_id_offset + 0, species_id_offset + 1);
  lambda_check(species_id_offset + 0, species_id_offset + 0);
  lambda_check(species_id_offset + 1, species_id_offset + 1);

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(DSMCCollisionCells, pair_sampler_no_replacement_bias) {

  int npart_cell = 257;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto A = A_t;
  auto sycl_target = sycl_target_t;
  auto cell_count = cell_count_t;
  A->add_particle_dat(Sym<INT>("SPECIES_ID"), 1);
  A->add_particle_dat(Sym<INT>("COLLISION_CELL"), 1);
  A->add_particle_dat(Sym<INT>("LAYER"), 2);

  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng_state(52234234 + rank);
  std::uniform_real_distribution<> rng_dist(0.0, 1.0);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng_state); };

  auto rng_function =
      std::make_shared<HostRNGGenerationFunction<REAL>>(rng_lambda);
  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, 2);

  auto aa = particle_sub_group(A, []() { return true; });

  const int num_species = 2;
  const int species_id_offset = 3;
  const int num_collision_cells = 2;

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


  // Get a linear index for each particle for species/collision cell.
  auto cdc_counts = std::make_shared<CellDatConst<int>>(
      sycl_target, cell_count, num_collision_cells, num_species);
  cdc_counts->fill(0);

  particle_loop(
      aa,
      [=](auto INDEX, auto SPECIES_ID, auto COLLISION_CELL, auto CDC_COUNTS, auto LAYER) {
        const int layer = CDC_COUNTS.fetch_add(COLLISION_CELL.at(0),
                           SPECIES_ID.at(0) - species_id_offset, 1);
        LAYER.at(0) = INDEX.cell;
        LAYER.at(1) = layer;
      },
      Access::read(ParticleLoopIndex{}),
      Access::read(Sym<INT>("SPECIES_ID")),
      Access::read(Sym<INT>("COLLISION_CELL")), Access::add(cdc_counts),
      Access::write(Sym<INT>("LAYER")))
      ->execute();

  const auto k_max_collision_cell_occupancy =
    collision_cell_partition->max_collision_cell_occupancy;

  int max_species_layer = 0;
  auto h_cdc_counts = cdc_counts->get_all_cells();
  for(int cellx=0 ; cellx<cell_count ; cellx++){
    for(int rowx=0 ; rowx<num_collision_cells ; rowx++){
      for(int colx=0 ; colx<num_species ; colx++){
        max_species_layer =
            std::max(max_species_layer, h_cdc_counts.at(cellx)->at(rowx, colx));
      }
    }
  }

  // [mesh_cell][collision_cell][species_id][layer_in_species]
  const std::size_t incidence_matrix_size =
      cell_count * k_max_collision_cell_occupancy * num_species * max_species_layer;

  BufferDevice<int> d_incidence_counts(sycl_target, incidence_matrix_size);
  int * k_incidence_counts = d_incidence_counts.ptr;

  auto lambda_check = [&](const INT species_id_a, const INT species_id_b) {
    std::vector<std::vector<int>> max_num_pairs_per_cell;
    collision_cell_partition->get_max_num_pairs(species_id_a, species_id_b,
                                                false, max_num_pairs_per_cell);

    for (int cx = 0; cx < cell_count; cx++) {
      const auto collision_cell_count = collision_cell_counts.at(cx);
      map_cells_to_counts.at(cx).resize(collision_cell_count);
      std::fill(map_cells_to_counts.at(cx).begin(),
                map_cells_to_counts.at(cx).end(), 0);

      for (int rx = 0; rx < collision_cell_count; rx++) {
        const int max_num_pairs = max_num_pairs_per_cell.at(cx).at(rx);
        if (max_num_pairs > 0) {
          std::uniform_int_distribution<int> rng_int(0, max_num_pairs - 1);
          const int num_pairs_to_sample = rng_int(rng_state);
          map_cells_to_counts.at(cx).at(rx) = num_pairs_to_sample;
        }
      }
    }

    sycl_target->queue.fill<int>(k_incidence_counts, 0, incidence_matrix_size)
        .wait_and_throw();

    const int Nsample = 1;

    pair_sampler_no_replacement->sample(collision_cell_partition, species_id_a,
                                        species_id_b, map_cells_to_counts);

    for (int stepx = 0; stepx < Nsample; stepx++) {

      auto pl0 = particle_pair_loop(
          "particle_pair_loop_test",
          {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
              A, A, pair_sampler_no_replacement)},
          [=](auto LAYER_a, auto LAYER_b, auto SPECIES_a, auto SPECIES_b,
             auto COLLISION_CELL) {
              
            const INT mesh_cell = LAYER_a.at(0);
            const INT collision_cell = COLLISION_CELL.at(0);
            const INT layer_a = LAYER_a.at(1);
            const INT layer_b = LAYER_b.at(1);
            
            const INT species_a = SPECIES_a.at(0) - species_id_offset;
            const INT species_b = SPECIES_b.at(0) - species_id_offset;

            // [mesh_cell][collision_cell][species_id][layer_in_species]
            
            const INT offset_a = 
              ((mesh_cell * k_max_collision_cell_occupancy + collision_cell) *
              num_species + species_a) * max_species_layer + layer_a;

            atomic_fetch_add(k_incidence_counts + offset_a, 1);

            const INT offset_b = 
              ((mesh_cell * k_max_collision_cell_occupancy + collision_cell) *
              num_species + species_b) * max_species_layer + layer_b;

            atomic_fetch_add(k_incidence_counts + offset_b, 1);

          },
          Access::A(Access::read(Sym<INT>("LAYER"))),
          Access::B(Access::read(Sym<INT>("LAYER"))),
          Access::A(Access::read(Sym<INT>("COLLISION_CELL"))),
          Access::A(Access::read(Sym<INT>("SPECIES_ID"))),
          Access::B(Access::read(Sym<INT>("SPECIES_ID"))));

      pl0->execute();
    }










  };

  lambda_check(species_id_offset + 0, species_id_offset + 1);

  sycl_target->free();
  A->domain->mesh->free();
}
