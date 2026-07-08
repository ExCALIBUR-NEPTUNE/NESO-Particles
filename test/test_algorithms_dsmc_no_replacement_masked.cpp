#include "include/test_neso_particles.hpp"

TEST(DSMCCollisionCells, pair_sampler_no_replacement_masked) {

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
  A->add_particle_dat(Sym<INT>("COUNT"), 1);
  A->add_particle_dat(Sym<INT>("MASK"), 1);

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

  auto particle_mask = std::make_shared<ParticleMask>(sycl_target);
  particle_mask->reset(A);

  particle_loop(
      A,
      [=](auto INDEX, auto SPECIES_ID, auto COLLISION_CELL, auto RNG, auto MASK,
          auto PARTICLE_MASK) {
        SPECIES_ID.at(0) =
            RNG.at(INDEX, 0) < 0.8 ? species_id_offset : species_id_offset + 1;
        COLLISION_CELL.at(0) = INDEX.layer % num_collision_cells;
        const bool mask = RNG.at(INDEX, 1) < 0.5;
        MASK.at(0) = mask ? 1 : 0;
        PARTICLE_MASK.set(INDEX, mask);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("SPECIES_ID")),
      Access::write(Sym<INT>("COLLISION_CELL")), Access::read(rng_kernel),
      Access::write(Sym<INT>("MASK")), Access::write(particle_mask))
      ->execute();

  std::vector<INT> species_ids(num_species);
  std::iota(species_ids.begin(), species_ids.end(), species_id_offset);

  std::shared_ptr<DSMC::CollisionCellPartition> collision_cell_partition =
      std::make_shared<DSMC::CollisionCellPartition>(sycl_target, cell_count,
                                                     species_ids);

  std::vector<int> collision_cell_counts(cell_count);
  std::fill(collision_cell_counts.begin(), collision_cell_counts.end(),
            num_collision_cells);

  collision_cell_partition->construct(aa, particle_mask, collision_cell_counts,
                                      Sym<INT>("SPECIES_ID"), 0,
                                      Sym<INT>("COLLISION_CELL"), 0);

  auto max_num_pairs =
      collision_cell_partition->get_collision_cell_num_pairs_instance();

  auto pair_sampler_no_replacement =
      std::make_shared<DSMC::PairSamplerNoReplacement>(sycl_target, cell_count,
                                                       rng_function);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  auto lambda_check = [&](const INT species_id_a, const INT species_id_b) {
    collision_cell_partition->get_max_num_pairs(species_id_a, species_id_b,
                                                false, max_num_pairs);

    pair_sampler_no_replacement->sample(collision_cell_partition, species_id_a,
                                        species_id_b, max_num_pairs);

    particle_loop(
        A, [=](auto COUNT) { COUNT.at(0) = 0; },
        Access::write(Sym<INT>("COUNT")))
        ->execute();

    particle_pair_loop(
        {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
            A, A, pair_sampler_no_replacement)},
        [=](auto MASK_A, auto MASK_B, auto COUNT_A, auto COUNT_B) {
          NESO_KERNEL_ASSERT(MASK_A.at(0) == 1, k_ep);
          NESO_KERNEL_ASSERT(MASK_B.at(0) == 1, k_ep);
          NESO_KERNEL_ASSERT(COUNT_A.at(0) == 0, k_ep);
          NESO_KERNEL_ASSERT(COUNT_B.at(0) == 0, k_ep);
          COUNT_A.at(0)++;
          COUNT_B.at(0)++;
        },
        Access::A(Access::read(Sym<INT>("MASK"))),
        Access::B(Access::read(Sym<INT>("MASK"))),
        Access::A(Access::write(Sym<INT>("COUNT"))),
        Access::B(Access::write(Sym<INT>("COUNT"))))
        ->execute();
    ASSERT_FALSE(ep.get_flag());

    particle_pair_loop(
        {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
            A, A, pair_sampler_no_replacement)},
        [=](auto MASK_A, auto MASK_B, auto COUNT_A, auto COUNT_B) {
          NESO_KERNEL_ASSERT(MASK_A.at(0) == 1, k_ep);
          NESO_KERNEL_ASSERT(MASK_B.at(0) == 1, k_ep);
          NESO_KERNEL_ASSERT(COUNT_A.at(0) == 1, k_ep);
          NESO_KERNEL_ASSERT(COUNT_B.at(0) == 1, k_ep);
        },
        Access::A(Access::read(Sym<INT>("MASK"))),
        Access::B(Access::read(Sym<INT>("MASK"))),
        Access::A(Access::read(Sym<INT>("COUNT"))),
        Access::B(Access::read(Sym<INT>("COUNT"))))
        ->execute();
    ASSERT_FALSE(ep.get_flag());

    particle_loop(
        A,
        [=](auto MASK, auto COUNT) {
          NESO_KERNEL_ASSERT(((MASK.at(0) == 0) && (COUNT.at(0) == 0)) ||
                                 (MASK.at(0) == 1),
                             k_ep);
        },
        Access::read(Sym<INT>("MASK")), Access::read(Sym<INT>("COUNT")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  };

  lambda_check(species_id_offset + 0, species_id_offset + 1);
  lambda_check(species_id_offset + 0, species_id_offset + 1);
  lambda_check(species_id_offset + 1, species_id_offset + 0);

  sycl_target->free();
  A->domain->mesh->free();
}
