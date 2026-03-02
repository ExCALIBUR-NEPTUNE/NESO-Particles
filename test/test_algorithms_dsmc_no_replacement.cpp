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

  sycl_target->free();
  A->domain->mesh->free();
}
