#include <neso_particles/algorithms/dsmc/collision_cell_partition.hpp>

namespace NESO::Particles::DSMC {

CollisionCellPartition::CollisionCellPartition(SYCLTargetSharedPtr sycl_target,
                                               const int cell_count,
                                               std::vector<INT> species_ids)
    : sycl_target(sycl_target), cell_count(cell_count),
      species_ids(species_ids) {

  NESOASSERT(cell_count > 0, "Bad cell count: " + std::to_string(cell_count));

  this->h_map_species_id_linear_id =
      std::make_unique<BlockedBinaryTree<INT, INT>>(this->sycl_target);
  this->d_collision_cell_offsets =
      std::make_unique<BufferDevice<INT>>(this->sycl_target, 32);

  {
    INT index = 0;
    for (auto id : this->species_ids) {
      this->h_map_species_id_linear_id->add(id, index);
      index++;
    }
    this->num_species = index;
  }

  this->collision_cell_counts.resize(this->cell_count);
}

void CollisionCellPartition::construct(
    ParticleSubGroupSharedPtr particle_sub_group,
    std::vector<int> &collision_cell_counts, Sym<INT> species_id_sym,
    const int species_id_component, Sym<INT> collision_cell_sym,
    const int collision_cell_component) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CollisionCellPartition", "construct");

  NESOASSERT(collision_cell_counts.size() >= this->cell_count,
             "collision_cell_counts vector is too small.");

  // These two loops could be on device if needed.
  std::copy(collision_cell_counts.begin(),
            collision_cell_counts.begin() + this->cell_count,
            this->collision_cell_counts.begin());
  const int max_num_collision_cells = *std::max_element(
      this->collision_cell_counts.begin(), this->collision_cell_counts.end());

  const INT layer_matrix_total_size =
      static_cast<INT>(max_num_collision_cells) *
      static_cast<INT>(this->cell_count) * this->num_species;

  auto d_cell_counts =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_cell_counts->realloc_no_copy(layer_matrix_total_size);
  auto *k_cell_counts = d_cell_counts->ptr;

  auto d_layers =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_layers->realloc_no_copy(particle_sub_group->get_npart_local());
  auto *k_layers = d_layers->ptr;

  EventStack es;
  es.push(this->sycl_target->queue.fill<int>(k_cell_counts, 0,
                                             layer_matrix_total_size));

  auto k_tree_root = this->h_map_species_id_linear_id->root;
  const INT k_cell_count = this->cell_count;
  const INT k_num_species = this->num_species;
  es.wait();

  // This loop could be atomics into local memory to determine the layers rather
  // than a particle loop into global memory. i.e. atomics into local memory
  // then reduce the values and ex scan such that the counts never have to exist
  // in a buffer?
  particle_loop(
      "CollisionCellPartition::determine_layers", particle_sub_group,
      [=](auto INDEX, auto SPECIES_ID, auto COLLISION_CELL) {
        const INT species_id = SPECIES_ID.at(species_id_component);

        INT species_id_linear = -1;
        const bool found = k_tree_root->get(species_id, &species_id_linear);
        if (found) {

          const INT collision_cell =
              COLLISION_CELL.at(collision_cell_component);
          const auto linear_index = INDEX.get_loop_linear_index();
          const auto mesh_cell = INDEX.cell;

          const int new_layer = atomic_fetch_add(
              k_cell_counts +
                  mesh_cell * max_num_collision_cells * k_num_species +
                  collision_cell * k_num_species + species_id_linear,
              1);

          k_layers[linear_index] = new_layer;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(species_id_sym),
      Access::read(collision_cell_sym))
      ->execute();

  // The map reallocation can be async with the above atomics loop.

  // TODO get max species count in dsmc cells from above array

  const INT total_num_collision_cells = max_num_collision_cells * k_cell_count;

  auto d_array_sizes =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_array_sizes->realloc_no_copy(total_num_collision_cells);
  auto *k_array_sizes = d_array_sizes->ptr;
  auto d_array_offsets =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_array_offsets->realloc_no_copy(total_num_collision_cells);
  auto *k_array_offsets = d_array_offsets->ptr;

  this->sycl_target->queue
      .parallel_for(sycl::range<1>(total_num_collision_cells),
                    [=](auto ix) {
                      k_array_sizes[ix] = k_num_species;
                      k_array_offsets[ix] = ix * k_num_species;
                    })
      .wait_and_throw();

  // After this call we have the incscan of the species counts in each collision
  // cell.
  joint_inclusive_scan_n(this->sycl_target, total_num_collision_cells,
                         k_array_sizes, k_array_offsets, k_cell_counts,
                         k_cell_counts)
      .wait_and_throw();

  this->d_collision_cell_offsets->realloc_no_copy(total_num_collision_cells);
  INT *k_collision_cell_offsets = this->d_collision_cell_offsets->ptr;

  this->sycl_target->queue
      .parallel_for(sycl::range<1>(total_num_collision_cells),
                    [=](auto ix) {
                      const INT index =
                          ix * k_num_species + (k_num_species - 1);
                      const INT contribution = k_cell_counts[index];
                      k_collision_cell_offsets[ix] = contribution;
                    })
      .wait_and_throw();

  int last_collision_cell_sum = 0;
  this->sycl_target->queue
      .memcpy(&last_collision_cell_sum,
              k_cell_counts + layer_matrix_total_size - 1, sizeof(int))
      .wait_and_throw();

  joint_exclusive_scan(this->sycl_target, total_num_collision_cells,
                       k_collision_cell_offsets, k_collision_cell_offsets)
      .wait_and_throw();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_array_offsets);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_array_sizes);

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_layers);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_cell_counts);
  this->sycl_target->profile_map.end_region(r0);
}
} // namespace NESO::Particles::DSMC
