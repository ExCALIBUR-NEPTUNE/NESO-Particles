#include <neso_particles/algorithms/dsmc/collision_cell_partition.hpp>

namespace NESO::Particles::DSMC {

CollisionCellPartition::CollisionCellPartition(SYCLTargetSharedPtr sycl_target,
                                               const int cell_count,
                                               std::vector<INT> species_ids)
    : sycl_target(sycl_target), cell_count(cell_count) {

  NESOASSERT(cell_count > 0, "Bad cell count: " + std::to_string(cell_count));

  this->h_map_species_id_linear_id =
      std::make_unique<BlockedBinaryTree<INT, INT>>(this->sycl_target);
  this->d_collision_cell_offsets =
      std::make_unique<BufferDevice<INT>>(this->sycl_target, 32);
  this->d_map_entries =
      std::make_unique<BufferDevice<int>>(this->sycl_target, 32);

  std::set<INT> species_id_set;
  species_id_set.insert(species_ids.begin(), species_ids.end());
  this->species_ids.reserve(species_id_set.size());

  {
    INT index = 0;
    for (auto id : species_id_set) {
      this->h_map_species_id_linear_id->add(id, index);
      this->map_species_id_to_linear[id] = index;
      this->species_ids.push_back(id);
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

  this->particle_sub_group = particle_sub_group;

  // These two loops could be on device if needed.
  std::copy(collision_cell_counts.begin(),
            collision_cell_counts.begin() + this->cell_count,
            this->collision_cell_counts.begin());
  const int max_num_collision_cells = *std::max_element(
      this->collision_cell_counts.begin(), this->collision_cell_counts.end());
  this->max_num_collision_cells = max_num_collision_cells;

  const INT layer_matrix_total_size =
      static_cast<INT>(max_num_collision_cells) *
      static_cast<INT>(this->cell_count) * this->num_species;

  auto d_cell_counts =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_cell_counts->realloc_no_copy(layer_matrix_total_size + 1);
  auto *k_cell_counts = d_cell_counts->ptr;

  auto d_layers =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_layers->realloc_no_copy(particle_sub_group->get_npart_local());
  auto *k_layers = d_layers->ptr;

  EventStack es;
  es.push(this->sycl_target->queue.fill<INT>(k_cell_counts, 0,
                                             layer_matrix_total_size + 1));

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
              static_cast<INT>(1));

          k_layers[linear_index] = new_layer;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(species_id_sym),
      Access::read(collision_cell_sym))
      ->execute();

  this->d_collision_cell_offsets->realloc_no_copy(layer_matrix_total_size + 1);
  INT *k_collision_cell_offsets = this->d_collision_cell_offsets->ptr;

  // TODO make joint exclusive scan more efficient for large arrays
  joint_exclusive_scan(this->sycl_target, layer_matrix_total_size + 1,
                       k_cell_counts, k_collision_cell_offsets)
      .wait_and_throw();

  this->d_map_entries->realloc_no_copy(particle_sub_group->get_npart_local());
  auto k_map_entries = this->d_map_entries->ptr;

  auto k_map = this->get_device();

  particle_loop(
      "CollisionCellPartition::populate_map", particle_sub_group,
      [=](auto INDEX, auto SPECIES_ID, auto COLLISION_CELL) {
        const INT species_id_label = SPECIES_ID.at(species_id_component);
        INT species_id_linear = 0;
        const bool found = k_map.get_linear_species_index(species_id_label,
                                                          &species_id_linear);

        if (found) {
          const INT offset_species = k_map.get_offset_cell_species(
              INDEX.cell, COLLISION_CELL.at(collision_cell_component),
              species_id_linear);

          const INT offset_particle =
              k_map.d_collision_cell_offsets[offset_species] +
              k_layers[INDEX.get_loop_linear_index()];

          k_map_entries[offset_particle] = INDEX.layer;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(species_id_sym),
      Access::read(collision_cell_sym))
      ->execute();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_layers);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_cell_counts);

  this->sycl_target->profile_map.end_region(r0);
}

INT CollisionCellPartition::get_linear_species_id(const INT species_id) {

  NESOASSERT(this->map_species_id_to_linear.count(species_id),
             "Could not find requested species id: " +
                 std::to_string(species_id));

  return this->map_species_id_to_linear[species_id];
}

CollisionCellPartitionDevice CollisionCellPartition::get_device() {

  return {this->d_collision_cell_offsets->ptr,
          this->h_map_species_id_linear_id->root,
          this->cell_count,
          this->max_num_collision_cells,
          this->num_species,
          this->d_map_entries->ptr};
}

} // namespace NESO::Particles::DSMC
