#include <limits>
#include <neso_particles/algorithms/dsmc/collision_cell_partition.hpp>
#include <neso_particles/particle_linear_index.hpp>
#include <type_traits>

namespace NESO::Particles::DSMC {

CollisionCellPartition::CollisionCellPartition(SYCLTargetSharedPtr sycl_target,
                                               const int num_mesh_cells,
                                               std::vector<INT> species_ids)
    : sycl_target(sycl_target), num_mesh_cells(num_mesh_cells) {

  NESOASSERT(num_mesh_cells > 0,
             "Bad cell count: " + std::to_string(num_mesh_cells));

  this->h_map_species_id_linear_id =
      std::make_unique<BlockedBinaryTree<INT, INT>>(this->sycl_target);
  this->d_collision_cell_offsets =
      std::make_unique<BufferDevice<INT>>(this->sycl_target, 32);
  this->d_map_entries =
      std::make_unique<BufferDevice<int>>(this->sycl_target, 32);
  this->d_max_collision_cell_occupancy =
      std::make_unique<BufferDevice<INT>>(this->sycl_target, 1);

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

  this->num_collision_cells.resize(this->num_mesh_cells);
}

void CollisionCellPartition::construct(
    ParticleSubGroupSharedPtr particle_sub_group,
    const std::vector<int> &num_collision_cells, Sym<INT> species_id_sym,
    const int species_id_component, Sym<INT> collision_cell_sym,
    const int collision_cell_component) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CollisionCellPartition", "construct");

  NESOASSERT(num_collision_cells.size() >= this->num_mesh_cells,
             "num_collision_cells vector is too small.");

  this->particle_sub_group = particle_sub_group;
  NESOASSERT(this->particle_sub_group != nullptr,
             "particle_sub_group is nullptr, has construct been called?");

  // These two loops could be on device if needed.
  std::copy(num_collision_cells.begin(),
            num_collision_cells.begin() + this->num_mesh_cells,
            this->num_collision_cells.begin());
  const int max_num_collision_cells = *std::max_element(
      this->num_collision_cells.begin(), this->num_collision_cells.end());
  this->max_num_collision_cells = max_num_collision_cells;

  const INT layer_matrix_total_size =
      static_cast<INT>(max_num_collision_cells) *
      static_cast<INT>(this->num_mesh_cells) * this->num_species;

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

  auto e0 = reduce_values(this->sycl_target, layer_matrix_total_size,
                          k_cell_counts, sycl::maximum<INT>{},
                          this->d_max_collision_cell_occupancy->ptr);

  joint_exclusive_scan_blocking(this->sycl_target, layer_matrix_total_size + 1,
                                k_cell_counts, k_collision_cell_offsets);

  this->sycl_target->queue
      .memcpy(&this->max_collision_cell_occupancy,
              this->d_max_collision_cell_occupancy->ptr, sizeof(INT), e0)
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

void CollisionCellPartition::construct(
    ParticleSubGroupSharedPtr particle_sub_group,
    ParticleMaskSharedPtr particle_mask,
    const std::vector<int> &num_collision_cells, Sym<INT> species_id_sym,
    const int species_id_component, Sym<INT> collision_cell_sym,
    const int collision_cell_component) {
  this->particle_mask = particle_mask;
  this->construct(particle_sub_group, num_collision_cells, species_id_sym,
                  species_id_component, collision_cell_sym,
                  collision_cell_component);
}

namespace {

struct ReplacementPairCounter {
  inline INT get_num_pairs_aa(const INT num_particles_a, const INT) const {
    constexpr int max_int = std::numeric_limits<int>::max();
    // If A == B then we need at least two particles in the collision
    // cell. Otherwise we need at least one of each type.
    return num_particles_a >= 2 ? max_int : 0;
  }
  inline INT get_num_pairs_ab(const INT num_particles_a,
                              const INT num_particles_b) const {
    constexpr int max_int = std::numeric_limits<int>::max();
    // If A == B then we need at least two particles in the collision
    // cell. Otherwise we need at least one of each type.
    return (num_particles_a >= 1) && (num_particles_b >= 1) ? max_int : 0;
  }
};

struct NoReplacementPairCounter {
  inline INT get_num_pairs_aa(const INT num_particles_a, const INT) const {
    return num_particles_a / 2;
  }
  inline INT get_num_pairs_ab(const INT num_particles_a,
                              const INT num_particles_b) const {
    return sycl::min(num_particles_a, num_particles_b);
  }
};

} // namespace

void CollisionCellPartition::get_max_num_pairs(
    const INT species_id_a, const INT species_id_b, const bool replacement,
    CollisionCellNumPairsSharedPtr &map_cell_to_num_pairs) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CollisionCellPartition", "get_max_num_pairs");

  sycl::event e0;

  const INT linear_species_id_a = this->get_linear_species_id(species_id_a);
  const INT linear_species_id_b = this->get_linear_species_id(species_id_b);

  const auto k_max_num_collision_cells = this->max_num_collision_cells;
  const auto k_cell_count = this->num_mesh_cells;

  auto d_counts =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_counts->realloc_no_copy(k_max_num_collision_cells * k_cell_count);
  auto *k_counts = d_counts->ptr;

  if (this->max_collision_cell_occupancy > 0) {

    sycl::range<2> iteration_set =
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(k_cell_count, k_max_num_collision_cells));

    const std::size_t local_size =
        sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;

    const std::size_t local_size_mem =
        this->sycl_target->get_num_local_work_items(sizeof(int), local_size);

    const std::size_t e2 =
        std::min(get_prev_power_of_two(
                     static_cast<std::size_t>(k_max_num_collision_cells)),
                 local_size_mem);
    const std::size_t e1 = local_size_mem / e2;

    NESOASSERT(e2 >= 1, "Bad e2 computed");
    NESOASSERT(e1 >= 1, "Bad e1 computed");
    NESOASSERT(e1 * e2 <= local_size_mem, "Bad e1 * e2 computed");

    sycl::nd_range<3> iteration_set_mask =
        this->sycl_target->device_limits.validate_nd_range(sycl::nd_range<3>(
            sycl::range<3>(k_cell_count,
                           get_next_multiple(k_max_num_collision_cells, e1),
                           e2),
            sycl::range<3>(1, e1, e2)));

    const bool k_a_is_b = species_id_a == species_id_b;
    const auto k_map = this->get_device();
    const bool k_masks_set = this->particle_mask != nullptr;
    const MaskArrayDevice k_mask_array_device =
        k_masks_set ? this->particle_mask->get_device() : MaskArrayDevice{};
    const ParticleLinearIndexDevice k_particle_linear_index =
        get_particle_linear_index_device(
            get_particle_group(this->particle_sub_group));

    auto lambda_get_num_pairs_event =
        [&](auto pair_count_instance) -> sycl::event {
      if (k_a_is_b) {
        return sycl_target->queue.parallel_for(
            iteration_set, [=](sycl::item<2> ix) {
              const std::size_t cell_mesh = ix.get_id(0);
              const std::size_t cell_collision = ix.get_id(1);

              const INT num_particles_a = k_map.get_num_particles_cell_species(
                  cell_mesh, cell_collision, linear_species_id_a);
              const INT num_particles_b = 0;
              const int num_pairs = pair_count_instance.get_num_pairs_aa(
                  num_particles_a, num_particles_b);

              k_counts[cell_mesh * k_max_num_collision_cells + cell_collision] =
                  num_pairs;
            });

      } else {
        return sycl_target->queue.parallel_for(
            iteration_set, [=](sycl::item<2> ix) {
              const std::size_t cell_mesh = ix.get_id(0);
              const std::size_t cell_collision = ix.get_id(1);

              const INT num_particles_a = k_map.get_num_particles_cell_species(
                  cell_mesh, cell_collision, linear_species_id_a);
              const INT num_particles_b = k_map.get_num_particles_cell_species(
                  cell_mesh, cell_collision, linear_species_id_b);

              const int num_pairs = pair_count_instance.get_num_pairs_ab(
                  num_particles_a, num_particles_b);

              k_counts[cell_mesh * k_max_num_collision_cells + cell_collision] =
                  num_pairs;
            });
      }
    };

    auto lambda_get_num_pairs_mask_event =
        [&](auto pair_count_instance) -> sycl::event {
      if (k_a_is_b) {
        return this->sycl_target->queue.submit([&](auto &cgh) {
          sycl::local_accessor<int, 1> la_counts(sycl::range<1>(1 * e1 * e2),
                                                 cgh);

          cgh.parallel_for(iteration_set_mask, [=](sycl::nd_item<3> ix) {
            const std::size_t cell_mesh = ix.get_global_id(0);
            const std::size_t cell_collision = ix.get_global_id(1);

            const bool workitem_active =
                cell_collision < k_max_num_collision_cells;

            auto group = ix.get_group();
            const std::size_t l2 = ix.get_local_id(2);
            const std::size_t e2 = ix.get_local_range(2);
            const std::size_t l1 = ix.get_local_id(1);
            const std::size_t e1 = ix.get_local_range(1);

            auto lambda_get_num_particles =
                [cell_mesh, cell_collision, l2,
                 e2](const auto &k_map, const auto &k_particle_linear_index,
                     const auto &k_mask_array_device,
                     const auto linear_species_id) {
                  const INT num_particles_unmasked =
                      k_map.get_num_particles_cell_species(
                          cell_mesh, cell_collision, linear_species_id);

                  int count_local = 0;
                  for (std::size_t px = l2; px < num_particles_unmasked;
                       px += e2) {
                    const auto layer = k_map.get_particle_layer(
                        cell_mesh, cell_collision, linear_species_id, px);
                    const auto linear_index =
                        k_particle_linear_index.get_local_linear_index(
                            cell_mesh, layer);
                    const bool mask = k_mask_array_device.get(linear_index, 0);
                    count_local += mask ? 1 : 0;
                  }

                  return count_local;
                };
            const int num_particles_a_local =
                workitem_active
                    ? lambda_get_num_particles(k_map, k_particle_linear_index,
                                               k_mask_array_device,
                                               linear_species_id_a)
                    : 0;
            const int num_particles_b = 0;

            la_counts[l1 * e2 + l2] = num_particles_a_local;
            const bool is_root = Kernel::reduce_over_group_block_wise(
                la_counts.get_multi_ptr<sycl::access::decorated::no>().get(),
                ix, sycl::plus<int>{});

            const int num_particles_a = la_counts[l1 * e2 + l2];

            const int num_pairs = workitem_active
                                      ? pair_count_instance.get_num_pairs_aa(
                                            num_particles_a, num_particles_b)
                                      : 0;

            if (is_root && workitem_active) {
              k_counts[cell_mesh * k_max_num_collision_cells + cell_collision] =
                  num_pairs;
            }
          });
        });

      } else {
        return this->sycl_target->queue.submit([&](auto &cgh) {
          sycl::local_accessor<int, 1> la_counts(sycl::range<1>(1 * e1 * e2),
                                                 cgh);

          cgh.parallel_for(iteration_set_mask, [=](sycl::nd_item<3> ix) {
            const std::size_t cell_mesh = ix.get_global_id(0);
            const std::size_t cell_collision = ix.get_global_id(1);

            const bool workitem_active =
                cell_collision < k_max_num_collision_cells;

            auto group = ix.get_group();
            const std::size_t l2 = ix.get_local_id(2);
            const std::size_t e2 = ix.get_local_range(2);
            const std::size_t l1 = ix.get_local_id(1);
            const std::size_t e1 = ix.get_local_range(1);

            auto lambda_get_num_particles =
                [cell_mesh, cell_collision, l2,
                 e2](const auto &k_map, const auto &k_particle_linear_index,
                     const auto &k_mask_array_device,
                     const auto linear_species_id) {
                  const INT num_particles_unmasked =
                      k_map.get_num_particles_cell_species(
                          cell_mesh, cell_collision, linear_species_id);

                  int count_local = 0;
                  for (std::size_t px = l2; px < num_particles_unmasked;
                       px += e2) {
                    const auto layer = k_map.get_particle_layer(
                        cell_mesh, cell_collision, linear_species_id, px);
                    const auto linear_index =
                        k_particle_linear_index.get_local_linear_index(
                            cell_mesh, layer);
                    const bool mask = k_mask_array_device.get(linear_index, 0);
                    count_local += mask ? 1 : 0;
                  }

                  return count_local;
                };
            const int num_particles_a_local =
                workitem_active
                    ? lambda_get_num_particles(k_map, k_particle_linear_index,
                                               k_mask_array_device,
                                               linear_species_id_a)
                    : 0;
            la_counts[l1 * e2 + l2] = num_particles_a_local;
            const bool is_root = Kernel::reduce_over_group_block_wise(
                la_counts.get_multi_ptr<sycl::access::decorated::no>().get(),
                ix, sycl::plus<int>{});

            const int num_particles_a = la_counts[l1 * e2 + l2];

            const int num_particles_b_local =
                workitem_active
                    ? lambda_get_num_particles(k_map, k_particle_linear_index,
                                               k_mask_array_device,
                                               linear_species_id_b)
                    : 0;

            la_counts[l1 * e2 + l2] = num_particles_b_local;
            Kernel::reduce_over_group_block_wise(
                la_counts.get_multi_ptr<sycl::access::decorated::no>().get(),
                ix, sycl::plus<int>{});

            const int num_particles_b = la_counts[l1 * e2 + l2];

            const int num_pairs = workitem_active
                                      ? pair_count_instance.get_num_pairs_ab(
                                            num_particles_a, num_particles_b)
                                      : 0;

            if (is_root && workitem_active) {
              k_counts[cell_mesh * k_max_num_collision_cells + cell_collision] =
                  num_pairs;
            }
          });
        });
      }
    };

    if (replacement) {
      if (k_masks_set) {
        e0 = lambda_get_num_pairs_mask_event(ReplacementPairCounter{});
      } else {
        e0 = lambda_get_num_pairs_event(ReplacementPairCounter{});
      }
    } else {
      if (k_masks_set) {
        e0 = lambda_get_num_pairs_mask_event(NoReplacementPairCounter{});
      } else {
        e0 = lambda_get_num_pairs_event(NoReplacementPairCounter{});
      }
    }
  }

  if (map_cell_to_num_pairs.get() == nullptr) {
    map_cell_to_num_pairs = this->get_collision_cell_num_pairs_instance();
  }
  if ((map_cell_to_num_pairs->num_mesh_cells != this->num_mesh_cells) ||
      (map_cell_to_num_pairs->max_num_collision_cells !=
       this->max_num_collision_cells)) {
    map_cell_to_num_pairs = this->get_collision_cell_num_pairs_instance();
  }

  e0.wait_and_throw();

  if (this->max_collision_cell_occupancy > 0) {

    e0 = this->sycl_target->queue.memcpy(
        map_cell_to_num_pairs->get_host_pointer(), k_counts,
        k_cell_count * k_max_num_collision_cells * sizeof(int));

    e0.wait_and_throw();

  } else {
    map_cell_to_num_pairs->fill(0);
  }

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_counts);

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
          this->num_mesh_cells,
          this->max_num_collision_cells,
          this->num_species,
          this->d_map_entries->ptr};
}

CollisionCellNumPairsSharedPtr
CollisionCellPartition::get_collision_cell_num_pairs_instance() {
  return std::make_shared<CollisionCellNumPairs>(
      this->sycl_target, this->num_mesh_cells, this->max_num_collision_cells);
}

ParticleMaskSharedPtr CollisionCellPartition::get_particle_mask() {
  return this->particle_mask;
}

} // namespace NESO::Particles::DSMC
