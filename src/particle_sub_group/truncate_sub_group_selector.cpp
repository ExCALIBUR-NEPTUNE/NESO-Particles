#include <neso_particles/particle_sub_group/truncate_sub_group_selector.hpp>

namespace NESO::Particles::ParticleSubGroupImplementation {

template TruncateSubGroupSelector::TruncateSubGroupSelector(
    std::shared_ptr<ParticleGroup> parent, const int num_particles);
template TruncateSubGroupSelector::TruncateSubGroupSelector(
    std::shared_ptr<ParticleSubGroup> parent, const int num_particles);

void TruncateSubGroupSelector::create(Selection *created_selection) {

  const int cell_count = this->particle_group->domain->mesh->get_cell_count();
  auto sycl_target = this->particle_group->sycl_target;

  auto [h_npart_cell_ptr, d_npart_cell_ptr, h_npart_cell_es_ptr,
        d_npart_cell_es_ptr] = this->sub_group_particle_map->get_helper_ptrs();

  auto k_npart_cell = d_npart_cell_ptr;
  auto pr0 = sycl_target->profile_map.start_region("TruncateSubGroupSelector",
                                                   "create");

  auto d_INT =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_INT->realloc_no_copy(cell_count);
  auto k_INT = d_INT->ptr;

  int *orig_d_npart_cell = nullptr;
  int npart_local = -1;
  MapLoopLayerToLayer k_parent_map;

  const int k_num_particles = this->num_particles;
  if (this->parent_is_whole_group) {

    auto dat_cell_id = this->particle_group->cell_id_dat;
    orig_d_npart_cell = dat_cell_id->d_npart_cell;

  } else {

    NESOASSERT(this->particle_sub_group != nullptr,
               "Parent is a sub group but we have no ParticleSubGroup.");

    this->particle_sub_group->create_if_required();
    auto s_parent = this->particle_sub_group->get_selection();

    orig_d_npart_cell = s_parent.d_npart_cell;
    k_parent_map = s_parent.d_map_cells_to_particles;
  }

  sycl_target->queue
      .parallel_for(sycl::range<1>(cell_count),
                    [=](auto idx) {
                      const int new_occupancy =
                          sycl::min(k_num_particles, orig_d_npart_cell[idx]);
                      k_npart_cell[idx] = new_occupancy;
                      k_INT[idx] = new_occupancy;
                    })
      .wait_and_throw();

  auto e0 = sycl_target->queue.memcpy(h_npart_cell_ptr, k_npart_cell,
                                      cell_count * sizeof(int));

  auto e1 =
      joint_exclusive_scan(sycl_target, cell_count, k_INT, d_npart_cell_es_ptr);
  e0.wait_and_throw();

  int max_occ = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    max_occ = std::max(max_occ, h_npart_cell_ptr[cellx]);
  }

  e1.wait_and_throw();

  sycl_target->queue
      .memcpy(h_npart_cell_es_ptr, d_npart_cell_es_ptr,
              sizeof(INT) * cell_count)
      .wait_and_throw();

  npart_local = static_cast<int>(h_npart_cell_es_ptr[cell_count - 1]) +
                h_npart_cell_ptr[cell_count - 1];

  this->sub_group_particle_map->create(0, cell_count, h_npart_cell_ptr,
                                       h_npart_cell_es_ptr);

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;
  const std::size_t range_cell = get_next_multiple(max_occ, local_size);
  const auto k_cell_starts = this->sub_group_particle_map->d_cell_starts->ptr;

  if (max_occ > 0) {
    if (this->parent_is_whole_group) {
      sycl_target->queue
          .parallel_for(
              sycl_target->device_limits.validate_nd_range(
                  sycl::nd_range<2>(sycl::range<2>(cell_count, range_cell),
                                    sycl::range<2>(1, local_size))),
              [=](sycl::nd_item<2> idx) {
                const std::size_t index_cell = idx.get_global_id(0);
                const std::size_t index_layer = idx.get_global_id(1);
                if (index_layer < k_npart_cell[index_cell]) {
                  k_cell_starts[index_cell][index_layer] = index_layer;
                }
              })
          .wait_and_throw();
    } else {
      sycl_target->queue
          .parallel_for(
              sycl_target->device_limits.validate_nd_range(
                  sycl::nd_range<2>(sycl::range<2>(cell_count, range_cell),
                                    sycl::range<2>(1, local_size))),
              [=](sycl::nd_item<2> idx) {
                const std::size_t index_cell = idx.get_global_id(0);
                const std::size_t index_layer = idx.get_global_id(1);
                if (index_layer < k_npart_cell[index_cell]) {
                  k_cell_starts[index_cell][index_layer] =
                      k_parent_map.map_loop_layer_to_layer(index_cell,
                                                           index_layer);
                }
              })
          .wait_and_throw();
    }
  }

  NESOASSERT(npart_local > -1, "Bad npart local");

  created_selection->npart_local = npart_local;
  created_selection->ncell = cell_count;
  created_selection->h_npart_cell = h_npart_cell_ptr;
  created_selection->d_npart_cell = d_npart_cell_ptr;
  created_selection->d_npart_cell_es = d_npart_cell_es_ptr;
  created_selection->d_map_cells_to_particles = {
      this->sub_group_particle_map->d_cell_starts->ptr};

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_INT);
  sycl_target->profile_map.end_region(pr0);
}

} // namespace NESO::Particles::ParticleSubGroupImplementation

namespace NESO::Particles {

/**
 * Create a ParticleSubGroup from a parent by selecting only the first n
 * particles in each cell.
 *
 * @param parent Particle(Sub)Group which is the parent.
 * @param num_particles Number of particles to keep from each cell.
 * @param make_static Make the ParticleSubGroup static (default false).
 */
std::shared_ptr<ParticleSubGroup>
particle_sub_group_truncate(std::shared_ptr<ParticleGroup> particle_group,
                            const int num_particles, const bool make_static) {

  auto s = std::make_shared<
      ParticleSubGroupImplementation::TruncateSubGroupSelector>(particle_group,
                                                                num_particles);

  auto group = std::make_shared<ParticleSubGroup>(
      std::dynamic_pointer_cast<
          ParticleSubGroupImplementation::SubGroupSelector>(s));

  group->static_status(make_static);

  return group;
}

/**
 * Create a ParticleSubGroup from a parent by selecting only the first n
 * particles in each cell.
 *
 * @param parent Particle(Sub)Group which is the parent.
 * @param num_particles Number of particles to keep from each cell.
 * @param make_static Make the ParticleSubGroup static (default false).
 */
std::shared_ptr<ParticleSubGroup> particle_sub_group_truncate(
    std::shared_ptr<ParticleSubGroup> particle_sub_group,
    const int num_particles, const bool make_static) {

  auto s = std::make_shared<
      ParticleSubGroupImplementation::TruncateSubGroupSelector>(
      particle_sub_group, num_particles);

  auto group = std::make_shared<ParticleSubGroup>(
      std::dynamic_pointer_cast<
          ParticleSubGroupImplementation::SubGroupSelector>(s));

  group->static_status(make_static);

  return group;
}
} // namespace NESO::Particles
