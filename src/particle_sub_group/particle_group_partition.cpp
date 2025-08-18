#include <neso_particles/common_impl.hpp>
#include <neso_particles/particle_sub_group/particle_group_partition.hpp>

namespace NESO::Particles {

namespace ParticleSubGroupImplementation {

void ParticleGroupPartitionSelector::create(Selection *created_selection) {
  this->create_handle(created_selection);
}

std::shared_ptr<SubGroupParticleMap>
ParticleGroupPartitionSelector::get_sub_group_particle_map() {
  return this->sub_group_particle_map;
}

void ParticleGroupPartitionSelector::set_create_handle(
    std::function<void(Selection *created_selection)> create_handle) {
  this->create_handle = create_handle;
}
void ParticleGroupPartitionSelector::set_destroy_handle(
    std::function<void()> destroy_handle) {
  this->destroy_handle = destroy_handle;
}

void ParticleGroupPartitioner::create_indexed(const std::size_t index,
                                              Selection *created_selection) {
  NESOASSERT(index < this->num_partitions, "Bad index passed.");

  // This is where the versions get checked and the internal representation
  // for all the partitions gets updated.
  this->get(nullptr);
  *created_selection = this->partition_selections.at(index);
}

void ParticleGroupPartitioner::create(Selection *created_selection) {
  if (this->num_partitions == 0) {
    return;
  }

  NESOASSERT(created_selection == nullptr, "Expected a nullptr.");
  auto sycl_target = this->particle_group->sycl_target;
  const std::size_t cell_count = static_cast<std::size_t>(
      this->particle_group->domain->mesh->get_cell_count());

  // Find once at the begining of this function which of the selectors
  // actually still exist and have not been deleted.
  std::vector<int> still_exists(this->num_partitions);
  for (std::size_t px = 0; px < this->num_partitions; px++) {
    still_exists[px] =
        static_cast<int>(!this->sub_group_particle_maps.at(px).expired());
  }
  this->la_still_exists->set(still_exists);

  // Create overall map.
  auto d_cell_counts =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);

  auto d_cell_counts_INT =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  auto d_particle_layers =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);

  const std::size_t cell_count_all_partitions =
      cell_count * this->num_partitions;
  d_cell_counts->realloc_no_copy(cell_count_all_partitions);
  d_particle_layers->realloc_no_copy(this->particle_group->get_npart_local());
  d_cell_counts_INT->realloc_no_copy(cell_count_all_partitions);

  int *k_cell_counts = d_cell_counts->ptr;
  int *k_particle_layers = d_particle_layers->ptr;
  INT *k_cell_counts_INT = d_cell_counts_INT->ptr;

  sycl_target->queue
      .fill(k_cell_counts, static_cast<int>(0), cell_count_all_partitions)
      .wait_and_throw();

  // Assemble the global particle maps.
  std::vector<int *> tmp = {k_particle_layers, k_cell_counts};
  this->map_ptrs->set(tmp);
  this->loop_0->execute();
  EventStack es;

  for (std::size_t px = 0; px < this->num_partitions; px++) {
    if (still_exists[px]) {
      int *k_cell_counts_int = this->sub_group_particle_maps.at(px)
                                   .lock()
                                   ->dh_npart_cell->d_buffer.ptr;
      INT *k_cell_counts_es = this->sub_group_particle_maps.at(px)
                                  .lock()
                                  ->dh_npart_cell_es->d_buffer.ptr;
      es.push(sycl_target->queue.parallel_for(
          sycl::range<1>(cell_count), [=](auto idx) {
            const int cell_count_inner = k_cell_counts[cell_count * px + idx];
            k_cell_counts_int[idx] = cell_count_inner;
            k_cell_counts_INT[cell_count * px + idx] =
                static_cast<INT>(cell_count_inner);
            k_cell_counts_es[idx] = 0;
          }));
    }
  }
  es.wait();

  for (std::size_t px = 0; px < this->num_partitions; px++) {
    if (still_exists[px]) {
      INT *k_cell_counts_es = this->sub_group_particle_maps.at(px)
                                  .lock()
                                  ->dh_npart_cell_es->d_buffer.ptr;
      es.push(joint_exclusive_scan(sycl_target, cell_count,
                                   k_cell_counts_INT + px * cell_count,
                                   k_cell_counts_es));
    }
  }
  for (std::size_t px = 0; px < this->num_partitions; px++) {
    if (still_exists[px]) {
      this->sub_group_particle_maps.at(px)
          .lock()
          ->dh_npart_cell->device_to_host();
    }
  }
  es.wait();

  for (std::size_t px = 0; px < this->num_partitions; px++) {
    if (still_exists[px]) {
      this->sub_group_particle_maps.at(px)
          .lock()
          ->dh_npart_cell_es->device_to_host();
    }
  }

  // Create the particle map for each partition.
  for (std::size_t px = 0; px < this->num_partitions; px++) {
    if (still_exists[px]) {
      int *h_cell_counts = this->sub_group_particle_maps.at(px)
                               .lock()
                               ->dh_npart_cell->h_buffer.ptr;
      INT *h_cell_counts_es = this->sub_group_particle_maps.at(px)
                                  .lock()
                                  ->dh_npart_cell_es->h_buffer.ptr;
      this->sub_group_particle_maps.at(px).lock()->create(
          0, cell_count, h_cell_counts, h_cell_counts_es);
    }
  }

  // Populate the particle maps from the particles
  std::vector<INT *> h_layer_maps(this->num_partitions);
  std::vector<INT *> h_npart_cell_es(this->num_partitions);
  for (std::size_t px = 0; px < this->num_partitions; px++) {
    if (still_exists[px]) {
      h_layer_maps[px] =
          this->sub_group_particle_maps.at(px).lock()->d_layer_map->ptr;
      h_npart_cell_es[px] = this->sub_group_particle_maps.at(px)
                                .lock()
                                ->dh_npart_cell_es->d_buffer.ptr;
    } else {
      h_layer_maps[px] = nullptr;
      h_npart_cell_es[px] = nullptr;
    }
  }
  this->la_layer_maps->set(h_layer_maps);
  this->la_npart_cell_es->set(h_npart_cell_es);
  this->loop_1->submit();

  // Create the selections from the particle maps.
  for (std::size_t px = 0; px < this->num_partitions; px++) {

    if (still_exists[px]) {
      auto [h_npart_cell_ptr, d_npart_cell_ptr, h_npart_cell_es_ptr,
            d_npart_cell_es_ptr] =
          this->sub_group_particle_maps.at(px).lock()->get_helper_ptrs();

      INT npart_total =
          this->sub_group_particle_maps.at(px).lock()->npart_total;
      auto d_cell_starts_ptr =
          this->sub_group_particle_maps.at(px).lock()->d_cell_starts->ptr;

      Selection s;
      s.npart_local = npart_total;
      s.ncell = cell_count;
      s.h_npart_cell = h_npart_cell_ptr;
      s.d_npart_cell = d_npart_cell_ptr;
      s.d_npart_cell_es = d_npart_cell_es_ptr;
      s.d_map_cells_to_particles = {d_cell_starts_ptr};
      this->partition_selections.at(px) = s;
    }
  }

  this->loop_1->wait();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_particle_layers);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_cell_counts_INT);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_cell_counts);
}

template ParticleGroupPartitioner::ParticleGroupPartitioner(
    std::shared_ptr<ParticleGroup> parent, Sym<INT> partition_sym,
    const std::size_t num_partitions,
    std::vector<std::shared_ptr<ParticleGroupPartitionSelector>>
        &partition_selectors_in);

template ParticleGroupPartitioner::ParticleGroupPartitioner(
    std::shared_ptr<ParticleSubGroup> parent, Sym<INT> partition_sym,
    const std::size_t num_partitions,
    std::vector<std::shared_ptr<ParticleGroupPartitionSelector>>
        &partition_selectors_in);

} // namespace ParticleSubGroupImplementation

template std::vector<ParticleSubGroupSharedPtr>
particle_group_partition(std::shared_ptr<ParticleGroup> parent,
                         Sym<INT> partition_sym, const int num_partitions);

template std::vector<ParticleSubGroupSharedPtr>
particle_group_partition(std::shared_ptr<ParticleSubGroup> parent,
                         Sym<INT> partition_sym, const int num_partitions);

} // namespace NESO::Particles
