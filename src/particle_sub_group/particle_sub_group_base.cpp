#include <neso_particles/common_impl.hpp>
#include <neso_particles/particle_sub_group/copy_selector.hpp>
#include <neso_particles/particle_sub_group/particle_sub_group_base.hpp>

namespace NESO::Particles {

int ParticleSubGroup::get_cells_layers(std::vector<INT> &cells,
                                       std::vector<INT> &layers) {
  this->create_if_required();
  cells.resize(this->npart_local);
  layers.resize(this->npart_local);
  const int cell_count = this->particle_group->domain->mesh->get_cell_count();

  auto map_cells_to_particles = get_host_map_cells_to_particles(
      this->particle_group->sycl_target, this->selection);

  INT index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int nrow = map_cells_to_particles.at(cellx).size();
    for (int rowx = 0; rowx < nrow; rowx++) {
      cells[index] = cellx;
      const int layerx = map_cells_to_particles.at(cellx).at(rowx);
      layers[index] = layerx;
      index++;
    }
  }
  return npart_local;
}

void ParticleSubGroup::prepare_ephemeral_dats() { this->create_if_required(); }

bool ParticleSubGroup::invalidate_ephemeral_dats_if_required() {

  bool bool_dats = false;
  bool bool_group = false;
  this->selector->update_required(&bool_dats, &bool_group);
  const bool required =
      (this->is_static) ? bool_group : bool_group || bool_dats;

  if (required) {
    this->reset_ephemeral_dats(
        this->selection.npart_local, this->selection.h_npart_cell,
        this->selection.d_npart_cell, this->selection.d_npart_cell_es);
  }
  return required;
}

bool ParticleSubGroup::create_inner() {
  const bool was_updated = this->selector->get(&this->selection);
  this->npart_local = this->selection.npart_local;

  if (was_updated) {
    this->reset_ephemeral_dats(
        this->selection.npart_local, this->selection.h_npart_cell,
        this->selection.d_npart_cell, this->selection.d_npart_cell_es);
  }

  return was_updated;
}

void ParticleSubGroup::check_selector(
    ParticleSubGroupImplementation::SubGroupSelectorBaseSharedPtr selector) {

  NESOASSERT(!selector->consumed,
             "Attempting to create a ParticleSubGroup from a Selector that "
             "has already been used to make another ParticleSubGroup.");
  selector->consumed = true;
}

ParticleSubGroup::ParticleSubGroup(ParticleGroupSharedPtr particle_group)
    : ParticleSubGroup(std::dynamic_pointer_cast<
                       ParticleSubGroupImplementation::SubGroupSelectorBase>(
          std::make_shared<
              ParticleSubGroupImplementation::SubGroupSelectorWholeGroup>(
              particle_group))) {}

ParticleSubGroup::ParticleSubGroup(
    std::shared_ptr<ParticleSubGroup> particle_sub_group)
    : ParticleSubGroup(std::dynamic_pointer_cast<
                       ParticleSubGroupImplementation::SubGroupSelectorBase>(
          std::make_shared<ParticleSubGroupImplementation::CopySelector>(
              particle_sub_group))) {}

ParticleSubGroup::ParticleSubGroup(
    ParticleSubGroupImplementation::SubGroupSelectorSharedPtr selector)
    : EphemeralDats(selector->particle_group->sycl_target,
                    selector->particle_group->domain->mesh->get_cell_count(),
                    &selector->particle_group->particle_dats_int,
                    &selector->particle_group->particle_dats_real),
      is_static(false), particle_group(selector->particle_group),
      selector(selector),
      is_whole_particle_group(selector->is_whole_particle_group) {
  this->check_selector(
      std::dynamic_pointer_cast<
          ParticleSubGroupImplementation::SubGroupSelectorBase>(selector));
}

ParticleSubGroup::ParticleSubGroup(
    ParticleSubGroupImplementation::SubGroupSelectorBaseSharedPtr selector)
    : EphemeralDats(selector->particle_group->sycl_target,
                    selector->particle_group->domain->mesh->get_cell_count(),
                    &selector->particle_group->particle_dats_int,
                    &selector->particle_group->particle_dats_real),
      is_static(false), particle_group(selector->particle_group),
      selector(selector),
      is_whole_particle_group(selector->is_whole_particle_group) {
  this->check_selector(selector);
}

bool ParticleSubGroup::static_status(const std::optional<bool> status) {
  if (status != std::nullopt) {
    // If the two static values are the same then nothing has to change.
    const bool new_is_static = status.value();
    if (!(this->is_static == new_is_static)) {
      if (new_is_static) {
        // Create the sub group before we disable creating the sub group.
        this->create_if_required();
      }
      this->is_static = new_is_static;
    }
  }
  return this->is_static;
}

void ParticleSubGroup::create() { this->create_inner(); }

bool ParticleSubGroup::is_valid() {
  if (this->is_static) {
    return !this->particle_group->check_validation(
        this->selector->particle_group_version, false);
  }
  return true;
}

bool ParticleSubGroup::create_if_required() {
  NESOASSERT(this->is_valid(), "This ParticleSubGroup has been invalidated.");
  if (this->is_static) {
    return false;
  } else {
    return this->create_inner();
  }
}

INT ParticleSubGroup::get_npart_local() {
  this->create_if_required();
  return this->npart_local;
}

INT ParticleSubGroup::get_npart_cell(const int cell) {
  this->create_if_required();
  return this->selection.h_npart_cell[cell];
}

ParticleGroupSharedPtr ParticleSubGroup::get_particle_group() {
  return this->particle_group;
}

bool ParticleSubGroup::is_entire_particle_group() {
  return this->is_whole_particle_group;
}

ParticleSetSharedPtr ParticleSubGroup::get_particles(std::vector<INT> &cells,
                                                     std::vector<INT> &layers) {
  if (this->is_whole_particle_group) {
    return this->particle_group->get_particles(cells, layers);
  } else {
    auto r0 = this->particle_group->sycl_target->profile_map.start_region(
        "ParticleSubGroup", "get_particles");
    this->create_if_required();
    NESOASSERT(cells.size() == layers.size(),
               "Cells and layers vectors have different sizes.");
    const std::size_t num_particles = cells.size();

    if (num_particles > 0) {

      auto sycl_target = this->particle_group->sycl_target;
      auto tmp_buffer =
          get_resource<BufferDeviceHost<INT>,
                       ResourceStackInterfaceBufferDeviceHost<INT>>(
              sycl_target->resource_stack_map,
              ResourceStackKeyBufferDeviceHost<INT>{}, sycl_target);

      tmp_buffer->realloc_no_copy(num_particles * 3);

      INT *d_cells = tmp_buffer->d_buffer.ptr;
      INT *d_layers = d_cells + num_particles;
      INT *d_inner_layers = d_layers + num_particles;

      EventStack es;
      es.push(sycl_target->queue.memcpy(d_cells, cells.data(),
                                        num_particles * sizeof(INT)));
      es.push(sycl_target->queue.memcpy(d_layers, layers.data(),
                                        num_particles * sizeof(INT)));

      const INT num_cells =
          this->particle_group->domain->mesh->get_cell_count();
      for (std::size_t px = 0; px < num_particles; px++) {
        const INT cellx = cells.at(px);
        NESOASSERT((cellx > -1) && (cellx < num_cells),
                   "Cell index not in range.");
        const INT layerx = layers.at(px);
        NESOASSERT((layerx > -1) && (layerx < this->get_npart_cell(cellx)),
                   "Layer index not in range.");
      }

      es.wait();

      auto k_map_cells_to_particles = this->selection.d_map_cells_to_particles;

      sycl_target->queue
          .parallel_for<>(
              sycl::range<1>(num_particles),
              [=](sycl::id<1> idx) {
                const INT cell = d_cells[idx];
                const INT layer = d_layers[idx];
                const INT inner_layer =
                    k_map_cells_to_particles.map_loop_layer_to_layer(cell,
                                                                     layer);
                d_inner_layers[idx] = inner_layer;
              })
          .wait_and_throw();

      auto ps = this->particle_group->get_particles(num_particles, d_cells,
                                                    d_inner_layers);
      restore_resource(sycl_target->resource_stack_map,
                       ResourceStackKeyBufferDeviceHost<INT>{}, tmp_buffer);

      this->particle_group->sycl_target->profile_map.end_region(r0);
      return ps;
    } else {
      this->particle_group->sycl_target->profile_map.end_region(r0);
      return std::make_shared<ParticleSet>(
          0, this->particle_group->get_particle_spec());
    }
  }
}

const ParticleSubGroupImplementation::Selection &
ParticleSubGroup::get_selection() const {
  return this->selection;
}

void ParticleSubGroup::get_cells_layers(INT *d_cells, INT *d_layers) {

  auto lambda_loop = [&](auto iteration_set) {
    particle_loop(
        iteration_set,
        [=](auto index) {
          const INT px = index.get_loop_linear_index();
          d_cells[px] = index.cell;
          d_layers[px] = index.layer;
        },
        Access::read(ParticleLoopIndex{}))
        ->execute();
  };

  if (this->is_entire_particle_group()) {
    lambda_loop(this->particle_group);
  } else {
    lambda_loop(std::shared_ptr<ParticleSubGroup>(
        this, []([[maybe_unused]] auto x) {}));
  }
}
} // namespace NESO::Particles
