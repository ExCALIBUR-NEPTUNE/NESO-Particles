#include <neso_particles/common_impl.hpp>
#include <neso_particles/particle_sub_group/sub_group_selector.hpp>

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

SubGroupSelector::SubGroupSelector(std::shared_ptr<ParticleGroup> parent)
    : SubGroupSelectorBase(parent) {
  this->loop_1 = particle_loop(
      "sub_group_selector_1", parent,
      [=](auto loop_index, auto k_map_cell_to_particles, auto k_map_ptrs) {
        INT **base_map_cell_to_particles = k_map_cell_to_particles.at(0);
        const INT particle_linear_index = loop_index.get_local_linear_index();
        const int layer = k_map_ptrs.at(0)[particle_linear_index];
        const bool required = layer > -1;
        if (required) {
          INT *base_map_for_cell = base_map_cell_to_particles[loop_index.cell];
          base_map_for_cell[layer] = loop_index.layer;
        }
      },
      Access::read(ParticleLoopIndex{}),
      Access::write(this->map_cell_to_particles_ptrs),
      Access::read(this->map_ptrs));
}

SubGroupSelector::SubGroupSelector(std::shared_ptr<ParticleSubGroup> parent)
    : SubGroupSelectorBase(parent) {
  this->loop_1 = particle_loop(
      "sub_group_selector_1", parent,
      [=](auto loop_index, auto k_map_cell_to_particles, auto k_map_ptrs) {
        INT **base_map_cell_to_particles = k_map_cell_to_particles.at(0);
        const INT particle_linear_index = loop_index.get_local_linear_index();
        const int layer = k_map_ptrs.at(0)[particle_linear_index];
        const bool required = layer > -1;
        if (required) {
          INT *base_map_for_cell = base_map_cell_to_particles[loop_index.cell];
          base_map_for_cell[layer] = loop_index.layer;
        }
      },
      Access::read(ParticleLoopIndex{}),
      Access::write(this->map_cell_to_particles_ptrs),
      Access::read(this->map_ptrs));
}

void SubGroupSelector::create(Selection *created_selection) {
  const int cell_count = this->particle_group->domain->mesh->get_cell_count();
  auto sycl_target = this->particle_group->sycl_target;

  auto pg_map_layers =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);

  auto [h_npart_cell_ptr, d_npart_cell_ptr, h_npart_cell_es_ptr,
        d_npart_cell_es_ptr] = this->sub_group_particle_map->get_helper_ptrs();

  auto e0 = sycl_target->queue.fill<int>(d_npart_cell_ptr, 0, cell_count);
  const auto npart_local = this->particle_group->get_npart_local();
  pg_map_layers->realloc_no_copy(npart_local);

  std::vector<int *> tmp = {pg_map_layers->ptr, d_npart_cell_ptr};
  this->map_ptrs->set(tmp);
  e0.wait_and_throw();

  this->loop_0->execute();
  sycl_target->queue
      .memcpy(h_npart_cell_ptr, d_npart_cell_ptr, cell_count * sizeof(int))
      .wait_and_throw();

  INT running_total = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const INT npart_cell = h_npart_cell_ptr[cellx];
    h_npart_cell_es_ptr[cellx] = running_total;
    running_total += npart_cell;
  }

  this->sub_group_particle_map->create(0, cell_count, h_npart_cell_ptr,
                                       h_npart_cell_es_ptr);
  auto d_cell_starts_ptr = this->sub_group_particle_map->d_cell_starts->ptr;

  this->map_cell_to_particles_ptrs->set({d_cell_starts_ptr});
  this->loop_1->submit();

  sycl_target->queue
      .memcpy(d_npart_cell_es_ptr, h_npart_cell_es_ptr,
              cell_count * sizeof(INT))
      .wait_and_throw();

  this->loop_1->wait();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, pg_map_layers);

  Selection s;
  s.npart_local = running_total;
  s.ncell = cell_count;
  s.h_npart_cell = h_npart_cell_ptr;
  s.d_npart_cell = d_npart_cell_ptr;
  s.d_npart_cell_es = d_npart_cell_es_ptr;
  s.d_map_cells_to_particles = {d_cell_starts_ptr};
  *created_selection = s;
}

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles
