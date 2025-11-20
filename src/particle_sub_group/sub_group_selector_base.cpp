#include <neso_particles/particle_sub_group/particle_sub_group.hpp>
#include <neso_particles/particle_sub_group/sub_group_selector_base.hpp>

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

std::vector<std::vector<INT>>
get_host_map_cells_to_particles(SYCLTargetSharedPtr sycl_target,
                                const Selection &selection) {
  const int cell_count = selection.ncell;
  std::vector<std::vector<INT>> return_map(cell_count);
  std::vector<INT *> d_map_ptrs(cell_count);

  sycl_target->queue
      .memcpy(d_map_ptrs.data(), selection.d_map_cells_to_particles.map_ptr,
              cell_count * sizeof(INT *))
      .wait_and_throw();

  EventStack es;
  for (int cx = 0; cx < cell_count; cx++) {
    return_map.at(cx) = std::vector<INT>(selection.h_npart_cell[cx]);
    if (selection.h_npart_cell[cx] > 0) {
      es.push(
          sycl_target->queue.memcpy(return_map.at(cx).data(), d_map_ptrs.at(cx),
                                    selection.h_npart_cell[cx] * sizeof(INT)));
    }
  }
  es.wait();
  return return_map;
}

void SubGroupSelectorBase::add_sym_dependency(Sym<INT> sym) {
  this->particle_dat_versions[sym] = 0;
}
void SubGroupSelectorBase::add_sym_dependency(Sym<REAL> sym) {
  this->particle_dat_versions[sym] = 0;
}

void SubGroupSelectorBase::printing_create_outer_start() {
  if (this->particle_group->debug_sub_group_create) {
    if (!this->particle_group->debug_sub_group_indent) {
      std::cout << std::string(80, '-') << std::endl;
      std::cout << "Testing recreation criterion: " << (void *)this
                << std::endl;
    }
    this->particle_group->debug_sub_group_indent += 4;
  }
}

void SubGroupSelectorBase::printing_create_outer_end() {
  if (this->particle_group->debug_sub_group_create) {
    this->particle_group->debug_sub_group_indent -= 4;
  }
}

void SubGroupSelectorBase::printing_create_inner_start(const bool bool_dats,
                                                       const bool bool_group) {
  if (this->particle_group->debug_sub_group_create) {

    std::string indent(this->particle_group->debug_sub_group_indent, ' ');
    std::cout << indent << "Recreating Selector: " << (void *)this
              << " reason_dats: " << bool_dats
              << " reason_group: " << bool_group << std::endl;
  }
}

void SubGroupSelectorBase::printing_create_inner_end() {
  if (this->particle_group->debug_sub_group_create) {
    if (!this->particle_group->debug_sub_group_indent) {
      std::cout << std::string(80, '-') << std::endl;
    }
  }
}

bool SubGroupSelectorBase::get(Selection *selection) {

  this->printing_create_outer_start();

  const bool bool_dats =
      this->particle_group->check_validation(this->particle_dat_versions);
  const bool bool_group =
      this->particle_group->check_validation(this->particle_group_version);

  if (bool_dats || bool_group) {
    this->printing_create_inner_start(bool_dats, bool_group);
    this->create(selection);
    this->printing_create_inner_end();
    this->printing_create_outer_end();
    return true;
  }

  this->printing_create_outer_end();
  return false;
}

bool SubGroupSelectorBase::update_required(bool *bool_dats, bool *bool_group) {

  *bool_dats = this->particle_group->check_validation(
      this->particle_dat_versions, false);
  *bool_group = this->particle_group->check_validation(
      this->particle_group_version, false);

  return (*bool_dats) || (*bool_group);
}
SubGroupSelectorBase::SubGroupSelectorBase(
    std::shared_ptr<ParticleGroup> parent)
    : particle_group(get_particle_group(parent)), particle_sub_group(nullptr),
      particle_group_version(0) {
  this->add_parent_dependencies(parent);

  NESOASSERT(this->sub_group_selector_resource == nullptr,
             "Sub-group resource is already allocated somehow.");
  this->sub_group_selector_resource =
      this->particle_group->resource_stack_sub_group_resource->get();
  this->map_ptrs = this->sub_group_selector_resource->map_ptrs;
  this->map_cell_to_particles_ptrs =
      this->sub_group_selector_resource->map_cell_to_particles_ptrs;
  this->sub_group_particle_map =
      this->sub_group_selector_resource->sub_group_particle_map;
}

SubGroupSelectorBase::SubGroupSelectorBase(
    std::shared_ptr<ParticleSubGroup> parent)
    : particle_group(get_particle_group(parent)), particle_sub_group(parent),
      particle_group_version(0) {
  this->add_parent_dependencies(parent);

  NESOASSERT(this->sub_group_selector_resource == nullptr,
             "Sub-group resource is already allocated somehow.");
  this->sub_group_selector_resource =
      this->particle_group->resource_stack_sub_group_resource->get();
  this->map_ptrs = this->sub_group_selector_resource->map_ptrs;
  this->map_cell_to_particles_ptrs =
      this->sub_group_selector_resource->map_cell_to_particles_ptrs;
  this->sub_group_particle_map =
      this->sub_group_selector_resource->sub_group_particle_map;
}

void SubGroupSelectorBase::add_parent_dependencies(
    std::shared_ptr<ParticleSubGroupImplementation::SubGroupSelectorBase>
        selector) {
  for (const auto &dep : selector->particle_dat_versions) {
    this->particle_dat_versions[dep.first] = 0;
  }
}

void SubGroupSelectorBase::add_parent_dependencies(
    std::shared_ptr<ParticleSubGroup> parent) {
  if (parent != nullptr) {
    this->add_parent_dependencies(parent->selector);
  }
}

} // namespace ParticleSubGroupImplementation

bool is_whole_group(ParticleGroupSharedPtr) { return true; }
bool is_whole_group(std::shared_ptr<ParticleSubGroup> parent) {
  return parent->is_entire_particle_group();
}

} // namespace NESO::Particles
