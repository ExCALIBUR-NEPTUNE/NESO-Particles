#include <neso_particles/particle_sub_group/disjoint_union_sub_group_selector.hpp>
#include <neso_particles/particle_sub_group/particle_sub_group.hpp>

namespace NESO::Particles {

namespace ParticleSubGroupImplementation {

DisjointUnionSubGroupSelector::DisjointUnionSubGroupSelector(
    std::vector<std::shared_ptr<ParticleSubGroup>> &parents)
    : SubGroupSelector(parents.at(0)), parents(parents) {
  this->particle_group = get_particle_group(parents.at(0));

  for (auto &px : parents) {
    NESOASSERT(
        !is_whole_group(px),
        "This selector assumes that the ParticleSubGroups are not whole "
        "particle groups. Use the helper function particle_sub_group_union.");
    NESOASSERT(get_particle_group(px) == this->particle_group,
               "Source ParticleSubGroups have different base ParticleGroup.");
    this->add_parent_dependencies(px);
  }
}

void DisjointUnionSubGroupSelector::create(Selection *created_selection) {

  const int cell_count = this->particle_group->domain->mesh->get_cell_count();
  const int sub_group_count = this->parents.size();

  auto sycl_target = this->particle_group->sycl_target;

  auto [h_npart_cell_ptr, d_npart_cell_ptr, h_npart_cell_es_ptr,
        d_npart_cell_es_ptr] = this->sub_group_particle_map->get_helper_ptrs();

  auto k_npart_cell = d_npart_cell_ptr;
  auto k_npart_cell_es = d_npart_cell_es_ptr;

  auto pr0 = sycl_target->profile_map.start_region(
      "DisjointUnionSubGroupSelector", "create");

  std::vector<Selection> h_selections(sub_group_count);
  int npart_local_test = 0;
  for (int px = 0; px < sub_group_count; px++) {
    this->parents[px]->create_if_required();
    npart_local_test += this->parents[px]->get_npart_local();
    h_selections[px] = this->parents[px]->get_selection();
  }

  auto d_selections =
      get_resource<BufferDevice<Selection>,
                   ResourceStackInterfaceBufferDevice<Selection>>(
          sycl_target->resource_stack_map,
          ResourceStackKeyBufferDevice<Selection>{}, sycl_target);
  d_selections->realloc_no_copy(sub_group_count);
  auto k_selections = d_selections->ptr;

  auto e0 = sycl_target->queue.memcpy(k_selections, h_selections.data(),
                                      sub_group_count * sizeof(Selection));

  auto e1 = sycl_target->queue.parallel_for(
      sycl::range<1>(cell_count), e0, [=](auto idx) {
        int new_occupancy = 0;
        INT new_es = 0;
        for (int px = 0; px < sub_group_count; px++) {
          const int npart_cell = k_selections[px].d_npart_cell[idx];
          new_occupancy += npart_cell;
          new_es += k_selections[px].d_npart_cell_es[idx];
        }
        k_npart_cell[idx] = new_occupancy;
        k_npart_cell_es[idx] = new_es;
      });

  auto e2 = sycl_target->queue.memcpy(h_npart_cell_ptr, k_npart_cell,
                                      cell_count * sizeof(int), e1);
  auto e3 = sycl_target->queue.memcpy(h_npart_cell_es_ptr, k_npart_cell_es,
                                      cell_count * sizeof(INT), e1);
  e2.wait_and_throw();
  e3.wait_and_throw();

  this->sub_group_particle_map->create(0, cell_count, h_npart_cell_ptr,
                                       h_npart_cell_es_ptr);
  const auto k_cell_starts = this->sub_group_particle_map->d_cell_starts->ptr;

  std::size_t local_size =
      sycl_target->parameters->get<SizeTParameter>("LOOP_LOCAL_SIZE")->value;

  sycl_target->queue
      .parallel_for(
          sycl::nd_range<2>(sycl::range<2>(cell_count, local_size),
                            sycl::range<2>(1, local_size)),
          [=](sycl::nd_item<2> idx) {
            const std::size_t index_cell = idx.get_global_id(0);
            const std::size_t index = idx.get_global_id(1);

            std::size_t offset = 0;
            for (int px = 0; px < sub_group_count; px++) {
              const std::size_t npart =
                  k_selections[px].d_npart_cell[index_cell];
              for (std::size_t ix = index; ix < npart; ix += local_size) {
                k_cell_starts[index_cell][offset + ix] =
                    k_selections[px]
                        .d_map_cells_to_particles.map_loop_layer_to_layer(
                            index_cell, ix);
              }
              offset += npart;
            }
          })
      .wait_and_throw();

  const int npart_local =
      static_cast<int>(h_npart_cell_es_ptr[cell_count - 1]) +
      h_npart_cell_ptr[cell_count - 1];

  NESOASSERT(npart_local == npart_local_test,
             "Consistency check failed for particle count.");

  created_selection->npart_local = npart_local;
  created_selection->ncell = cell_count;
  created_selection->h_npart_cell = h_npart_cell_ptr;
  created_selection->d_npart_cell = d_npart_cell_ptr;
  created_selection->d_npart_cell_es = d_npart_cell_es_ptr;
  created_selection->d_map_cells_to_particles = {
      this->sub_group_particle_map->d_cell_starts->ptr};

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<Selection>{}, d_selections);
  sycl_target->profile_map.end_region(pr0);
}

} // namespace ParticleSubGroupImplementation

std::shared_ptr<ParticleSubGroup> particle_sub_group_disjoint_union(
    std::vector<std::shared_ptr<ParticleSubGroup>> &parents,
    const bool make_static) {

  NESOASSERT(parents.size(), "No sub-groups passed.");
  ParticleSubGroupSharedPtr sub_group = nullptr;

  if (parents.size() == 1) {
    // If there is only one sub group use the copy selector.
    sub_group = particle_sub_group(parents[0]);
  } else {

    for (auto &px : parents) {
      NESOASSERT((!is_whole_group(px)),
                 "Source ParticleSubGroups which are references to the whole "
                 "ParticleGroup are not currently supported unless only one is "
                 "passed, i.e. parents.size() == 1.");
    }

    auto s = std::make_shared<
        ParticleSubGroupImplementation::DisjointUnionSubGroupSelector>(parents);
    sub_group = std::make_shared<ParticleSubGroup>(
        std::dynamic_pointer_cast<
            ParticleSubGroupImplementation::SubGroupSelector>(s));
  }

  sub_group->static_status(make_static);
  return sub_group;
}
} // namespace NESO::Particles
