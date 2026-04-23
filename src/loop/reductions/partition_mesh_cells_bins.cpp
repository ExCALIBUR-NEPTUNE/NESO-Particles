#include <neso_particles/loop/reductions/partition_mesh_cells_bins.hpp>
#include <neso_particles/particle_sub_group/particle_sub_group.hpp>

namespace NESO::Particles {

namespace {

template <typename GROUP_TYPE>
inline void
partition_mesh_cells_bins_inner(std::shared_ptr<GROUP_TYPE> particle_sub_group,
                                const int num_bins, Sym<INT> bin_sym,
                                const int bin_component,
                                IndexMapSharedPtr<2, 1> index_map) {

  auto particle_group = get_particle_group(particle_sub_group);
  NESOASSERT(index_map->sycl_target.get() == particle_group->sycl_target.get(),
             "Compute device missmatch.");

  auto sycl_target = particle_group->sycl_target;

  auto r0 = sycl_target->profile_map.start_region("partition_mesh_cells_bins",
                                                  "inner");
  int key_strides[2] = {particle_group->domain->mesh->get_cell_count(),
                        num_bins};

  index_map->set_key_strides(key_strides);

  auto d_layer_buffer =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);

  d_layer_buffer->realloc_no_copy(particle_sub_group->get_npart_local());
  int *k_layer_buffer = d_layer_buffer->ptr;

  auto d_counts = index_map->get_tmp_buffer_num_values();
  INT *k_counts = d_counts->ptr;
  auto d_index_map = index_map->get_device();

  particle_loop(
      particle_sub_group,
      [=](auto INDEX, auto BIN) {
        const int bin_index = static_cast<int>(BIN.at(bin_component));
        if ((-1 < bin_index) && (bin_index < num_bins)) {
          const int key[2] = {static_cast<int>(INDEX.cell), bin_index};
          const INT index = d_index_map.get_linear_index(key);
          const int layer =
              static_cast<int>(atomic_fetch_add(k_counts + index, (INT)1));
          k_layer_buffer[INDEX.get_loop_linear_index()] = layer;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(bin_sym))
      ->execute();

  index_map->populate_offsets_buffer(k_counts);
  d_index_map = index_map->get_device();

  particle_loop(
      particle_sub_group,
      [=](auto INDEX, auto BIN) {
        const int bin_index = static_cast<int>(BIN.at(bin_component));
        if ((-1 < bin_index) && (bin_index < num_bins)) {
          const int key[2] = {static_cast<int>(INDEX.cell), bin_index};
          const int layer = k_layer_buffer[INDEX.get_loop_linear_index()];
          d_index_map.at(key, layer, 0) = static_cast<int>(INDEX.layer);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(bin_sym))
      ->execute();

  index_map->restore_tmp_buffer_num_values(d_counts);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_layer_buffer);

  sycl_target->profile_map.end_region(r0);
}
} // namespace

void partition_mesh_cells_bins(ParticleGroupSharedPtr particle_group,
                               const int num_bins, Sym<INT> bin_sym,
                               const int bin_component,
                               IndexMapSharedPtr<2, 1> index_map) {
  partition_mesh_cells_bins_inner(particle_group, num_bins, bin_sym,
                                  bin_component, index_map);
}

void partition_mesh_cells_bins(ParticleSubGroupSharedPtr particle_sub_group,
                               const int num_bins, Sym<INT> bin_sym,
                               const int bin_component,
                               IndexMapSharedPtr<2, 1> index_map) {
  partition_mesh_cells_bins_inner(particle_sub_group, num_bins, bin_sym,
                                  bin_component, index_map);
}

} // namespace NESO::Particles
