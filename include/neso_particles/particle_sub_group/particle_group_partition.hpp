#ifndef _NESO_PARTICLES_PARTICLE_SUB_GROUP_PARTICLE_GROUP_PARTITION_HPP_
#define _NESO_PARTICLES_PARTICLE_SUB_GROUP_PARTICLE_GROUP_PARTITION_HPP_

#include "../device_functions.hpp"
#include "../loop/particle_loop.hpp"
#include "../loop/particle_loop_base.hpp"
#include "particle_sub_group_base.hpp"
#include "sub_group_selector.hpp"

namespace NESO::Particles::ParticleSubGroupImplementation {

class ParticleGroupPartitioner;

/**
 * This is the Selector type which is actually used to create the
 * ParticleSubGroups. Instead of creating a Selection this implementation
 * extracts the corresponding Selection from the parent orchestrating
 * implementation.
 */
class ParticleGroupPartitionSelector
    : public ParticleSubGroupImplementation::SubGroupSelectorBase {

  friend class ParticleGroupPartitioner;

protected:
  std::function<void(Selection *created_selection)> create_handle;
  virtual inline void create(Selection *created_selection) override {
    this->create_handle(created_selection);
  }

  inline std::shared_ptr<SubGroupParticleMap> get_sub_group_particle_map() {
    return this->sub_group_particle_map;
  }

public:
  template <typename PARENT>
  ParticleGroupPartitionSelector(
      std::shared_ptr<PARENT> parent, Sym<INT> partition_sym,
      std::function<void(Selection *created_selection)> create_handle)
      : SubGroupSelectorBase(parent), create_handle(create_handle) {
    this->add_sym_dependency(partition_sym);
  }
};

/**
 * This is a Selector which partitions the parent into num_partitions
 * partitions. It creates the individual selections which are extracted by
 * ParticleGroupPartitionSelector. Users should create an instance of
 * ParticleGroupPartition to use this class.
 */
class ParticleGroupPartitioner
    : public ParticleSubGroupImplementation::SubGroupSelectorBase {
protected:
  std::vector<std::shared_ptr<ParticleGroupPartitionSelector>>
      partition_selectors;

  std::vector<Selection> partition_selections;
  std::vector<std::shared_ptr<SubGroupParticleMap>> sub_group_particle_maps;

  std::shared_ptr<LocalArray<INT *>> la_layer_maps;
  std::shared_ptr<LocalArray<INT *>> la_npart_cell_es;

  ParticleLoopSharedPtr loop_0, loop_1;

  inline void create_indexed(const std::size_t index,
                             Selection *created_selection) {
    NESOASSERT(index < this->num_partitions, "Bad index passed.");

    // This is where the versions get checked and the internal representation
    // for all the partitions gets updated.
    this->get(nullptr);
    *created_selection = this->partition_selections.at(index);
  }

  /**
   * This is the overrided method which is called by get to create all the
   * partitions. The pointer passed should be nullptr as the corresponding class
   * ParticleGroupPartitionSelector will extract its selection from the vector
   * of selections.
   */
  virtual inline void create(Selection *created_selection) override {
    if (this->num_partitions == 0) {
      return;
    }

    NESOASSERT(created_selection == nullptr, "Expected a nullptr.");
    auto sycl_target = this->particle_group->sycl_target;
    const std::size_t cell_count = static_cast<std::size_t>(
        this->particle_group->domain->mesh->get_cell_count());

    // Create overall map.
    auto d_cell_counts = get_resource<BufferDevice<int>,
                                      ResourceStackInterfaceBufferDevice<int>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
        sycl_target);

    auto d_cell_counts_INT =
        get_resource<BufferDevice<INT>,
                     ResourceStackInterfaceBufferDevice<INT>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<INT>{}, sycl_target);
    auto d_particle_layers =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);

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
      int *k_cell_counts_int =
          this->sub_group_particle_maps.at(px)->dh_npart_cell->d_buffer.ptr;
      INT *k_cell_counts_es =
          this->sub_group_particle_maps.at(px)->dh_npart_cell_es->d_buffer.ptr;
      es.push(sycl_target->queue.parallel_for(
          sycl::range<1>(cell_count), [=](auto idx) {
            const int cell_count_inner = k_cell_counts[cell_count * px + idx];
            k_cell_counts_int[idx] = cell_count_inner;
            k_cell_counts_INT[cell_count * px + idx] =
                static_cast<INT>(cell_count_inner);
            k_cell_counts_es[idx] = 0;
          }));
    }
    es.wait();

    for (std::size_t px = 0; px < this->num_partitions; px++) {
      INT *k_cell_counts_es =
          this->sub_group_particle_maps.at(px)->dh_npart_cell_es->d_buffer.ptr;
      es.push(joint_exclusive_scan(sycl_target, cell_count,
                                   k_cell_counts_INT + px * cell_count,
                                   k_cell_counts_es));
    }
    for (std::size_t px = 0; px < this->num_partitions; px++) {
      this->sub_group_particle_maps.at(px)->dh_npart_cell->device_to_host();
    }
    es.wait();

    for (std::size_t px = 0; px < this->num_partitions; px++) {
      this->sub_group_particle_maps.at(px)->dh_npart_cell_es->device_to_host();
    }

    // Create the particle map for each partition.
    for (std::size_t px = 0; px < this->num_partitions; px++) {
      int *h_cell_counts =
          this->sub_group_particle_maps.at(px)->dh_npart_cell->h_buffer.ptr;
      INT *h_cell_counts_es =
          this->sub_group_particle_maps.at(px)->dh_npart_cell_es->h_buffer.ptr;
      this->sub_group_particle_maps.at(px)->create(0, cell_count, h_cell_counts,
                                                   h_cell_counts_es);
    }

    // Populate the particle maps from the particles
    std::vector<INT *> h_layer_maps(this->num_partitions);
    std::vector<INT *> h_npart_cell_es(this->num_partitions);
    for (std::size_t px = 0; px < this->num_partitions; px++) {
      h_layer_maps[px] = this->sub_group_particle_maps.at(px)->d_layer_map->ptr;
      h_npart_cell_es[px] =
          this->sub_group_particle_maps.at(px)->dh_npart_cell_es->d_buffer.ptr;
    }
    this->la_layer_maps->set(h_layer_maps);
    this->la_npart_cell_es->set(h_npart_cell_es);
    this->loop_1->submit();

    // Create the selections from the particle maps.
    for (std::size_t px = 0; px < this->num_partitions; px++) {
      auto [h_npart_cell_ptr, d_npart_cell_ptr, h_npart_cell_es_ptr,
            d_npart_cell_es_ptr] =
          this->sub_group_particle_maps.at(px)->get_helper_ptrs();

      INT npart_total = this->sub_group_particle_maps.at(px)->npart_total;
      auto d_cell_starts_ptr =
          this->sub_group_particle_maps.at(px)->d_cell_starts->ptr;

      Selection s;
      s.npart_local = npart_total;
      s.ncell = cell_count;
      s.h_npart_cell = h_npart_cell_ptr;
      s.d_npart_cell = d_npart_cell_ptr;
      s.d_npart_cell_es = d_npart_cell_es_ptr;
      s.d_map_cells_to_particles = {d_cell_starts_ptr};
      this->partition_selections.at(px) = s;
    }

    this->loop_1->wait();

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_particle_layers);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_cell_counts_INT);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_cell_counts);
  }

public:
  /// The particle property used to partition the ParticleGroup.
  Sym<INT> partition_sym;
  /// The number of partitions that particles can belong to.
  std::size_t num_partitions;

  /**
   * Get the child SubGroupSelector for a particular partition.
   *
   * @param partition Partition to extract.
   * @returns SubGroupSelector for partition.
   */
  inline SubGroupSelectorBaseSharedPtr
  get_selector(const std::size_t partition) {
    NESOASSERT(partition < this->num_partitions, "Bad partition index.");
    auto ptr = std::dynamic_pointer_cast<SubGroupSelectorBase>(
        this->partition_selectors.at(partition));
    NESOASSERT(ptr != nullptr, "Bad pointer cast.");
    return ptr;
  }

  /**
   * Create a orchestration instance that creates num_partition partitions.
   *
   * @param parent ParticleGroup or ParticleSubGroup which is the parent.
   * @param partition_sym Sym to use for partitioning the parent.
   * @param num_partitions Number of partitions to consider.
   */
  template <typename PARENT>
  ParticleGroupPartitioner(std::shared_ptr<PARENT> parent,
                           Sym<INT> partition_sym,
                           const std::size_t num_partitions)
      : SubGroupSelectorBase(parent), partition_sym(partition_sym),
        num_partitions(num_partitions) {
    this->add_sym_dependency(partition_sym);

    this->partition_selectors.reserve(num_partitions);
    this->partition_selections.reserve(num_partitions);
    this->sub_group_particle_maps.reserve(num_partitions);

    for (std::size_t px = 0; px < this->num_partitions; px++) {
      this->partition_selections.emplace_back();
    }

    for (std::size_t px = 0; px < this->num_partitions; px++) {
      std::function<void(Selection * created_selection)> create_handle =
          [=](Selection *created_selection) -> void {
        return this->create_indexed(px, created_selection);
      };

      auto ptr = std::make_shared<ParticleGroupPartitionSelector>(
          parent, this->partition_sym, create_handle);
      this->partition_selectors.push_back(ptr);

      this->sub_group_particle_maps.push_back(
          ptr->get_sub_group_particle_map());
    }

    const auto k_cell_count = static_cast<std::size_t>(
        this->particle_group->domain->mesh->get_cell_count());
    const INT k_num_partitions = static_cast<INT>(num_partitions);

    this->loop_0 = particle_loop(
        "ParticleGroupPartitionCreate0", parent,
        [=](auto INDEX, auto MAP_PTRS, auto PARTITION_SYM) {
          int *k_particle_layers = MAP_PTRS.at(0);
          int *k_cell_counts = MAP_PTRS.at(1);
          const INT partition = PARTITION_SYM.at(0);
          const std::size_t offset =
              static_cast<std::size_t>(partition) * k_cell_count +
              static_cast<std::size_t>(INDEX.cell);
          if ((0 <= partition) && (partition < k_num_partitions)) {
            const int layer = atomic_fetch_add(k_cell_counts + offset, 1);
            k_particle_layers[INDEX.get_loop_linear_index()] = layer;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(this->map_ptrs),
        Access::read(this->partition_sym));

    auto sycl_target = this->particle_group->sycl_target;
    this->la_layer_maps =
        std::make_shared<LocalArray<INT *>>(sycl_target, num_partitions);
    this->la_npart_cell_es =
        std::make_shared<LocalArray<INT *>>(sycl_target, num_partitions);

    this->loop_1 = particle_loop(
        "ParticleGroupPartitionCreate1", parent,
        [=](auto INDEX, auto MAP_PTRS, auto LAYER_MAPS, auto NPART_CELL_ES,
            auto PARTITION_SYM) {
          int *k_particle_layers = MAP_PTRS.at(0);
          const INT partition = PARTITION_SYM.at(0);
          const int layer = k_particle_layers[INDEX.get_loop_linear_index()];
          if ((0 <= partition) && (partition < k_num_partitions)) {
            INT *layer_map = LAYER_MAPS.at(partition);
            INT *npart_cell_es = NPART_CELL_ES.at(partition);
            INT offset = npart_cell_es[INDEX.cell];
            layer_map[offset + layer] = INDEX.layer;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(this->map_ptrs),
        Access::read(this->la_layer_maps), Access::read(this->la_npart_cell_es),
        Access::read(this->partition_sym));
  }
};

} // namespace NESO::Particles::ParticleSubGroupImplementation

namespace NESO::Particles {

/**
 * This is the class which defines the interface that end users should use to
 * partition ParticleGroups and ParticleSubGroups.
 */
class ParticleGroupPartition {
protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif
  std::unique_ptr<ParticleSubGroupImplementation::ParticleGroupPartitioner>
      particle_group_partitioner;

public:
  /// Disable (implicit) copies.
  ParticleGroupPartition(const ParticleGroupPartition &st) = delete;
  /// Disable (implicit) copies.
  ParticleGroupPartition &operator=(ParticleGroupPartition const &a) = delete;

  /**
   * Create a partitioner from a parent ParticleGroup or ParticleSubGroup.
   *
   * @param parent Parent ParticleGroup or ParticleSubGroup to partition.
   * @param partition_sym Particle property to use for partitioning the parent.
   * The first component will be inspected.
   * @param num_partitions Number of partitions to consider.
   */
  template <typename PARENT>
  ParticleGroupPartition(std::shared_ptr<PARENT> parent, Sym<INT> partition_sym,
                         const int num_partitions) {
    NESOASSERT(num_partitions >= 0, "Bad number of partitions");
    this->particle_group_partitioner = std::make_unique<
        ParticleSubGroupImplementation::ParticleGroupPartitioner>(
        parent, partition_sym, static_cast<std::size_t>(num_partitions));
  }

  /**
   * @returns A vector of size num_partitions containing the partitions of the
   * parent.
   */
  inline std::vector<ParticleSubGroupSharedPtr> get() {

    const std::size_t num_partitions =
        this->particle_group_partitioner->num_partitions;
    std::vector<ParticleSubGroupSharedPtr> output(num_partitions);

    for (std::size_t px = 0; px < num_partitions; px++) {
      output[px] = std::make_shared<ParticleSubGroup>(
          this->particle_group_partitioner->get_selector(px));
    }

    return output;
  }
};

} // namespace NESO::Particles

#endif
