#ifndef _NESO_PARTICLES_PARTICLE_SUB_GROUP_PARTICLE_GROUP_PARTITION_HPP_
#define _NESO_PARTICLES_PARTICLE_SUB_GROUP_PARTICLE_GROUP_PARTITION_HPP_

#include "../device_functions.hpp"
#include "../loop/particle_loop.hpp"
#include "../loop/particle_loop_base.hpp"
#include "particle_sub_group_base.hpp"
#include "sub_group_selector.hpp"

namespace NESO::Particles {
template <typename PARENT>
std::vector<ParticleSubGroupSharedPtr>
particle_group_partition(std::shared_ptr<PARENT> parent, Sym<INT> partition_sym,
                         const int num_partitions);
}

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

  template <typename PARENT>
  friend std::vector<ParticleSubGroupSharedPtr>
  NESO::Particles::particle_group_partition(std::shared_ptr<PARENT> parent,
                                            Sym<INT> partition_sym,
                                            const int num_partitions);

protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  /// This shared_ptr stops the partitioner going out of scope until all of the
  /// partition selectors which use it are freed.
  std::shared_ptr<ParticleGroupPartitioner> particle_group_partitioner;

  std::function<void(Selection *created_selection)> create_handle;
  std::function<void()> destroy_handle;

  virtual inline void create(Selection *created_selection) override {
    this->create_handle(created_selection);
  }

  inline std::shared_ptr<SubGroupParticleMap> get_sub_group_particle_map() {
    return this->sub_group_particle_map;
  }

  inline void set_create_handle(
      std::function<void(Selection *created_selection)> create_handle) {
    this->create_handle = create_handle;
  }
  inline void set_destroy_handle(std::function<void()> destroy_handle) {
    this->destroy_handle = destroy_handle;
  }

public:
  virtual ~ParticleGroupPartitionSelector() override { this->destroy_handle(); }

  template <typename PARENT>
  ParticleGroupPartitionSelector(std::shared_ptr<PARENT> parent,
                                 Sym<INT> partition_sym)
      : SubGroupSelectorBase(parent) {
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
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  std::vector<Selection> partition_selections;

  /// The selectors used by the sub groups hold a shared pointer to an instance
  /// of this class, which actually does the partitioning. Hence in this class
  /// we hold weak pointers to the sub group selectors to avoid a cyclic set of
  /// shared pointers.
  std::vector<std::weak_ptr<ParticleGroupPartitionSelector>>
      partition_selectors;
  std::vector<std::weak_ptr<SubGroupParticleMap>> sub_group_particle_maps;

  std::shared_ptr<LocalArray<int>> la_still_exists;
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

    // Find once at the begining of this function which of the selectors
    // actually still exist and have not been deleted.
    std::vector<int> still_exists(this->num_partitions);
    for (std::size_t px = 0; px < this->num_partitions; px++) {
      still_exists[px] =
          static_cast<int>(!this->sub_group_particle_maps.at(px).expired());
    }
    this->la_still_exists->set(still_exists);

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

public:
  /// The particle property used to partition the ParticleGroup.
  Sym<INT> partition_sym;
  /// The number of partitions that particles can belong to.
  std::size_t num_partitions;

  /**
   * Create a orchestration instance that creates num_partition partitions.
   *
   * @param parent ParticleGroup or ParticleSubGroup which is the parent.
   * @param partition_sym Sym to use for partitioning the parent.
   * @param num_partitions Number of partitions to consider.
   * @param partition_selectors_in The selectors which will be used to construct
   * the ParticleSubGroups.
   */
  template <typename PARENT>
  ParticleGroupPartitioner(
      std::shared_ptr<PARENT> parent, Sym<INT> partition_sym,
      const std::size_t num_partitions,
      std::vector<std::shared_ptr<ParticleGroupPartitionSelector>>
          &partition_selectors_in)
      : SubGroupSelectorBase(parent), partition_sym(partition_sym),
        num_partitions(num_partitions) {
    NESOASSERT(partition_selectors_in.size() == num_partitions,
               "size missmatch");

    this->add_sym_dependency(partition_sym);

    this->partition_selectors.reserve(num_partitions);
    this->partition_selections.reserve(num_partitions);
    this->sub_group_particle_maps.reserve(num_partitions);

    for (std::size_t px = 0; px < this->num_partitions; px++) {
      this->partition_selections.emplace_back();
    }

    for (std::size_t px = 0; px < this->num_partitions; px++) {
      auto ptr_shared = partition_selectors_in.at(px);
      std::weak_ptr<ParticleGroupPartitionSelector> ptr_weak = ptr_shared;
      this->partition_selectors.push_back(ptr_weak);

      std::function<void(Selection * created_selection)> create_handle =
          [=](Selection *created_selection) -> void {
        return this->create_indexed(px, created_selection);
      };
      ptr_shared->set_create_handle(create_handle);
      std::function<void()> destroy_handle = [=]() -> void {
        this->partition_selectors.at(px).reset();
        this->sub_group_particle_maps.at(px).reset();
      };
      ptr_shared->set_destroy_handle(destroy_handle);

      this->sub_group_particle_maps.push_back(
          ptr_shared->get_sub_group_particle_map());
    }

    const auto k_cell_count = static_cast<std::size_t>(
        this->particle_group->domain->mesh->get_cell_count());
    const INT k_num_partitions = static_cast<INT>(num_partitions);

    auto sycl_target = this->particle_group->sycl_target;
    this->la_still_exists =
        std::make_shared<LocalArray<int>>(sycl_target, num_partitions);

    this->loop_0 = particle_loop(
        "ParticleGroupPartitionCreate0", parent,
        [=](auto INDEX, auto MAP_PTRS, auto PARTITION_SYM, auto STILL_EXISTS) {
          int *k_particle_layers = MAP_PTRS.at(0);
          int *k_cell_counts = MAP_PTRS.at(1);
          const INT partition = PARTITION_SYM.at(0);
          const std::size_t offset =
              static_cast<std::size_t>(partition) * k_cell_count +
              static_cast<std::size_t>(INDEX.cell);
          if ((0 <= partition) && (partition < k_num_partitions)) {
            // If the sub group for this partition has been deleted then there
            // is no point doing this atomic.
            if (STILL_EXISTS.at(partition)) {
              const int layer = atomic_fetch_add(k_cell_counts + offset, 1);
              k_particle_layers[INDEX.get_loop_linear_index()] = layer;
            }
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(this->map_ptrs),
        Access::read(this->partition_sym), Access::read(this->la_still_exists));

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
            // If the layer_map pointer is null then the ParticleSubGroup for
            // this partition has been deleted, e.g. by going out of scope, and
            // hence we do not try to build the map for this partition.
            if (layer_map != nullptr) {
              INT offset = npart_cell_es[INDEX.cell];
              layer_map[offset + layer] = INDEX.layer;
            }
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
 * Partition a ParticleGroup or ParticleSubGroup into N paritions based on a
 * value held on each particle.
 *
 * This function is a computationally cheaper implementation of
 *
 *  std::vector<ParticleSubGroupSharedPtr> partitions(num_partitions);
 *  for(int px=0 ; px<num_partitions ; px++){
 *    partitions[px] = particle_sub_group(
 *      parent,
 *      [=](auto P) {
 *        return P.at(0) == px;
 *      },
 *      Access::read(partition_sym)
 *    );
 *  }
 *
 *  and will update all the ParticleSubGroups at the same time.
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup to partition.
 * @param partition_sym Particle property to use for partitioning the parent.
 * The first component will be inspected.
 * @param num_partitions Number of partitions to consider.
 * @returns Vector of ParticleSubGroups partitioned using the requested Sym.
 */
template <typename PARENT>
std::vector<ParticleSubGroupSharedPtr>
particle_group_partition(std::shared_ptr<PARENT> parent, Sym<INT> partition_sym,
                         const int num_partitions) {

  std::vector<std::shared_ptr<
      ParticleSubGroupImplementation::ParticleGroupPartitionSelector>>
      partition_selectors(num_partitions);

  for (int px = 0; px < num_partitions; px++) {
    auto ptr = std::make_shared<
        ParticleSubGroupImplementation::ParticleGroupPartitionSelector>(
        parent, partition_sym);
    partition_selectors[px] = ptr;
  }

  auto particle_group_partitioner = std::make_shared<
      ParticleSubGroupImplementation::ParticleGroupPartitioner>(
      parent, partition_sym, static_cast<std::size_t>(num_partitions),
      partition_selectors);

  // The selectors used to create the ParticleSubGroups each hold a shared_ptr
  // to the partitioner object. The partitioner holds weak_ptrs to the data
  // structures inside the selectors.
  for (int px = 0; px < num_partitions; px++) {
    partition_selectors[px]->particle_group_partitioner =
        particle_group_partitioner;
  }

  std::vector<ParticleSubGroupSharedPtr> sub_groups(num_partitions);
  for (int px = 0; px < num_partitions; px++) {
    sub_groups[px] =
        std::make_shared<ParticleSubGroup>(partition_selectors[px]);
  }
  return sub_groups;
}

} // namespace NESO::Particles

#endif
