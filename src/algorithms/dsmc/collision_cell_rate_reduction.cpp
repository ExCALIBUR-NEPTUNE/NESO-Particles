#include <neso_particles/algorithms/dsmc/collision_cell_rate_reduction.hpp>
#include <neso_particles/pair_loop/particle_pair_loop_cellwise_pair_list.hpp>

namespace NESO::Particles::DSMC {

CollisionCellRateReduction::CollisionCellRateReduction(

    CollisionCellPartitionSharedPtr collision_cell_partition

    )
    :

      sycl_target(collision_cell_partition->sycl_target),
      collision_cell_partition(collision_cell_partition)

{

  const auto num_mesh_cells = collision_cell_partition->num_mesh_cells;
  this->d_num_collision_cells =
      std::make_shared<BufferDevice<int>>(this->sycl_target, num_mesh_cells);
  this->d_accumulation_max =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, num_mesh_cells);

  this->sycl_target->queue
      .fill<int>(this->d_num_collision_cells->ptr, 0, num_mesh_cells)
      .wait_and_throw();
  this->sycl_target->queue
      .fill<REAL>(this->d_accumulation_max->ptr, 0.0, num_mesh_cells)
      .wait_and_throw();

  this->d_staging_values =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, num_mesh_cells);
  this->d_staging_indices =
      std::make_shared<BufferDevice<int>>(this->sycl_target, num_mesh_cells);
}

void CollisionCellRateReduction::setup(const int num_contributors) {
  this->num_contributors = num_contributors;
}

void CollisionCellRateReduction::resize() {
  auto r0 = this->sycl_target->profile_map.start_region(
      "CollisionCellRateReduction", "resize");

  const int max_num_collision_cells_old =
      this->last_reset_max_num_collision_cells;
  const int max_num_collision_cells_new =
      std::max(this->collision_cell_partition->max_num_collision_cells,
               static_cast<INT>(1));

  NESOWARN(this->collision_cell_partition->max_num_collision_cells > 0,
           "max_num_collision_cells is zero");

  const INT num_mesh_cells = this->collision_cell_partition->num_mesh_cells;

  NESOASSERT((this->last_reset_num_mesh_cells < 0) ||
                 (this->last_reset_num_mesh_cells == num_mesh_cells),
             "The number of mesh cells has changed.");
  this->last_reset_num_mesh_cells = num_mesh_cells;
  this->last_reset_max_num_collision_cells = max_num_collision_cells_new;

  auto e0 = this->sycl_target->queue.memcpy(
      this->d_num_collision_cells->ptr,
      this->collision_cell_partition->num_collision_cells.data(),
      num_mesh_cells * sizeof(int));

  // Validate this iteration set for all later loops.
  this->sycl_target->device_limits.validate_range_global(sycl::range<2>(
      num_mesh_cells, this->collision_cell_partition->max_num_collision_cells));

  const bool realloc =
      max_num_collision_cells_new != max_num_collision_cells_old;

  if (realloc) {

    this->num_entries = num_mesh_cells * max_num_collision_cells_new;

    auto d_accumulation_max_new = std::make_shared<BufferDevice<REAL>>(
        this->sycl_target, this->num_entries * this->num_contributors);

    REAL const *const RESTRICT k_accumulation_old =
        this->d_accumulation_max->ptr;
    REAL *RESTRICT k_accumulation_new = d_accumulation_max_new->ptr;

    auto iteration_set = this->sycl_target->device_limits.validate_range_global(
        sycl::range<3>(this->num_contributors, num_mesh_cells,
                       max_num_collision_cells_new));

    const std::size_t stride_old = num_mesh_cells * max_num_collision_cells_old;
    const std::size_t stride_new = num_mesh_cells * max_num_collision_cells_new;

    this->sycl_target->queue
        .parallel_for(
            iteration_set,
            [=](sycl::item<3> idx) {
              const std::size_t contributor = idx.get_id(0);
              const std::size_t mesh_cell = idx.get_id(1);
              const int collision_cell = idx.get_id(2);
              const REAL value =
                  (collision_cell < max_num_collision_cells_old)
                      ? k_accumulation_old[stride_old * contributor +
                                           mesh_cell *
                                               max_num_collision_cells_old +
                                           collision_cell]
                      : 0.0;

              k_accumulation_new[idx.get_linear_id()] = value;
            })
        .wait_and_throw();

    this->d_accumulation_max = d_accumulation_max_new;
  }

  e0.wait_and_throw();

  this->sycl_target->profile_map.end_region(r0);
}

void CollisionCellRateReduction::update(
    const int contributor_id,
    CellwisePairListAbsolute<ParticleGroup, CellwisePairList> &pair_list,
    const int cell_start, const int cell_end, Sym<INT> collision_cell_sym,
    const int collision_cell_component,
    LocalArraySharedPtr<REAL> device_rate_buffer) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CollisionCellRateReduction", "update(0)");

  NESOASSERT((0 <= contributor_id) && (contributor_id < this->num_contributors),
             "Bad contributor ID passed: " + std::to_string(contributor_id));
  NESOASSERT(cell_start >= 0, "Bad cell_start.");
  NESOASSERT(cell_end <= this->collision_cell_partition->num_mesh_cells,
             "Bad cell_end.");
  NESOASSERT(cell_start < cell_end,
             "Bad relationship between cell_start and cell_end.");

  const INT num_pairs =
      pair_list.pair_list->get_num_pairs_range(cell_start, cell_end);

  NESOASSERT(device_rate_buffer->size >= num_pairs,
             "device_rate_buffer is too small for the passed number of pairs.");

  this->d_staging_values->realloc_no_copy(num_pairs);
  this->d_staging_indices->realloc_no_copy(num_pairs * 2);

  REAL *RESTRICT k_staging_values = this->d_staging_values->ptr;
  int *RESTRICT k_staging_indices = this->d_staging_indices->ptr;
  REAL const *const RESTRICT k_rates = device_rate_buffer->ptr();

  auto event_rate_copy = this->sycl_target->queue.memcpy(
      k_staging_values, k_rates, num_pairs * sizeof(REAL));

  particle_pair_loop(
      "CollisionCellRateReduction::submit", pair_list,
      [=](auto INDEX, auto COLLISION_CELL) {
        const auto linear_loop_index = INDEX.get_loop_linear_index();
        k_staging_indices[linear_loop_index] = INDEX.cell;
        k_staging_indices[num_pairs + linear_loop_index] =
            COLLISION_CELL.at(collision_cell_component);
      },
      Access::read(ParticlePairLoopIndex{}),
      Access::A(Access::read(collision_cell_sym))

          )
      ->execute(cell_start, cell_end);
  event_rate_copy.wait_and_throw();

  sycl::range<1> iteration_set_max =
      this->sycl_target->device_limits.validate_range_global(
          sycl::range<1>(num_pairs));

  REAL const *const RESTRICT k_max_src = this->d_staging_values->ptr;
  int const *const RESTRICT k_max_indices = this->d_staging_indices->ptr;
  REAL *RESTRICT k_max_dst = this->d_accumulation_max->ptr;

  const auto k_max_num_collision_cells =
      this->collision_cell_partition->max_num_collision_cells;
  const auto num_mesh_cells = this->collision_cell_partition->num_mesh_cells;

  const std::size_t offset =
      contributor_id * num_mesh_cells * k_max_num_collision_cells;

  auto event_rate_max = this->sycl_target->queue.parallel_for(
      iteration_set_max, [=](sycl::item<1> idx) {
        const int cell_mesh = k_max_indices[idx];
        const int cell_collision = k_max_indices[num_pairs + idx];
        const std::size_t index =
            offset + cell_mesh * k_max_num_collision_cells + cell_collision;
        atomic_fetch_max(k_max_dst + index, k_max_src[idx]);
      });

  event_rate_max.wait_and_throw();

  this->sycl_target->profile_map.end_region(r0);
}

void CollisionCellRateReduction::update(
    const int contributor_id,
    CellwisePairListAbsolute<ParticleGroup, CellwisePairList> &pair_list,
    const int cell_start, const int cell_end, Sym<INT> collision_cell_sym,
    const int collision_cell_component,
    LocalArraySharedPtr<REAL> device_rate_buffer,
    const std::vector<int> &cell_mask, const REAL rate_init) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CollisionCellRateReduction", "update(1)");

  this->update(contributor_id, pair_list, cell_start, cell_end,
               collision_cell_sym, collision_cell_component,
               device_rate_buffer);
  this->update(contributor_id, cell_mask, rate_init);

  this->sycl_target->profile_map.end_region(r0);
}

void CollisionCellRateReduction::update(const int contributor_id,
                                        const std::vector<int> &cell_mask,
                                        const REAL rate_init) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CollisionCellRateReduction", "update(2)");

  const std::size_t num_mesh_cells =
      this->collision_cell_partition->num_mesh_cells;
  const std::size_t max_num_collision_cells =
      this->collision_cell_partition->max_num_collision_cells;

  NESOASSERT((0 <= contributor_id) && (contributor_id < this->num_contributors),
             "Bad contributor ID passed: " + std::to_string(contributor_id));
  NESOASSERT(cell_mask.size() >= num_mesh_cells,
             "cell_mask has fewer entries than the number of mesh cells.");

  this->d_staging_indices->realloc_no_copy(num_mesh_cells);
  int *RESTRICT k_staging_indices = this->d_staging_indices->ptr;

  auto e0 = this->sycl_target->queue.memcpy(k_staging_indices, cell_mask.data(),
                                            num_mesh_cells * sizeof(int));

  int const *const RESTRICT k_num_collision_cells =
      this->d_num_collision_cells->ptr;

  REAL *RESTRICT k_accumulation_max = this->d_accumulation_max->ptr;
  const std::size_t offset =
      contributor_id * num_mesh_cells * max_num_collision_cells;

  this->sycl_target->queue
      .parallel_for(
          // We validate the iteration set in reset.
          sycl::range<2>(num_mesh_cells, max_num_collision_cells), e0,
          [=](sycl::item<2> idx) {
            const std::size_t mesh_cell = idx.get_id(0);
            const int collision_cell = idx.get_id(1);
            const int num_collision_cells = k_num_collision_cells[mesh_cell];
            const bool mask_set = k_staging_indices[mesh_cell];
            if (mask_set) {
              const REAL value =
                  (collision_cell < num_collision_cells) ? rate_init : 0.0;
              k_accumulation_max[offset + idx.get_linear_id()] = value;
            }
          })
      .wait_and_throw();

  this->sycl_target->profile_map.end_region(r0);
}

void CollisionCellRateReduction::get(
    NDLocalArraySharedPtr<REAL, 2> &accumulated_rates) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CollisionCellRateReduction", "get");

  NESOASSERT(
      this->last_reset_num_mesh_cells ==
          this->collision_cell_partition->num_mesh_cells,
      "Miss-match between the number of mesh cells held and number of mesh "
      "cells in the CollisionCellPartition. Was reset called?");
  NESOASSERT(this->last_reset_max_num_collision_cells ==
                 this->collision_cell_partition->max_num_collision_cells,
             "Miss-match between the max number of collision cells and "
             "the max number of collision cells in the CollisionCellPartition. "
             "Was reset called?");

  const int num_mesh_cells = this->collision_cell_partition->num_mesh_cells;
  const auto max_num_collision_cells =
      this->collision_cell_partition->max_num_collision_cells;
  auto shape = nd_index<2>(num_mesh_cells, max_num_collision_cells);
  if (accumulated_rates == nullptr) {
    accumulated_rates = std::make_shared<NDLocalArray<REAL, 2>>(
        sycl_target, num_mesh_cells, max_num_collision_cells);
  } else {
    NESOASSERT(accumulated_rates->index == shape,
               "accumulated_rates has incorrect shape.");
  }

  {
    REAL *RESTRICT k_output = accumulated_rates->ptr();
    REAL const *const RESTRICT k_input = this->d_accumulation_max->ptr;

    const int k_num_contributors = this->num_contributors;
    int const *const RESTRICT k_num_collision_cells =
        this->d_num_collision_cells->ptr;

    const std::size_t stride = num_mesh_cells * max_num_collision_cells;

    this->sycl_target->queue
        .parallel_for(sycl::range<2>(num_mesh_cells, max_num_collision_cells),
                      [=](sycl::item<2> idx) {
                        const std::size_t mesh_cell = idx.get_id(0);
                        const std::size_t collision_cell = idx.get_id(1);

                        REAL value = 0.0;
                        for (int cx = 0; cx < k_num_contributors; cx++) {
                          const std::size_t offset = cx * stride;
                          value += k_input[offset + idx.get_linear_id()];
                        }

                        const REAL write_value =
                            (collision_cell < k_num_collision_cells[mesh_cell])
                                ? value
                                : 0.0;

                        k_output[idx.get_linear_id()] = write_value;
                      })
        .wait_and_throw();
  }

  this->sycl_target->profile_map.end_region(r0);
}
} // namespace NESO::Particles::DSMC
