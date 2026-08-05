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
  this->d_accumulation_max =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, 16);
  this->d_accumulation_plus =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, 16);

  this->d_staging_values =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, 16);
  this->d_staging_indices =
      std::make_shared<BufferDevice<int>>(this->sycl_target, 16);
}

void CollisionCellRateReduction::reset(const int cell_start,
                                       const int cell_end) {

  this->event_reduce_max.wait_and_throw();
  this->event_reduce_plus.wait_and_throw();

  NESOASSERT(cell_start >= 0, "Bad cell_start.");
  NESOASSERT(cell_end <= this->collision_cell_partition->num_mesh_cells,
             "Bad cell_end.");
  NESOASSERT(cell_start < cell_end,
             "Bad relationship between cell_start and cell_end.");

  const INT max_num_collision_cells =
      this->collision_cell_partition->max_num_collision_cells;
  const INT num_mesh_cells = cell_end - cell_start;
  this->num_entries = num_mesh_cells * max_num_collision_cells;

  this->d_accumulation_max->realloc_no_copy(this->num_entries);
  this->d_accumulation_plus->realloc_no_copy(this->num_entries);

  this->event_fill_plus = this->sycl_target->queue.fill<REAL>(
      this->d_accumulation_plus->ptr, 0.0, this->num_entries);

  this->cell_start = cell_start;
  this->cell_end = cell_end;
}

void CollisionCellRateReduction::submit(
    CellwisePairListAbsolute<ParticleGroup, CellwisePairList> &pair_list,
    Sym<INT> collision_cell_sym, const int collision_cell_component,
    LocalArraySharedPtr<REAL> device_rate_buffer) {

  NESOASSERT(this->cell_start != -1, "Reset not called.");
  NESOASSERT(this->cell_end != -1, "Reset not called.");

  const INT num_pairs = pair_list.pair_list->get_num_pairs_range(
      this->cell_start, this->cell_end);

  NESOASSERT(device_rate_buffer->size >= num_pairs,
             "device_rate_buffer is too small for the passed number of pairs.");

  // We cannot touch the staging arrays whilst a reduction MAX is occuring.
  this->event_reduce_max.wait_and_throw();

  // We queue the zeroing of the max buffer on the previous add event as when
  // the previous add event has completed the max buffer can be zeroed again.
  auto event_fill_max = this->sycl_target->queue.fill<REAL>(
      this->d_accumulation_max->ptr, 0.0, num_entries, this->event_reduce_plus);

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
      Access::read(ParticlePairLoopIndex{}), Access::read(collision_cell_sym)

          )
      ->execute();

  // We cannot touch the accumulation arrays whilst a reduction ADD is occuring
  // or a reduce MAX is occuring. The event_reduce_max is waited on above.
  std::vector<sycl::event> dependent_events_max = {
      event_rate_copy, this->event_reduce_plus, event_fill_max};

  sycl::range<1> iteration_set_max =
      this->sycl_target->device_limits.validate_range_global(
          sycl::range<1>(num_pairs));

  REAL const *const RESTRICT k_max_src = this->d_staging_values->ptr;
  int const *const RESTRICT k_max_indices = this->d_staging_indices->ptr;
  REAL *RESTRICT k_max_dst = this->d_accumulation_max->ptr;

  const auto k_max_num_collision_cells =
      this->collision_cell_partition->max_num_collision_cells;

  this->event_reduce_max = this->sycl_target->queue.parallel_for(
      iteration_set_max, dependent_events_max, [=](sycl::item<1> idx) {
        const int cell_mesh = k_max_indices[idx];
        const int cell_collision = k_max_indices[num_pairs + idx];
        const std::size_t index =
            cell_mesh * k_max_num_collision_cells + cell_collision;
        atomic_fetch_max(k_max_dst + index, k_max_src[idx]);
      });

  // We have to wait for the fill from the reset of the add buffer to complete
  // before starting a new add.
  std::vector<sycl::event> dependent_events_plus = {this->event_reduce_max,
                                                    this->event_fill_plus};
  sycl::range<1> iteration_set_plus =
      this->sycl_target->device_limits.validate_range_global(
          sycl::range<1>(this->num_entries));

  REAL const *const RESTRICT k_plus_src = this->d_accumulation_max->ptr;
  REAL *RESTRICT k_plus_dst = this->d_accumulation_plus->ptr;

  this->event_reduce_plus = sycl_target->queue.parallel_for(
      iteration_set_plus, dependent_events_plus,
      [=](sycl::item<1> idx) { k_plus_dst[idx] += k_plus_src[idx]; });
}

void CollisionCellRateReduction::get(
    NDLocalArraySharedPtr<REAL, 2> &accumulated_rates) {

  const int num_cells = this->cell_end - this->cell_start;
  const auto max_num_collision_cells =
      this->collision_cell_partition->max_num_collision_cells;
  auto shape = nd_index<2>(num_cells, max_num_collision_cells);
  if (accumulated_rates == nullptr) {
    accumulated_rates = std::make_shared<NDLocalArray<REAL, 2>>(
        sycl_target, num_cells, max_num_collision_cells);
  } else {
    NESOASSERT(accumulated_rates->index == shape,
               "accumulated_rates has incorrect shape.");
  }

  REAL *RESTRICT k_output = accumulated_rates->ptr();
  REAL const *const RESTRICT k_input = this->d_accumulation_plus->ptr;

  std::vector<sycl::event> dep_events = {this->event_fill_plus,
                                         this->event_reduce_plus};

  this->sycl_target->queue
      .memcpy(k_output, k_input,
              num_cells * max_num_collision_cells * sizeof(REAL), dep_events)
      .wait_and_throw();
}
} // namespace NESO::Particles::DSMC
