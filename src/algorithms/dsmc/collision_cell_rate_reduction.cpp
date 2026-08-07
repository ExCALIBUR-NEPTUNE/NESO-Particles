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

  this->d_accumulation_max =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, num_mesh_cells);
  this->d_accumulation_plus =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, num_mesh_cells);

  this->d_staging_values =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, num_mesh_cells);
  this->d_staging_indices =
      std::make_shared<BufferDevice<int>>(this->sycl_target, num_mesh_cells);
}

void CollisionCellRateReduction::setup(const int num_contributors) {
  this->num_contributors = num_contributors;
}

void CollisionCellRateReduction::resize() {

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

  const bool realloc =
      max_num_collision_cells_new != max_num_collision_cells_old;

  if (realloc) {

    this->num_entries = num_mesh_cells * max_num_collision_cells_new;

    this->d_accumulation_plus->realloc_no_copy(this->num_entries);

    auto d_accumulation_max_new = std::make_shared<BufferDevice<REAL>>(
        this->sycl_target, this->num_entries * this->num_contributors);

    REAL const *const RESTRICT k_accumulation_old =
        this->d_accumulation_max->ptr;
    REAL *RESTRICT k_accumulation_new = d_accumulation_max_new->ptr;

    auto iteration_set = this->sycl_target->device_limits.validate_range_global(
        sycl::range<3>(this->num_contributors, num_mesh_cells,
                       max_num_collision_cells_new));

    const int k_max_num_collision_cells_old =
        this->last_reset_max_num_collision_cells;

    const std::size_t stride_old = num_mesh_cells * max_num_collision_cells_old;
    const std::size_t stride_new = num_mesh_cells * max_num_collision_cells_new;

    this->sycl_target->queue
        .parallel_for(
            iteration_set,
            [=](sycl::item<3> idx) {
              const std::size_t contributor = idx.get_id(0);
              const std::size_t mesh_cell = idx.get_id(1);
              const std::size_t collision_cell = idx.get_id(2);
              const REAL value =
                  (collision_cell < k_max_num_collision_cells_old)
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
}

void CollisionCellRateReduction::reset() {

  this->event_fill_plus.wait_and_throw();
  this->event_reduce_max.wait_and_throw();
  this->event_reduce_plus.wait_and_throw();

  const INT max_num_collision_cells =
      std::max(this->collision_cell_partition->max_num_collision_cells,
               static_cast<INT>(1));

  NESOWARN(this->collision_cell_partition->max_num_collision_cells > 0,
           "max_num_collision_cells is zero");

  const INT num_mesh_cells = this->collision_cell_partition->num_mesh_cells;
  this->num_entries = num_mesh_cells * max_num_collision_cells;

  this->d_accumulation_max->realloc_no_copy(this->num_entries);
  this->d_accumulation_plus->realloc_no_copy(this->num_entries);

  this->event_fill_plus = this->sycl_target->queue.fill<REAL>(
      this->d_accumulation_plus->ptr, 0.0, this->num_entries);

  this->last_reset_num_mesh_cells =
      this->collision_cell_partition->num_mesh_cells;
  this->last_reset_max_num_collision_cells =
      this->collision_cell_partition->max_num_collision_cells;

  // At the end of this call:
  //  * There is a zero of the d_accumulation_plus buffer in flight.
}

void CollisionCellRateReduction::update(const int contributor_id,
                                        const std::vector<int> &cell_mask,
                                        const REAL rate_init) {}

void CollisionCellRateReduction::submit(
    CellwisePairListAbsolute<ParticleGroup, CellwisePairList> &pair_list,
    const int cell_start, const int cell_end, Sym<INT> collision_cell_sym,
    const int collision_cell_component,
    LocalArraySharedPtr<REAL> device_rate_buffer) {

  NESOASSERT(cell_start >= 0, "Bad cell_start.");
  NESOASSERT(cell_end <= this->collision_cell_partition->num_mesh_cells,
             "Bad cell_end.");
  NESOASSERT(cell_start < cell_end,
             "Bad relationship between cell_start and cell_end.");

  const INT num_pairs =
      pair_list.pair_list->get_num_pairs_range(cell_start, cell_end);

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
      ->execute(cell_start, cell_end);

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

  // We cannot return control to the user until we have finished copying the
  // supplied rates.
  event_rate_copy.wait_and_throw();

  // At the end of this call:
  //  * There is a zero of the d_accumulation_plus buffer in flight still in
  //  flight from reset.
  //  * A zero of the d_accumulation_max buffer in flight.
  //  * An atomic max loop in flight.
  //  * An increment loop in flight.
}

void CollisionCellRateReduction::get(
    NDLocalArraySharedPtr<REAL, 2> &accumulated_rates) {

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
    REAL const *const RESTRICT k_input = this->d_accumulation_plus->ptr;
    this->sycl_target->queue
        .memcpy(k_output, k_input,
                num_mesh_cells * max_num_collision_cells * sizeof(REAL))
        .wait_and_throw();
  }
}
} // namespace NESO::Particles::DSMC
