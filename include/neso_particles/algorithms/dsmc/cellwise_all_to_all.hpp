#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_ALL_TO_ALL_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_ALL_TO_ALL_HPP_

#include "../../compute_target.hpp"
#include "../../containers/cell_dat.hpp"
#include "../../device_buffers.hpp"

#include <map>
#include <vector>

namespace NESO::Particles::DSMC {

struct CellwiseAllToAll {
  const int workgroup_size{0};
  const int workgroup_index{0};

  CellwiseAllToAll(const int workgroup_size, const int workgroup_index)
      : workgroup_size(workgroup_size), workgroup_index(workgroup_index) {}

  template <typename KERNEL_FUNC_TYPE, typename SYNC_FUNC_TYPE>
  inline void
  apply_inner_block(const int workgroup_block_item,
                    const int workgroup_block_size, const int block_start_row,
                    const int block_start_col, const int n,
                    KERNEL_FUNC_TYPE &kernel_func, SYNC_FUNC_TYPE &sync_func,
                    const bool mask_required) const {

    int index = workgroup_block_item;
    const int mask = workgroup_block_size - 1;
    if (mask_required) {
      for (int rowx = block_start_row;
           rowx < (block_start_row + workgroup_block_size); rowx++) {
        const int colx = index + block_start_col;
        if ((rowx < n) && (colx < n)) {
          kernel_func(rowx, colx);
        }
        sync_func();
        index = (index + 1) & mask;
      }
    } else {
      for (int rowx = block_start_row;
           rowx < (block_start_row + workgroup_block_size); rowx++) {
        const int colx = index + block_start_col;
        kernel_func(rowx, colx);
        sync_func();
        index = (index + 1) & mask;
      }
    }
  }
  template <typename KERNEL_FUNC_TYPE, typename SYNC_FUNC_TYPE>
  inline void apply_diagonal(const int n, const int block_offset,
                             KERNEL_FUNC_TYPE &kernel_func,
                             SYNC_FUNC_TYPE &sync_func,
                             const bool mask_required) const {
    int size = this->workgroup_size;
    while (size > 0) {
      const int workgroup_block = this->workgroup_index / size;
      const int workgroup_block_item = this->workgroup_index % size;

      int block_end_row = size + workgroup_block * 2 * size;
      int block_start_row = block_end_row - size;
      int block_start_col = block_end_row;

      block_start_row += block_offset;
      block_start_col += block_offset;

      this->apply_inner_block(workgroup_block_item, size, block_start_row,
                              block_start_col, n, kernel_func, sync_func,
                              mask_required);

      size /= 2;
    }
  }
  template <typename KERNEL_FUNC_TYPE, typename SYNC_FUNC_TYPE>
  inline void apply(const int n, KERNEL_FUNC_TYPE kernel_func,
                    SYNC_FUNC_TYPE sync_func) const {
    const int block_size_inner = this->workgroup_size;
    const int block_size = this->workgroup_size * 2;
    const auto n_padded = get_next_multiple(n, block_size);
    const int num_blocks = n_padded / block_size;

    for (int blockx = 0; blockx < num_blocks; blockx++) {

      const int block_offset = blockx * block_size;
      const bool masking_diagonal = blockx == (num_blocks - 1);
      // Process the diagonal blocks. The last block needs masking.
      apply_diagonal(n, block_offset, kernel_func, sync_func, masking_diagonal);

      // process the rest of the blocks in the remainder of the row
      const int row_start = block_offset;
      const int row_end = block_offset + block_size;
      const int col_start = block_offset + block_size;
      const int col_end = n_padded;

      for (int rowx = row_start; rowx < row_end; rowx += block_size_inner) {
        for (int colx = col_start; colx < col_end; colx += block_size_inner) {
          apply_inner_block(this->workgroup_index, this->workgroup_size, rowx,
                            colx, n, kernel_func, sync_func,
                            ((colx + block_size_inner) > n) ||
                                masking_diagonal);
        }
      }
    }
  }
};

} // namespace NESO::Particles::DSMC

#endif
