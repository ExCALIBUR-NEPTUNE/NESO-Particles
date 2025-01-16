#ifndef _NESO_PARTICLES_CELL_DAT_H_
#define _NESO_PARTICLES_CELL_DAT_H_

#include "../loop/particle_loop_base.hpp"
#include "cell_data.hpp"

namespace NESO::Particles {

class ParticlePacker;
template <typename T> class CellDat;

/**
 *  Defines the access implementations and types for CellDat objects.
 */
namespace Access::CellDat {

/**
 * Access:CellDat::Read<T> read access cellwise.
 */
template <typename T> struct Read {
  /// Pointer to underlying data for the array.
  Read() = default;
  T *const *const *ptr;
  int cell;
  inline const T at(const int row, const int col) {
    return ptr[cell][col][row];
  }
};

/**
 * Access:CellDat::Add<T> add access cellwise.
 */
template <typename T> struct Add {
  /// Pointer to underlying data for the array.
  Add() = default;
  T ***ptr;
  int cell;
  inline T fetch_add(const int row, const int col, const T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        element_atomic(ptr[cell][col][row]);
    return element_atomic.fetch_add(value);
  }
};

/**
 * Access:CellDat::Add<T> write access cellwise.
 */
template <typename T> struct Write {
  /// Pointer to underlying data for the array.
  Write() = default;
  T ***ptr;
  int cell;
  inline T &at(const int row, const int col) { return ptr[cell][col][row]; }
};

} // namespace Access::CellDat

namespace ParticleLoopImplementation {
template <typename T> struct LoopParameter<Access::Read<CellDat<T>>> {
  using type = T *const *const *;
};
template <typename T> struct LoopParameter<Access::Write<CellDat<T>>> {
  using type = T ***;
};
template <typename T> struct LoopParameter<Access::Add<CellDat<T>>> {
  using type = T ***;
};
template <typename T> struct KernelParameter<Access::Read<CellDat<T>>> {
  using type = Access::CellDat::Read<T>;
};
template <typename T> struct KernelParameter<Access::Write<CellDat<T>>> {
  using type = Access::CellDat::Write<T>;
};
template <typename T> struct KernelParameter<Access::Add<CellDat<T>>> {
  using type = Access::CellDat::Add<T>;
};
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              T *const *const *&rhs,
                              Access::CellDat::Read<T> &lhs) {
  lhs.ptr = rhs;
  lhs.cell = iterationx.cellx;
}
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx, T ***&rhs,
                              Access::CellDat::Write<T> &lhs) {
  lhs.ptr = rhs;
  lhs.cell = iterationx.cellx;
}
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx, T ***&rhs,
                              Access::CellDat::Add<T> &lhs) {
  lhs.ptr = rhs;
  lhs.cell = iterationx.cellx;
}
template <typename T>
inline T *const *const *
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<CellDat<T> *> &a) {
  return a.obj->impl_get_const();
}
template <typename T>
inline T ***
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<CellDat<T> *> &a) {
  return a.obj->impl_get();
}
template <typename T>
inline T ***
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Add<CellDat<T> *> &a) {
  return a.obj->impl_get();
}
} // namespace ParticleLoopImplementation

template <typename T> class SymVectorPointerCache;

/**
 * Store data on each cell where the number of columns required per cell is
 * constant but the number of rows is variable. Data is stored in a column
 * major manner with a new device pointer per column.
 */
template <typename T> class CellDat {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS> friend class ParticleLoop;
  template <typename U> friend class ParticleDatT;
  friend class ParticlePacker;
  friend class SymVectorPointerCache<INT>;
  friend class SymVectorPointerCache<REAL>;

  friend T ***ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Add<CellDat<T> *> &a);
  friend T ***ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<CellDat<T> *> &a);
  friend T *const *const *ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<CellDat<T> *> &a);

private:
  // If the lifespan of this member changes then SymVectorPointerCache also
  // needs updating.
  T ***d_ptr;
  std::vector<T **> h_ptr_cells;
  std::vector<T *> h_ptr_cols;
  int nrow_max = -1;
  int nrow_min = -1;

  EventStack stack_events;
  std::stack<T *> stack_ptrs;

protected:
  std::function<void(const int)> write_callback;
  inline void add_write_callback(std::function<void(const int)> fn) {
    this->write_callback = fn;
  }

  inline void set_nrow_inner(const INT cell, const INT nrow_required) {
    const INT nrow_alloced = this->nrow_alloc[cell];
    const INT nrow_existing = this->nrow[cell];

    if (nrow_required == nrow_existing) {
      return;
    } else if (nrow_required < nrow_existing) {
      this->nrow[cell] = nrow_required;
      this->nrow_max = -1;
      return;
    } else {
      if (nrow_required > nrow_alloced) {
        const INT new_nrow_alloced_t =
            (cell == 0) ? 1.2 * nrow_required : 1.1 * nrow_required;
        const INT new_nrow_alloced =
            std::max(nrow_required, new_nrow_alloced_t);
        const int ncol = this->ncol;

        T *cell_ptr_old = this->h_ptr_cols[cell * ncol];
        this->stack_ptrs.push(cell_ptr_old);
        T *cell_ptr_new = (T *)this->sycl_target->malloc_device(
            new_nrow_alloced * ncol * sizeof(T));

        for (int colx = 0; colx < ncol; colx++) {
          T *col_ptr_old = cell_ptr_old + colx * nrow_alloced;
          T *col_ptr_new = cell_ptr_new + colx * new_nrow_alloced;

          if (nrow_alloced > 0) {
            this->stack_events.push(this->sycl_target->queue.memcpy(
                col_ptr_new, col_ptr_old, nrow_existing * sizeof(T)));
          }
          this->h_ptr_cols[cell * ncol + colx] = col_ptr_new;
        }

        this->nrow_alloc[cell] = new_nrow_alloced;
        this->stack_events.push(sycl_target->queue.memcpy(
            this->h_ptr_cells[cell], &this->h_ptr_cols[cell * this->ncol],
            this->ncol * sizeof(T *)));
      }
      this->nrow[cell] = nrow_required;
      this->nrow_max = -1;
    }
  }

  /**
   * Non-const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline T ***impl_get() {
    if (this->write_callback) {
      this->write_callback(0);
    }
    return this->d_ptr;
  }

  /**
   * Const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline T *const *const *impl_get_const() { return this->d_ptr; }

  /**
   * Get the device pointer for a column in a cell.
   *
   * @param cell Cell index to get pointer for.
   * @param col Column in cell to get pointer for.
   * @returns Device pointer to data for the specified column.
   */
  inline T *col_device_ptr(const int cell, const int col) {
    return this->h_ptr_cols[cell * this->ncol + col];
  }

public:
  /// Disable (implicit) copies.
  CellDat(const CellDat &st) = delete;
  /// Disable (implicit) copies.
  CellDat &operator=(CellDat const &a) = delete;

  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  /// Number of cells.
  const int ncells;
  /// Number of rows in each cell.
  std::vector<INT> nrow;
  /// Number of columns, uniform across all cells.
  const int ncol;
  /// Number of rows currently allocated for each cell.
  std::vector<INT> nrow_alloc;
  ~CellDat() {
    this->sycl_target->free(this->h_ptr_cells.at(0));
    // for (int colx = 0; colx < ncells * this->ncol; colx++) {
    for (int cellx = 0; cellx < ncells; cellx++) {
      if (this->h_ptr_cols[cellx * this->ncol] != NULL) {
        this->sycl_target->free(this->h_ptr_cols[cellx * this->ncol]);
      }
    }
    this->sycl_target->free(this->d_ptr);
  };

  /**
   * Create new CellDat on a specified compute target with a specified number of
   * cells and number of columns per cell.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param ncells Number of cells (fixed).
   * @param ncol Number of columns in each cell (fixed).
   */
  inline CellDat(SYCLTargetSharedPtr sycl_target, const int ncells,
                 const int ncol)
      : nrow_max(0), sycl_target(sycl_target), ncells(ncells), ncol(ncol) {

    this->nrow = std::vector<INT>(ncells);
    // If the lifespan of this member changes then SymVectorPointerCache also
    // needs updating.
    this->d_ptr =
        (T ***)this->sycl_target->malloc_device(ncells * sizeof(T **));
    this->h_ptr_cells = std::vector<T **>(ncells);
    this->h_ptr_cols = std::vector<T *>(ncells * ncol);
    this->nrow_alloc = std::vector<INT>(ncells);

    T **cols_base_ptr =
        (T **)this->sycl_target->malloc_device(ncells * ncol * sizeof(T *));

    for (int cellx = 0; cellx < ncells; cellx++) {
      this->nrow_alloc[cellx] = 0;
      this->nrow[cellx] = 0;
      this->h_ptr_cells[cellx] = cols_base_ptr;
      cols_base_ptr += ncol;
      for (int colx = 0; colx < ncol; colx++) {
        this->h_ptr_cols[cellx * ncol + colx] = NULL;
      }
    }

    sycl_target->queue.memcpy(d_ptr, this->h_ptr_cells.data(),
                              ncells * sizeof(T *));

    this->sycl_target->queue.wait();
  };

  /**
   * Pass a number of nrows for each cell where the user promises the new
   * number of rows are equal to or less than the current number of rows.
   */
  template <typename U> inline void reduce_nrow(const U *h_nrow_required) {
    this->nrow_max = -1;
    const int ncells = this->ncells;
    for (int cx = 0; cx < ncells; cx++) {
      this->nrow[cx] = h_nrow_required[cx];
    }
  }

  /**
   * Set the number of rows required in a provided cell. This will realloc if
   * needed and copy the existing data into the new space. May not shrink the
   * array if the requested size is smaller than the existing size.
   * wait_set_nrow should be called before using the dat.
   */
  inline void set_nrow(const INT cell, const INT nrow_required) {
    NESOASSERT(cell >= 0, "Cell index is negative");
    NESOASSERT(cell < this->ncells, "Cell index is >= ncells");
    NESOASSERT(nrow_required >= 0, "Requested number of rows is negative");
    set_nrow_inner(cell, nrow_required);
  }

  /**
   * Wait for set_nrow to complete
   */
  inline void wait_set_nrow() {
    // wait for the events - memcpys
    this->stack_events.wait();
    // can now free the pointers
    while (!this->stack_ptrs.empty()) {
      auto ptr = this->stack_ptrs.top();
      if (ptr != NULL) {
        this->sycl_target->free(ptr);
      }
      this->stack_ptrs.pop();
    }
  }

  /**
   *  Recompute nrow_max from current row counts.
   *
   *  @returns The maximum number of rows across all cells.
   */
  inline int compute_nrow_max() {
    this->nrow_max =
        *std::max_element(std::begin(this->nrow), std::end(this->nrow));
    this->nrow_min =
        *std::min_element(std::begin(this->nrow), std::end(this->nrow));
    return this->nrow_max;
  }

  /**
   * Get the maximum number of rows across all cells.
   *
   * @returns The maximum number of rows across all cells.
   */
  inline int get_nrow_max() {
    if (this->nrow_max < 0) {
      this->compute_nrow_max();
    }
    return this->nrow_max;
  }

  /**
   * Get the minimum number of rows across all cells.
   *
   * @returns The minimum number of rows across all cells.
   */
  inline int get_nrow_min() {
    if (this->nrow_max < 0) {
      this->compute_nrow_max();
    }
    return this->nrow_min;
  }

  /**
   * Get the contents of a provided cell on the host as a CellData instance.
   *
   * @param cell Cell to get data from.
   * @returns Cell contents of specified cell as CellData instance.
   */
  inline CellData<T> get_cell(const int cell) {
    auto t0 = profile_timestamp();

    auto cell_data = std::make_shared<CellDataT<T>>(
        this->sycl_target, this->nrow[cell], this->ncol);

    if (this->nrow[cell] > 0) {
      for (int colx = 0; colx < this->ncol; colx++) {
        this->sycl_target->queue.memcpy(
            cell_data->data[colx].data(),
            this->h_ptr_cols[cell * this->ncol + colx],
            this->nrow[cell] * sizeof(T));
      }
      this->sycl_target->queue.wait();
    }

    sycl_target->profile_map.inc("CellDat", "get_cell", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    return cell_data;
  }

  /**
   * Get a value from a cell, row and column.
   *
   * @param cell Cell index.
   * @param row Row index.
   * @param col Column index.
   * @returns Value at location.
   */
  inline T get_value(const int cell, const int row, const int col) {
    T value;
    this->sycl_target->queue
        .memcpy(&value, this->h_ptr_cols[cell * this->ncol + col] + row,
                sizeof(T))
        .wait_and_throw();
    return value;
  }

  /**
   * Set a value directly using cell, row, column and value.
   *
   * @param cell Cell index.
   * @param row Row index.
   * @param col Column index.
   * @param value Value to set at location.
   */
  inline void set_value(const int cell, const int row, const int col,
                        const T value) {
    this->sycl_target->queue
        .memcpy(this->h_ptr_cols[cell * this->ncol + col] + row, &value,
                sizeof(T))
        .wait_and_throw();
  }

  /**
   * Get the contents of a provided cell on the host as a CellData instance.
   *
   * @param cell Cell to get data from.
   * @param cell_data CellDataT instance to populate, must be sufficiently
   * sized.
   * @param event_stack EventStack instance to call wait on for copy.
   */
  inline void get_cell_async(const int cell, CellDataT<T> &cell_data,
                             EventStack &event_stack) {
    auto t0 = profile_timestamp();

    NESOASSERT(cell_data.nrow >= this->nrow[cell],
               "CellDataT has insufficent number of rows");
    NESOASSERT(cell_data.ncol >= this->ncol,
               "CellDataT has insufficent number of columns");

    if (this->nrow[cell] > 0) {
      for (int colx = 0; colx < this->ncol; colx++) {
        event_stack.push(this->sycl_target->queue.memcpy(
            cell_data.data[colx].data(),
            this->h_ptr_cols[cell * this->ncol + colx],
            this->nrow[cell] * sizeof(T)));
      }
    }

    sycl_target->profile_map.inc("CellDat", "get_cell_async", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    return;
  }

  /**
   * Set the contents of a cell on the device using a CellData instance.
   *
   * @param cell Cell index to set data in.
   * @param cell_data New cell data to set.
   */
  inline void set_cell(const int cell, CellData<T> cell_data) {
    auto t0 = profile_timestamp();
    if (this->write_callback) {
      this->write_callback(0);
    }
    NESOASSERT(cell_data->nrow >= this->nrow[cell],
               "CellData as insuffient row count.");
    NESOASSERT(cell_data->ncol >= this->ncol,
               "CellData as insuffient column count.");

    if (this->nrow[cell] > 0) {
      for (int colx = 0; colx < this->ncol; colx++) {

        this->sycl_target->queue.memcpy(
            this->h_ptr_cols[cell * this->ncol + colx],
            cell_data->data[colx].data(), this->nrow[cell] * sizeof(T));
      }
      this->sycl_target->queue.wait();
    }

    sycl_target->profile_map.inc("CellDat", "set_cell", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  }

  /**
   * Set the contents of a cell on the device using a CellData instance.
   *
   * @param cell Cell index to set data in.
   * @param cell_data New cell data to set.
   * @param event_stack EventStack instance to wait on.
   */
  inline void set_cell_async(const int cell, CellDataT<T> &cell_data,
                             EventStack &event_stack) {
    auto t0 = profile_timestamp();
    if (this->write_callback) {
      this->write_callback(0);
    }
    NESOASSERT(cell_data.nrow >= this->nrow[cell],
               "CellData as insuffient row count.");
    NESOASSERT(cell_data.ncol >= this->ncol,
               "CellData as insuffient column count.");

    if (this->nrow[cell] > 0) {
      for (int colx = 0; colx < this->ncol; colx++) {

        event_stack.push(this->sycl_target->queue.memcpy(
            this->h_ptr_cols[cell * this->ncol + colx],
            cell_data.data[colx].data(), this->nrow[cell] * sizeof(T)));
      }
    }

    sycl_target->profile_map.inc("CellDat", "set_cell_async", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  }

  /**
   * Get the root device pointer for the data storage. Data can be accessed
   * on the device in SYCL kernels with access like:
   *    d[cell_index][column_index][row_index]
   *
   * @returns Device pointer that can be used to access the underlying data.
   */
  T ***device_ptr() {
    if (this->write_callback) {
      this->write_callback(1);
    }
    return this->d_ptr;
  };

  /**
   *  Helper function to print the contents of all cells or a specified range of
   * cells.
   *
   *  @param start (optional) First cell to print.
   *  @param end (option) Last cell minus one to print.
   */
  inline void print(int start = -1, int end = -1) {

    start = (start < 0) ? 0 : start;
    end = (end < 0) ? ncells : end;

    for (int cx = start; cx < end; cx++) {
      std::cout << "------- " << cx << " -------" << std::endl;
      auto cell = this->get_cell(cx);
      cell->print();
    }
    std::cout << "-----------------" << std::endl;
  }

  /**
   * Number of bytes to store a row of this CellDat
   *
   * @returns Number of bytes required to store a row.
   */
  inline size_t row_size() { return ((size_t)this->ncol) * sizeof(T); }
};

} // namespace NESO::Particles

#endif
