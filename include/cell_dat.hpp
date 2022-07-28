#ifndef _NESO_PARTICLES_CELL_DAT
#define _NESO_PARTICLES_CELL_DAT

#include <CL/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stack>
#include <string>
#include <vector>

#include "access.hpp"
#include "compute_target.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 * Container for the data within a single cell stored on the host. Data is
 * store column wise. The data is typically particle data for a single
 * ParticleDat for a single cell and exists as a 2D data structure.
 */
template <typename T> class CellDataT {
private:
  // std::format is C++20.......
  std::string format(INT value) {
    char buffer[128];
    const int err = snprintf(buffer, 128, "%ld", value);
    return std::string(buffer);
  }
  std::string format(REAL value) {
    char buffer[128];
    const int err = snprintf(buffer, 128, "%f", value);
    return std::string(buffer);
  }
  std::string format(char value) { return std::string(1, value); }
  SYCLTarget &sycl_target;

public:
  /// Number of rows in the 2D data structure.
  const int nrow;
  /// Number of columns in the 2D data structure.
  const int ncol;
  // 2D data.
  std::vector<std::vector<T>> data;

  /**
   * Create a new, empty and uninitialised container for 2D data of the
   * specified shape.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param nrow Number of rows.
   * @param ncol Number of columns.
   */
  inline CellDataT(SYCLTarget &sycl_target, const int nrow, const int ncol)
      : sycl_target(sycl_target), nrow(nrow), ncol(ncol) {
    this->data = std::vector<std::vector<T>>(ncol);
    for (int colx = 0; colx < ncol; colx++) {
      this->data[colx] = std::vector<T>(nrow);
    }
  }

  /**
   *  Subscript operator for cell data. Data should be indexed by column then
   * row. e.g. CellData cell_data; T value = *cell_data[col][row];
   */
  inline std::vector<T> &operator[](int col) { return this->data[col]; }

  inline void print() {

    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        std::cout << this->format(this->data[colx][rowx]) << " ";
      }
      std::cout << std::endl;
    }
  }
};

template <typename T> using CellData = std::shared_ptr<CellDataT<T>>;

/**
 *  Container that allocates on the device a matrix of fixed size nrow X ncol
 *  for N cells. Data stored in column major format. i.e. Data order from
 *  slowest to fastest is: cell, column, row.
 */
template <typename T> class CellDatConst {
private:
  T *d_ptr;
  const int stride;

public:
  /// Compute device used by the instance.
  SYCLTarget &sycl_target;
  /// Number of cells, labeled 0,...,N-1.
  const int ncells;
  /// Number of rows in each cell.
  const int nrow;
  /// Number of columns in each cell.
  const int ncol;
  ~CellDatConst() { sycl::free(this->d_ptr, sycl_target.queue); };
  /**
   * Create new CellDatConst on the specified compute target with a fixed cell
   * count, fixed number of rows per cell and fixed number of columns per cell.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param ncells Number of cells.
   * @param nrow Number of rows.
   * @param ncol Number of columns.
   */
  CellDatConst(SYCLTarget &sycl_target, const int ncells, const int nrow,
               const int ncol)
      : sycl_target(sycl_target), ncells(ncells), nrow(nrow), ncol(ncol),
        stride(nrow * ncol) {
    this->d_ptr =
        sycl::malloc_device<T>(ncells * nrow * ncol, sycl_target.queue);
    this->sycl_target.queue.fill(this->d_ptr, ((T)0), ncells * nrow * ncol);
    this->sycl_target.queue.wait();
  };

  /**
   * Helper function to index into the stored data. Note column major format.
   *
   * @param cell Cell index to index into.
   * @param row Row to access.
   * @param col Column to access.
   * @returns Linear index into data structure for the specified row and column.
   */
  inline int idx(const int cell, const int row, const int col) {
    return (this->stride * cell) + (this->nrow * col + row);
  };

  /**
   * Get the device pointer for the underlying data. Only accessible on the
   * device.
   *
   * @return Returns a device pointer on the compute target.
   */
  T *device_ptr() { return this->d_ptr; };

  /**
   * Get the data stored in a provided cell on the host as a CellData
   * instance.
   *
   * @param cell Cell to access the underlying data for.
   * @returns Cell data for cell.
   */
  inline CellData<T> get_cell(const int cell) {
    auto cell_data = std::make_shared<CellDataT<T>>(this->sycl_target,
                                                    this->nrow, this->ncol);
    for (int colx = 0; colx < this->ncol; colx++) {
      this->sycl_target.queue.memcpy(
          cell_data->data[colx].data(),
          &this->d_ptr[cell * this->stride + colx * this->nrow],
          this->nrow * sizeof(T));
    }
    this->sycl_target.queue.wait();
    return cell_data;
  }
  /**
   * Set the data in a cell using a CellData instance.
   *
   * @param cell Cell to set data for.
   * @param cell_data Source data.
   */
  inline void set_cell(const int cell, CellData<T> cell_data) {
    NESOASSERT(cell_data->nrow >= this->nrow,
               "CellData as insuffient row count.");
    NESOASSERT(cell_data->ncol >= this->ncol,
               "CellData as insuffient column count.");

    for (int colx = 0; colx < this->ncol; colx++) {
      this->sycl_target.queue.memcpy(
          &this->d_ptr[cell * this->stride + colx * this->nrow],
          cell_data->data[colx].data(), this->nrow * sizeof(T));
    }
    this->sycl_target.queue.wait();
  }
};

/**
 * Store data on each cell where the number of columns required per cell is
 * constant but the number of rows is variable. Data is stored in a column
 * major manner with a new device pointer per column.
 */
template <typename T> class CellDat {
private:
  T ***d_ptr;
  std::vector<T **> h_ptr_cells;
  std::vector<T *> h_ptr_cols;
  int nrow_max = -1;

  EventStack stack_events;
  std::stack<T *> stack_ptrs;

public:
  /// Compute device used by the instance.
  SYCLTarget &sycl_target;
  /// Number of cells.
  const int ncells;
  /// Number of rows in each cell.
  std::vector<INT> nrow;
  /// Number of columns, uniform across all cells.
  const int ncol;
  /// Number of rows currently allocated for each cell.
  std::vector<INT> nrow_alloc;
  ~CellDat() {
    // issues on cuda backend w/o this NULL check.
    for (int cellx = 0; cellx < ncells; cellx++) {
      if ((this->nrow_alloc[cellx] != 0) &&
          (this->h_ptr_cells[cellx] != NULL)) {
        sycl::free(this->h_ptr_cells[cellx], sycl_target.queue);
      }
    }
    for (int colx = 0; colx < ncells * this->ncol; colx++) {
      if (this->h_ptr_cols[colx] != NULL) {
        sycl::free(this->h_ptr_cols[colx], sycl_target.queue);
      }
    }
    sycl::free(this->d_ptr, sycl_target.queue);
  };

  /**
   * Create new CellDat on a specified compute target with a specified number of
   * cells and number of columns per cell.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param ncells Number of cells (fixed).
   * @param ncol Number of columns in each cell (fixed).
   */
  inline CellDat(SYCLTarget &sycl_target, const int ncells, const int ncol)
      : sycl_target(sycl_target), ncells(ncells), ncol(ncol), nrow_max(0) {

    this->nrow = std::vector<INT>(ncells);
    this->d_ptr = sycl::malloc_device<T **>(ncells, sycl_target.queue);
    this->h_ptr_cells = std::vector<T **>(ncells);
    this->h_ptr_cols = std::vector<T *>(ncells * ncol);
    this->nrow_alloc = std::vector<INT>(ncells);

    for (int cellx = 0; cellx < ncells; cellx++) {
      this->nrow_alloc[cellx] = 0;
      this->nrow[cellx] = 0;
      this->h_ptr_cells[cellx] =
          sycl::malloc_device<T *>(ncol, sycl_target.queue);
      for (int colx = 0; colx < ncol; colx++) {
        this->h_ptr_cols[cellx * ncol + colx] = NULL;
      }
    }

    sycl_target.queue.memcpy(d_ptr, this->h_ptr_cells.data(),
                             ncells * sizeof(T *));

    this->sycl_target.queue.wait();
  };

  /**
   * Set the number of rows required in a provided cell. This will realloc if
   * needed and copy the existing data into the new space. May not shrink the
   * array if the requested size is smaller than the existing size.
   * wait_set_nrow should be called before using the dat.
   */
  inline void set_nrow(const INT cell, const INT nrow_required) {

    auto t0 = profile_timestamp();

    if (nrow_required < this->nrow[cell]) {
      this->nrow[cell] = nrow_required;
      this->nrow_max = -1;
      sycl_target.profile_map.inc("CellDat", "set_nrow", 1,
                                  profile_elapsed(t0, profile_timestamp()));
      return;
    }

    NESOASSERT(cell >= 0, "Cell index is negative");
    NESOASSERT(cell < this->ncells, "Cell index is >= ncells");
    NESOASSERT(nrow_required >= 0, "Requested number of rows is negative");
    const INT nrow_alloced = this->nrow_alloc[cell];
    const INT nrow_existing = this->nrow[cell];

    if (nrow_required != nrow_existing) {
      if (nrow_required > nrow_alloced) {
        for (int colx = 0; colx < this->ncol; colx++) {
          T *col_ptr_old = this->h_ptr_cols[cell * this->ncol + colx];
          this->stack_ptrs.push(col_ptr_old);
          T *col_ptr_new =
              sycl::malloc_device<T>(nrow_required, this->sycl_target.queue);

          if (nrow_alloced > 0) {
            this->stack_events.push(this->sycl_target.queue.memcpy(
                col_ptr_new, col_ptr_old, nrow_existing * sizeof(T)));
          }
          this->h_ptr_cols[cell * this->ncol + colx] = col_ptr_new;
        }
        this->nrow_alloc[cell] = nrow_required;
        this->stack_events.push(sycl_target.queue.memcpy(
            this->h_ptr_cells[cell], &this->h_ptr_cols[cell * this->ncol],
            this->ncol * sizeof(T *)));
      }
      this->nrow[cell] = nrow_required;
      this->nrow_max = -1;
    }
    sycl_target.profile_map.inc("CellDat", "set_nrow", 1,
                                profile_elapsed(t0, profile_timestamp()));
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
        sycl::free(ptr, this->sycl_target.queue);
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
   * Get the contents of a provided cell on the host as a CellData instance.
   *
   * @param cell Cell to get data from.
   * @returns Cell contents of specified cell as CellData instance.
   */
  inline CellData<T> get_cell(const int cell) {

    auto cell_data = std::make_shared<CellDataT<T>>(
        this->sycl_target, this->nrow[cell], this->ncol);
    for (int colx = 0; colx < this->ncol; colx++) {
      this->sycl_target.queue.memcpy(cell_data->data[colx].data(),
                                     this->h_ptr_cols[cell * this->ncol + colx],
                                     this->nrow[cell] * sizeof(T));
    }
    this->sycl_target.queue.wait();
    return cell_data;
  }

  /**
   * Set the contents of a cell on the device using a CellData instance.
   *
   * @param cell Cell index to set data in.
   * @param cell_data New cell data to set.
   */
  inline void set_cell(const int cell, CellData<T> cell_data) {
    NESOASSERT(cell_data->nrow >= this->nrow[cell],
               "CellData as insuffient row count.");
    NESOASSERT(cell_data->ncol >= this->ncol,
               "CellData as insuffient column count.");

    if (this->nrow[cell] > 0) {
      for (int colx = 0; colx < this->ncol; colx++) {

        this->sycl_target.queue.memcpy(
            this->h_ptr_cols[cell * this->ncol + colx],
            cell_data->data[colx].data(), this->nrow[cell] * sizeof(T));
      }
      this->sycl_target.queue.wait();
    }
  }

  /**
   * Get the root device pointer for the data storage. Data can be accessed
   * on the device in SYCL kernels with access like:
   *    d[cell_index][column_index][row_index]
   *
   * @returns Device pointer that can be used to access the underlying data.
   */
  T ***device_ptr() { return this->d_ptr; };

  /**
   * Get the device pointer for a column in a cell.
   *
   * @param cell Cell index to get pointer for.
   * @param col Column in cell to get pointer for.
   * @returns Device pointer to data for the specified column.
   */
  T *col_device_ptr(const int cell, const int col) {
    return this->h_ptr_cols[cell * this->ncol + col];
  }

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
