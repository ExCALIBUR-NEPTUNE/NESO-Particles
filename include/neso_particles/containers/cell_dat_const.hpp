#ifndef _NESO_PARTICLES_CELL_DAT_CONST_H_
#define _NESO_PARTICLES_CELL_DAT_CONST_H_

#include "../loop/particle_loop_base.hpp"
#include "cell_data.hpp"

namespace NESO::Particles {

/**
 *  Type the implementation methods return;
 */
template <typename T> struct CellDatConstDeviceType {
  T *ptr;
  int stride;
  int nrow;
};

template <typename T> struct CellDatConstDeviceTypeConst {
  T const *ptr;
  int stride;
  int nrow;
};

template <typename T> class CellDatConst;

/**
 *  Defines the access implementations and types for CellDatConst objects.
 */
namespace Access::CellDatConst {

/**
 * Access:CellDatConst::Read<T> is a kernel argument type for accessing
 * CellDatConst data in a kernel.
 */
template <typename T> struct Read {
  /// Pointer to underlying data for the array.
  Read() = default;
  T const *ptr;
  int nrow;
  inline const T at(const int row, const int col) {
    return ptr[nrow * col + row];
  }
  inline const T &operator[](const int component) { return ptr[component]; }
};

/**
 * Access:CellDatConst::Write<T> is a kernel argument type for accessing
 * CellDatConst data in a kernel.
 */
template <typename T> struct Write {
  /// Pointer to underlying data for the array.
  Write() = default;
  T *ptr;
  int nrow;
  inline T &at(const int row, const int col) { return ptr[nrow * col + row]; }
  inline T &operator[](const int component) { return ptr[component]; }
};

/**
 * Access:CellDatConst::Add<T> is a kernel argument type for accessing
 * CellDatConst data in a kernel.
 */
template <typename T> struct Add {
  /// Pointer to underlying data for the array.
  Add() = default;
  T *ptr;
  int nrow;
  inline T fetch_add(const int row, const int col, const T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        element_atomic(ptr[nrow * col + row]);
    return element_atomic.fetch_add(value);
  }
};

/**
 * Access:CellDatConst::Min<T> is a kernel argument type for accessing
 * CellDatConst data in a kernel.
 */
template <typename T> struct Min {
  /// Pointer to underlying data for the array.
  Min() = default;
  T *ptr;
  int nrow;
  inline T fetch_min(const int row, const int col, const T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        element_atomic(ptr[nrow * col + row]);
    return element_atomic.fetch_min(value);
  }
};

/**
 * Access:CellDatConst::Max<T> is a kernel argument type for accessing
 * CellDatConst data in a kernel.
 */
template <typename T> struct Max {
  /// Pointer to underlying data for the array.
  Max() = default;
  T *ptr;
  int nrow;
  inline T fetch_max(const int row, const int col, const T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        element_atomic(ptr[nrow * col + row]);
    return element_atomic.fetch_max(value);
  }
};

} // namespace Access::CellDatConst

namespace ParticleLoopImplementation {

/**
 *  Loop parameter for read access of a CellDatConst.
 */
template <typename T> struct LoopParameter<Access::Read<CellDatConst<T>>> {
  using type = CellDatConstDeviceTypeConst<T>;
};
/**
 *  Loop parameter for write access of a CellDatConst.
 */
template <typename T> struct LoopParameter<Access::Write<CellDatConst<T>>> {
  using type = CellDatConstDeviceType<T>;
};
/**
 *  Loop parameter for add access of a CellDatConst.
 */
template <typename T> struct LoopParameter<Access::Add<CellDatConst<T>>> {
  using type = CellDatConstDeviceType<T>;
};
/**
 *  Loop parameter for min access of a CellDatConst.
 */
template <typename T> struct LoopParameter<Access::Min<CellDatConst<T>>> {
  using type = CellDatConstDeviceType<T>;
};
/**
 *  Loop parameter for max access of a CellDatConst.
 */
template <typename T> struct LoopParameter<Access::Max<CellDatConst<T>>> {
  using type = CellDatConstDeviceType<T>;
};

/**
 *  KernelParameter type for read access to a CellDatConst.
 */
template <typename T> struct KernelParameter<Access::Read<CellDatConst<T>>> {
  using type = Access::CellDatConst::Read<T>;
};
/**
 *  KernelParameter type for write access to a CellDatConst.
 */
template <typename T> struct KernelParameter<Access::Write<CellDatConst<T>>> {
  using type = Access::CellDatConst::Write<T>;
};
/**
 *  KernelParameter type for add access to a CellDatConst.
 */
template <typename T> struct KernelParameter<Access::Add<CellDatConst<T>>> {
  using type = Access::CellDatConst::Add<T>;
};
/**
 *  KernelParameter type for min access to a CellDatConst.
 */
template <typename T> struct KernelParameter<Access::Min<CellDatConst<T>>> {
  using type = Access::CellDatConst::Min<T>;
};
/**
 *  KernelParameter type for max access to a CellDatConst.
 */
template <typename T> struct KernelParameter<Access::Max<CellDatConst<T>>> {
  using type = Access::CellDatConst::Max<T>;
};

/**
 *  Function to create the kernel argument for CellDatConst read access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              CellDatConstDeviceTypeConst<T> &rhs,
                              Access::CellDatConst::Read<T> &lhs) {
  T const *ptr = rhs.ptr + iterationx.cellx * rhs.stride;
  lhs.ptr = ptr;
  lhs.nrow = rhs.nrow;
}
/**
 *  Function to create the kernel argument for CellDatConst write access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              CellDatConstDeviceType<T> &rhs,
                              Access::CellDatConst::Write<T> &lhs) {
  T *ptr = rhs.ptr + iterationx.cellx * rhs.stride;
  lhs.ptr = ptr;
  lhs.nrow = rhs.nrow;
}
/**
 *  Function to create the kernel argument for CellDatConst add access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              CellDatConstDeviceType<T> &rhs,
                              Access::CellDatConst::Add<T> &lhs) {
  T *ptr = rhs.ptr + iterationx.cellx * rhs.stride;
  lhs.ptr = ptr;
  lhs.nrow = rhs.nrow;
}
/**
 *  Function to create the kernel argument for CellDatConst min access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              CellDatConstDeviceType<T> &rhs,
                              Access::CellDatConst::Min<T> &lhs) {
  T *ptr = rhs.ptr + iterationx.cellx * rhs.stride;
  lhs.ptr = ptr;
  lhs.nrow = rhs.nrow;
}
/**
 *  Function to create the kernel argument for CellDatConst max access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              CellDatConstDeviceType<T> &rhs,
                              Access::CellDatConst::Max<T> &lhs) {
  T *ptr = rhs.ptr + iterationx.cellx * rhs.stride;
  lhs.ptr = ptr;
  lhs.nrow = rhs.nrow;
}

/**
 * Method to compute access to a CellDatConst (read)
 */
template <typename T>
inline CellDatConstDeviceTypeConst<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info, [[maybe_unused]] sycl::handler &cgh,
                Access::Read<CellDatConst<T> *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a CellDatConst (write)
 */
template <typename T>
inline CellDatConstDeviceType<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info, [[maybe_unused]] sycl::handler &cgh,
                Access::Write<CellDatConst<T> *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a CellDatConst (add)
 */
template <typename T>
inline CellDatConstDeviceType<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info, [[maybe_unused]] sycl::handler &cgh,
                Access::Add<CellDatConst<T> *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a CellDatConst (min)
 */
template <typename T>
inline CellDatConstDeviceType<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info, [[maybe_unused]] sycl::handler &cgh,
                Access::Min<CellDatConst<T> *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a CellDatConst (max)
 */
template <typename T>
inline CellDatConstDeviceType<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info, [[maybe_unused]] sycl::handler &cgh,
                Access::Max<CellDatConst<T> *> &a) {
  return a.obj->impl_get();
}

} // namespace ParticleLoopImplementation

/**
 *  Container that allocates on the device a matrix of fixed size nrow X ncol
 *  for N cells. Data stored in column major format. i.e. Data order from
 *  slowest to fastest is: cell, column, row.
 */
template <typename T> class CellDatConst {
  template <typename KERNEL, typename... ARGS> friend class ParticleLoop;
  friend CellDatConstDeviceTypeConst<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<CellDatConst<T> *> &a);
  friend CellDatConstDeviceType<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<CellDatConst<T> *> &a);
  friend CellDatConstDeviceType<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Add<CellDatConst<T> *> &a);
  friend CellDatConstDeviceType<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Min<CellDatConst<T> *> &a);
  friend CellDatConstDeviceType<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Max<CellDatConst<T> *> &a);

private:
  T *d_ptr;
  const int stride;

protected:
  /**
   * Non-const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline CellDatConstDeviceType<T> impl_get() {
    static_assert(
        std::is_trivially_copyable<CellDatConstDeviceType<T>>::value == true);
    return {this->d_ptr, this->stride, this->nrow};
  }

  /**
   * Const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline CellDatConstDeviceTypeConst<T> impl_get_const() {
    static_assert(
        std::is_trivially_copyable<CellDatConstDeviceTypeConst<T>>::value ==
        true);
    return {this->d_ptr, this->stride, this->nrow};
  }

public:
  /// Disable (implicit) copies.
  CellDatConst(const CellDatConst &st) = delete;
  /// Disable (implicit) copies.
  CellDatConst &operator=(CellDatConst const &a) = delete;

  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  /// Number of cells, labeled 0,...,N-1.
  const int ncells;
  /// Number of rows in each cell.
  const int nrow;
  /// Number of columns in each cell.
  const int ncol;
  ~CellDatConst() { sycl::free(this->d_ptr, sycl_target->queue); };

  /**
   * Fill all the entries with a given value.
   *
   * @param value Value to place in all entries.
   */
  inline void fill(const T value) {
    this->sycl_target->queue.fill(this->d_ptr, value, ncells * nrow * ncol)
        .wait_and_throw();
  }

  /**
   * Create new CellDatConst on the specified compute target with a fixed cell
   * count, fixed number of rows per cell and fixed number of columns per cell.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param ncells Number of cells.
   * @param nrow Number of rows.
   * @param ncol Number of columns.
   */
  CellDatConst(SYCLTargetSharedPtr sycl_target, const int ncells,
               const int nrow, const int ncol)
      : sycl_target(sycl_target), ncells(ncells), nrow(nrow), ncol(ncol),
        stride(nrow * ncol) {
    this->d_ptr =
        sycl::malloc_device<T>(ncells * nrow * ncol, sycl_target->queue);
    this->fill(0);
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
    if (this->nrow > 0) {
      EventStack se;
      for (int colx = 0; colx < this->ncol; colx++) {
        se.push(this->sycl_target->queue.memcpy(
            cell_data->data[colx].data(),
            &this->d_ptr[cell * this->stride + colx * this->nrow],
            this->nrow * sizeof(T)));
      }
      se.wait();
    }
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

    if (this->nrow > 0) {
      EventStack se;
      for (int colx = 0; colx < this->ncol; colx++) {
        se.push(this->sycl_target->queue.memcpy(
            &this->d_ptr[cell * this->stride + colx * this->nrow],
            cell_data->data[colx].data(), this->nrow * sizeof(T)));
      }
      se.wait();
    }
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
        .memcpy(&value,
                &this->d_ptr[cell * this->stride + col * this->nrow + row],
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
        .memcpy(&this->d_ptr[cell * this->stride + col * this->nrow + row],
                &value, sizeof(T))
        .wait_and_throw();
  }
};

template <typename T>
using CellDatConstSharedPtr = std::shared_ptr<CellDatConst<T>>;

} // namespace NESO::Particles

#endif
