#ifndef _NESO_PARTICLES_CELL_DAT_CONST_H_
#define _NESO_PARTICLES_CELL_DAT_CONST_H_

#include "../loop/particle_loop_base.hpp"
#include "cell_data.hpp"

namespace NESO::Particles {

template <typename T> class CellDatConst;
template <typename T>
using CellDatConstSharedPtr = std::shared_ptr<CellDatConst<T>>;

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

template <typename T, typename OP> struct CellDatConstDeviceTypeReduction {
  T *ptr;
  int ncol;
  int nrow;
  OP binop;
  sycl::local_accessor<T, 1> la;
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
    return atomic_fetch_add(&ptr[nrow * col + row], value);
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
    return atomic_fetch_min(&ptr[nrow * col + row], value);
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
    return atomic_fetch_max(&ptr[nrow * col + row], value);
  }
};

/**
 * Kernel type for Reduction for CellDatConst.
 */
template <typename T, typename OP> struct Reduction {
  Reduction() = default;
  int local_sycl_index;
  int local_sycl_range;
  T *ptr;
  int nrow;
  OP binop;

  /**
   * Reduce the passed value into the provided index.
   *
   * @param row Row of CellDatConst to combine into.
   * @param col Column of CellDatConst to combine into.
   * @param value Value to reduce.
   */
  inline void combine(const int row, const int col, const T value) {
    const int component_linear_index = nrow * col + row;
    const int index =
        component_linear_index * local_sycl_range + local_sycl_index;

    const T current = ptr[index];
    ptr[index] = binop(current, value);
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
 *  Loop parameter for reduction access of a CellDatConst.
 */
template <typename T, typename OP>
struct LoopParameter<Access::Reduction<CellDatConst<T>, OP>> {
  using type = CellDatConstDeviceTypeReduction<T, OP>;
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
 *  KernelParameter type for reduction access to a CellDatConst.
 */
template <typename T, typename OP>
struct KernelParameter<Access::Reduction<CellDatConst<T>, OP>> {
  using type = Access::CellDatConst::Reduction<T, OP>;
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
 *  Function to create the kernel argument for CellDatConst reduction access.
 */
template <typename T, typename OP>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              CellDatConstDeviceTypeReduction<T, OP> &rhs,
                              Access::CellDatConst::Reduction<T, OP> &lhs) {
  lhs.local_sycl_index = static_cast<int>(iterationx.local_sycl_index);
  lhs.local_sycl_range = static_cast<int>(iterationx.local_sycl_range);
  lhs.ptr = &rhs.la[0];
  lhs.nrow = rhs.nrow;
  lhs.binop = rhs.binop;
}

/**
 * Method to compute access to a CellDatConst (read)
 */
template <typename T>
inline CellDatConstDeviceTypeConst<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<CellDatConst<T> *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a CellDatConst (write)
 */
template <typename T>
inline CellDatConstDeviceType<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<CellDatConst<T> *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a CellDatConst (add)
 */
template <typename T>
inline CellDatConstDeviceType<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Add<CellDatConst<T> *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a CellDatConst (min)
 */
template <typename T>
inline CellDatConstDeviceType<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Min<CellDatConst<T> *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a CellDatConst (max)
 */
template <typename T>
inline CellDatConstDeviceType<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Max<CellDatConst<T> *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a CellDatConst (reduction)
 */
template <typename T, typename OP>
inline CellDatConstDeviceTypeReduction<T, OP>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Reduction<std::shared_ptr<CellDatConst<T>>, OP> &a) {
  auto rhs = a.obj->impl_get();
  CellDatConstDeviceTypeReduction<T, OP> lhs;
  lhs.ptr = rhs.ptr;
  lhs.ncol = a.obj->ncol;
  lhs.nrow = a.obj->nrow;
  lhs.binop = a.binop;
  // The local memory is typed, hence this size does not have sizeof(T).
  const std::size_t size = a.obj->ncol * a.obj->nrow * global_info->local_size;
  lhs.la = sycl::local_accessor<T, 1>(sycl::range<1>(size), cgh);
  return lhs;
}

/**
 * Indicate that Reduction over CellDatConst requires local memory.
 */
template <template <typename> typename T, typename U, typename OP>
inline std::size_t
get_required_local_num_bytes(Access::Reduction<std::shared_ptr<T<U>>, OP> &a) {
  return sizeof(U) * a.obj->nrow * a.obj->ncol;
}

/**
 * The Reduction ParticleLoop implementations will call reduction_initialise for
 * all SYCL work items before the kernel is launched.
 */
template <typename T, typename OP>
inline void reduction_initialise(sycl::nd_item<2> &,
                                 ParticleLoopIteration &iterationx,
                                 CellDatConstDeviceTypeReduction<T, OP> &a) {

  const T initial_value = Kernel::get_identity(a.binop);
  const int stride = static_cast<int>(a.nrow * a.ncol);
  T *ptr = &a.la[0];
  const auto local_sycl_range = iterationx.local_sycl_range;
  const auto local_sycl_index = iterationx.local_sycl_index;
  for (int ix = 0; ix < stride; ix++) {
    ptr[ix * local_sycl_range + local_sycl_index] = initial_value;
  }
}

/**
 * The Reduction ParticleLoop implementations will call reduction_finalise for
 * all SYCL work items after kernels have launched.
 */
template <typename T, typename OP>
inline void reduction_finalise(sycl::nd_item<2> &idx,
                               ParticleLoopIteration &iterationx,
                               CellDatConstDeviceTypeReduction<T, OP> &a) {
  const int nrow = a.nrow;
  const int ncol = a.ncol;
  T *ptr = &a.la[0];
  const auto local_sycl_range = iterationx.local_sycl_range;
  const auto local_sycl_index = iterationx.local_sycl_index;
  const auto &binop = a.binop;
  const int half_sycl_range = local_sycl_range / 2;
  const int num_elements = nrow * ncol;

  for (int ex = 0; ex < num_elements; ex++) {
    const int offset = ex * local_sycl_range;
    for (unsigned int s = half_sycl_range; s > 0; s >>= 1) {
      if (local_sycl_index < s) {
        const T current = ptr[local_sycl_index + offset];
        ptr[local_sycl_index + offset] =
            binop(current, ptr[local_sycl_index + offset + s]);
      }
      idx.barrier(sycl::access::fence_space::local_space);
    }
    if (local_sycl_index == 0) {
      T *d_ptr = a.ptr + iterationx.cellx * num_elements + ex;
      Kernel::atomic_reduce(binop, d_ptr, ptr[offset]);
    }

    // ACPP omp.accelerated seems to not generate the correct loops if this
    // barrier is missing.
    idx.barrier(sycl::access::fence_space::local_space);
  }
}

/**
 * Sanity checkes at runtime that the reductions do actually work with the SYCL
 * backend and compiler flags.
 */
template <typename T, typename OP>
inline void
pre_loop(ParticleLoopGlobalInfo *global_info,
         Access::Reduction<std::shared_ptr<CellDatConst<T>>, OP> &a) {
  auto sycl_target = a.obj->sycl_target;
  const auto local_size = global_info->local_size;

  NESOASSERT(is_power_of_two(global_info->local_size),
             "Local size is not a power of two.");

  const auto nrow = a.obj->nrow;
  const auto ncol = a.obj->ncol;
  const auto num_elements_total = nrow * ncol * local_size;

  const std::vector<std::size_t> validation_key = {
      typeid(CellDatConst<T>).hash_code(),
      typeid(T).hash_code(),
      typeid(a.binop).hash_code(),
      static_cast<std::size_t>(nrow),
      static_cast<std::size_t>(ncol),
      static_cast<std::size_t>(local_size)};

  if (!sycl_target->device_limits.validated_types.count(validation_key) &&
      get_env_size_t("NESO_PARTICLES_ENABLE_REDUCTION_SELF_TEST", 1)) {

    auto dh_to_test = get_resource<BufferDeviceHost<T>,
                                   ResourceStackInterfaceBufferDeviceHost<T>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDeviceHost<T>{},
        sycl_target);
    dh_to_test->realloc_no_copy(nrow * ncol);
    sycl_target->queue.fill((T *)dh_to_test->d_buffer.ptr, (T)0, nrow * ncol)
        .wait_and_throw();

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          CellDatConstDeviceTypeReduction<T, OP> loop_arg;
          T *d_ptr = dh_to_test->d_buffer.ptr;
          loop_arg.ptr = d_ptr;
          loop_arg.ncol = ncol;
          loop_arg.nrow = nrow;
          loop_arg.binop = a.binop;
          loop_arg.la = sycl::local_accessor<T, 1>(
              sycl::range<1>(num_elements_total), cgh);
          cgh.parallel_for<>(
              sycl_target->device_limits.validate_nd_range(
                  sycl::nd_range<2>(sycl::range<2>(1, local_size),
                                    sycl::range<2>(1, local_size))),
              [=](sycl::nd_item<2> idx) {
                CellDatConstDeviceTypeReduction<T, OP> loop_arg_kernel =
                    loop_arg;

                ParticleLoopIteration iterationx;
                const auto local_sycl_index = idx.get_local_id(1);
                const auto local_sycl_range = idx.get_local_range(1);
                iterationx.local_sycl_index = local_sycl_index;
                iterationx.local_sycl_range = local_sycl_range;
                iterationx.cellx = 0;
                iterationx.layerx = 0;
                iterationx.loop_layerx = 0;

                for (int ex = 0; ex < nrow * ncol; ex++) {
                  const auto index = ex * local_sycl_range + local_sycl_index;
                  const T value = static_cast<T>(ex * local_sycl_index) + 1;
                  loop_arg.la[index] = value;
                }
                idx.barrier(sycl::access::fence_space::local_space);
                reduction_finalise<T, OP>(idx, iterationx, loop_arg_kernel);
              });
        })
        .wait_and_throw();

    dh_to_test->device_to_host();

    for (int cx = 0; cx < ncol; cx++) {
      for (int rx = 0; rx < nrow; rx++) {
        T correct = static_cast<T>(0);
        const auto linear_index = rx + nrow * cx;
        for (int idx = 0; idx < static_cast<int>(local_size); idx++) {
          correct = a.binop(correct, linear_index * idx + 1);
        }
        auto to_test = dh_to_test->h_buffer.ptr[linear_index];
        const REAL err_abs = std::abs(correct - to_test);
        const REAL err_rel =
            std::abs(correct) > 0 ? err_abs / std::abs(correct) : err_abs;
        NESOASSERT(err_rel < 1.0e-6,
                   "CellDatConst Reduction self test failed. This may be an "
                   "indication of a bug or the SYCL implementation. The "
                   "detected error has size: " +
                       std::to_string(err_rel));
      }
    }

    sycl_target->device_limits.validated_types.insert(validation_key);

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDeviceHost<T>{}, dh_to_test);
  }
}

} // namespace ParticleLoopImplementation

namespace Private {
template <typename T>
inline CellDatConstDeviceType<T>
cell_dat_const_impl_get(CellDatConstSharedPtr<T> cell_dat_const);
}

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

  template <typename U, typename OP>
  friend CellDatConstDeviceTypeReduction<U, OP>
  ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh,
      Access::Reduction<std::shared_ptr<CellDatConst<U>>, OP> &a);

  template <typename U>
  friend CellDatConstDeviceType<U>
  Private::cell_dat_const_impl_get(CellDatConstSharedPtr<U> cell_dat_const);

protected:
  T *d_ptr;
  const int stride;

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
    if (nrow && ncol) {
      this->sycl_target->queue.fill(this->d_ptr, value, ncells * nrow * ncol)
          .wait_and_throw();
    }
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
      : stride(nrow * ncol), sycl_target(sycl_target), ncells(ncells),
        nrow(nrow), ncol(ncol) {
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
    if (this->nrow > 0 && this->ncol > 0) {
      this->sycl_target->queue
          .memcpy(cell_data->get_column_ptr(0),
                  &this->d_ptr[cell * this->stride],
                  this->nrow * this->ncol * sizeof(T))
          .wait_and_throw();
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

    if (this->nrow > 0 && this->ncol > 0) {
      this->sycl_target->queue
          .memcpy(&this->d_ptr[cell * this->stride],
                  cell_data->get_column_ptr(0),
                  this->nrow * this->ncol * sizeof(T))
          .wait_and_throw();
    }
  }

  /**
   * Get the cell data for all cells.
   *
   * @returns CellData instances for all cells.
   */
  inline std::vector<CellData<T>> get_all_cells() {
    const auto num_elements = this->stride * this->ncells;

    auto tmp_buffer = get_resource<BufferDeviceHost<T>,
                                   ResourceStackInterfaceBufferDeviceHost<T>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDeviceHost<T>{},
        sycl_target);
    tmp_buffer->realloc_no_copy(num_elements);
    auto ptr = tmp_buffer->h_buffer.ptr;

    sycl::event copy_event;

    if (this->stride > 0) {
      // start the copy into the host temporary
      copy_event = this->sycl_target->queue.memcpy(ptr, this->d_ptr,
                                                   num_elements * sizeof(T));
    }
    // create the buffers to return
    std::vector<CellData<T>> return_vector;
    return_vector.reserve(this->ncells);
    for (int cellx = 0; cellx < this->ncells; cellx++) {
      return_vector.push_back(std::make_shared<CellDataT<T>>(
          this->sycl_target, this->nrow, this->ncol));
    }
    //
    //// wait for the copy event from the device
    copy_event.wait_and_throw();
    //
    if (this->stride > 0) {
      for (int cellx = 0; cellx < this->ncells; cellx++) {
        auto inner_ptr = return_vector[cellx]->get_column_ptr(0);
        std::memcpy(inner_ptr, ptr + this->stride * cellx,
                    this->stride * sizeof(T));
      }
    }

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDeviceHost<T>{}, tmp_buffer);
    return return_vector;
  }

  /**
   * Set the cell data for all cells.
   *
   * @param cell_data Vector of CellData instances each of size nrow, ncol.
   */
  inline void set_all_cells(const std::vector<CellData<T>> &cell_data) {
    auto tmp_buffer = get_resource<BufferDeviceHost<T>,
                                   ResourceStackInterfaceBufferDeviceHost<T>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDeviceHost<T>{},
        sycl_target);

    const auto num_elements = this->stride * this->ncells;
    tmp_buffer->realloc_no_copy(num_elements);
    auto ptr = tmp_buffer->h_buffer.ptr;

    NESOASSERT(cell_data.size() == static_cast<std::size_t>(this->ncells),
               "Bad length of cell_data.");

    for (int cellx = 0; cellx < this->ncells; cellx++) {
      NESOASSERT(cell_data[cellx]->nrow == this->nrow, "Bad number of rows.");
      NESOASSERT(cell_data[cellx]->ncol == this->ncol,
                 "Bad number of columns.");

      std::memcpy(ptr + this->stride * cellx,
                  cell_data[cellx]->get_column_ptr(0),
                  this->stride * sizeof(T));
    }

    this->sycl_target->queue.memcpy(this->d_ptr, ptr, num_elements * sizeof(T))
        .wait_and_throw();

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDeviceHost<T>{}, tmp_buffer);
    NESOASSERT(tmp_buffer == nullptr, "Expected nullptr.");
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

extern template class CellDatConst<REAL>;
extern template class CellDatConst<INT>;
extern template class CellDatConst<int>;

} // namespace NESO::Particles

#endif
