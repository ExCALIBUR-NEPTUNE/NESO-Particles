#ifndef _NESO_PARTICLES_ND_HOST_ARRAY_H_
#define _NESO_PARTICLES_ND_HOST_ARRAY_H_

#include "compute_target.hpp"
#include "containers/nd_index.hpp"
#include <cstring>

namespace NESO::Particles {

/**
 * Generic N-Dimensional array type on the host allocated in pinned memory.
 */
template <typename T, std::size_t N> class NDHostArray {
protected:
  std::vector<T, HostAllocator<T>> h_buffer;

public:
  ~NDHostArray() = default;
  NDHostArray() = delete;

  /// The compute device which the array is notionally in pinned memory for.
  SYCLTargetSharedPtr sycl_target = nullptr;

  /// Indexing function for the array.
  NDIndex<N> index;

  /**
   * Create a NDHostArray on a compute device with a given shape.
   *
   * @param sycl_target Compute device to create local array on.
   * @param index NDIndex that describes the extent of all the dimensions.
   */
  NDHostArray(SYCLTargetSharedPtr sycl_target, NDIndex<N> index)
      : h_buffer(std::vector<T, HostAllocator<T>>(
            index.size(), HostAllocator<int>(sycl_target->queue))),
        sycl_target(sycl_target), index(index) {
    this->fill(T());
  }

  /**
   * Create a NDHostArray on a compute device with a given shape.
   *
   * @param sycl_target Compute device to create local array on.
   * @param shape Parameter pack of size N which defines the extent of the
   * array in each of the N dimensions.
   */
  template <typename... SHAPE>
  NDHostArray(SYCLTargetSharedPtr sycl_target, SHAPE... shape)
      : NDHostArray(sycl_target, nd_index<N>(shape...)) {
    static_assert(sizeof...(shape) == N, "Missmatch between shape size and N.");
  }

  /**
   * @returns Pointer to underlying data. Data is linearised slowest to fastest.
   */
  inline T *ptr() { return this->h_buffer.data(); }

  /**
   *  Fill the array with a value.
   *
   *  @param value Value to fill the array with.
   */
  inline void fill(const T value) {
    std::fill(this->h_buffer.begin(), this->h_buffer.end(), value);
  }

  /**
   * Copy the entries from another NDHostArray into this array.
   *
   * @param nd_host_array Another NDHostArray of the same size.
   */
  inline void set(std::shared_ptr<NDHostArray<T, N>> nd_host_array) {
    NESOASSERT(this->index == nd_host_array->index,
               "NDHostArray size missmatch.");
    const std::size_t size = this->index.size() * sizeof(T);
    std::memcpy(this->h_buffer.data(), nd_host_array->h_buffer.data(), size);
  }

  /**
   * Copy the entries from a std::vector. Index is linearised slowest to
   * fastest.
   *
   * @param std_vector Vector to copy entries from.
   */
  inline void set(const std::vector<T> &std_vector) {

    const std::size_t num_elements = this->index.size();
    NESOASSERT(num_elements == static_cast<std::size_t>(this->index.size()),
               "Size missmatch.");
    const std::size_t size = num_elements * sizeof(T);
    std::memcpy(this->h_buffer.data(), std_vector.data(), size);
  }

  /**
   * Copy the entries into another NDHostArray. If the passed NDHostArray is a
   * nullptr then a new array will be created.
   *
   * @param nd_host_array NDHostArray to copy entries into.
   */
  inline void get(std::shared_ptr<NDHostArray<T, N>> &nd_host_array) {

    if (nd_host_array == nullptr) {
      nd_host_array =
          std::make_shared<NDHostArray<T, N>>(this->sycl_target, this->index);
    }

    NESOASSERT(this->index == nd_host_array->index,
               "NDHostArray size missmatch.");
    const std::size_t size = this->index.size() * sizeof(T);
    std::memcpy(nd_host_array->h_buffer.data(), this->h_buffer.data(), size);
  }

  /**
   * Copy the entries into a std::vector. Index is linearised slowest to
   * fastest. Output vector is resized if the vector is not the same size as the
   * NDHostArray.
   *
   * @param std_vector Vector to copy entries into. Resized if not the same size
   * as the NDHostArray.
   */
  inline void get(std::vector<T> &std_vector) {

    const std::size_t num_elements = this->index.size();
    const std::size_t size = num_elements * sizeof(T);

    if (num_elements != std_vector.size()) {
      std_vector.resize(num_elements);
    }

    std::memcpy(std_vector.data(), this->h_buffer.data(), size);
  }

  /**
   * Access array at entry.
   *
   * @param index Index of element to access.
   */
  template <typename... INDEX> T &at(INDEX... index) {
    return this->h_buffer[this->index.get_linear_index(index...)];
  }
};

extern template class NDHostArray<REAL, 2>;
extern template class NDHostArray<INT, 2>;
extern template class NDHostArray<int, 2>;
extern template class NDHostArray<REAL, 3>;
extern template class NDHostArray<INT, 3>;
extern template class NDHostArray<int, 3>;

template <typename T, std::size_t N>
using NDHostArraySharedPtr = std::shared_ptr<NDHostArray<T, N>>;

/**
 * Helper function to create a new ND array from a set of dimension extents.
 *
 *  @param sycl_target Compute device, used to pin memory in host memory.
 *  @param index NDIndex of dimensions.
 */
template <typename T, std::size_t N>
inline NDHostArraySharedPtr<T, N> nd_host_array(SYCLTargetSharedPtr sycl_target,
                                                NDIndex<N> index) {
  return std::make_shared<NDHostArray<T, N>>(sycl_target, index);
}

/**
 * Helper function to create a new ND array from a set of dimension extents.
 *
 * @param sycl_target Compute device, used to pin memory in host memory.
 * @param shape Parameter pack of size N which defines the extent of the
 * array in each of the N dimensions.
 */
template <typename T, std::size_t N, typename... SHAPE>
inline NDHostArraySharedPtr<T, N> nd_host_array(SYCLTargetSharedPtr sycl_target,
                                                SHAPE... shape) {
  return nd_host_array<T, N>(sycl_target, nd_index<N>(shape...));
}

} // namespace NESO::Particles

#endif
