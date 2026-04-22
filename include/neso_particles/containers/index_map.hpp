#ifndef _NESO_PARTICLES_CONTAINERS_INDEX_MAP_HPP_
#define _NESO_PARTICLES_CONTAINERS_INDEX_MAP_HPP_

#include "../compute_target.hpp"
#include "../device_buffers.hpp"

namespace NESO::Particles {

/**
 * Device type for generic maps from keys to int indices. Intended use is
 * partitions such as maps from cells to particles.
 */
template <int KEY_DIM, int VALUE_DIM> struct IndexMapDevice {

  static constexpr int key_dim = KEY_DIM;
  static constexpr int value_dim = VALUE_DIM;

  // Offsets from linearised keys to start of values. This is an exclusive scan
  // of the number of values in each key. Includes the last point + 1.
  INT *d_offsets{nullptr};

  // Map entries.
  int *d_values[VALUE_DIM]{};

  // Key strides
  int key_strides[KEY_DIM]{};

  /**
   * Compute linear index from key. Key indices are ordered from slow to fast.
   *
   * @param key Key to linearise from slowes to fastest.
   * @returns Linearised key.
   */
  inline INT get_linear_index(const int key[KEY_DIM]) const {
    int index = key[0];
    for (int dim = 1; dim < KEY_DIM; dim++) {
      index *= this->key_strides[dim];
      index += key[dim];
    }
    return index;
  }

  /**
   * Get the offset in the values array to the specified key.
   *
   * @param key Key to return offset for.
   * @returns Offset in values array for the specified key.
   */
  inline INT get_offset(const int key[KEY_DIM]) const {
    const auto index = this->get_linear_index(key);
    return this->d_offsets[index];
  }

  /**
   * Get the number of values stored for the specified key.
   *
   * @param key Key to return offset for.
   * @returns Number of values stored for the specified key.
   */
  inline int get_num_values(const int key[KEY_DIM]) const {
    const auto index = this->get_linear_index(key);
    return static_cast<int>(this->d_offsets[index + 1] -
                            this->d_offsets[index]);
  }

  /**
   * Get the i-th value stored for the specified key and dimension.
   *
   * @param key Key to retrieve value from.
   * @param value_index Inner index of value to retrieve.
   * @param dimension Dimension of value to retrieve.
   * @returns Value specified by index.
   */
  inline int get_value(const int key[KEY_DIM], const int value_index,
                       const int dimension) const {
    const INT offset = this->get_offset(key) + static_cast<INT>(value_index);
    return this->d_values[dimension][offset];
  }
};

/**
 * Type for generic maps from keys to int indices. Intended use is
 * partitions such as maps from cells to particles.
 */
template <int KEY_DIM, int VALUE_DIM> class IndexMap {
protected:
  std::unique_ptr<BufferDevice<INT>> d_offsets;
  INT total_num_values_per_dim = 0;
  std::unique_ptr<BufferDevice<int>> d_values;
  IndexMapDevice<KEY_DIM, VALUE_DIM> index_map_device;
  EventStack event_stack;
  int total_num_keys{0};

public:
  /// Disable (implicit) copies.
  IndexMap(const IndexMap &st) = delete;
  /// Disable (implicit) copies.
  IndexMap &operator=(IndexMap const &a) = delete;

  // Compute device.
  SYCLTargetSharedPtr sycl_target;

  /**
   * Create an IndexMap on a device.
   *
   * @param sycl_target Compute device to create map on.
   */
  IndexMap(SYCLTargetSharedPtr sycl_target) : sycl_target(sycl_target) {
    this->d_offsets = std::make_unique<BufferDevice<INT>>(sycl_target, 64);
    this->d_values = std::make_unique<BufferDevice<int>>(sycl_target, 64);
  }

  /**
   * Set the key strides for the map. Reallocates offsets buffer based on
   * current key space size.
   *
   * @param key_strides New key strides to set.
   */
  inline void set_key_strides(const int key_strides[KEY_DIM]) {
    for (int dx = 0; dx < KEY_DIM; dx++) {
      this->index_map_device.key_strides[dx] = key_strides[dx];
    }
    int n = 1;
    for (int dx = 0; dx < KEY_DIM; dx++) {
      n *= this->index_map_device.key_strides[dx];
    }
    this->total_num_keys = n;

    NESOASSERT(false, "REALLOCATE");
  }

  /**
   * @returns Total number of keys across all dimensions.
   */
  inline int get_num_keys() const { return total_num_keys; }

  /**
   * @returns DeviceBuffer that can be used to accumulate counts per key entry.
   * restore_tmp_buffer_num_values must be called.
   */
  inline std::shared_ptr<BufferDevice<INT>> get_tmp_buffer_num_values() {
    auto d_buffer = get_resource<BufferDevice<INT>,
                                 ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);
    d_buffer->realloc_no_copy(this->get_num_keys());
    return d_buffer;
  }

  /**
   * @param buffer DeviceBuffer that can be used to accumulate counts per key
   * entry.
   */
  inline void
  restore_tmp_buffer_num_values(std::shared_ptr<BufferDevice<INT>> buffer) {
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, buffer);
  }

  /**
   * @returns Device type for map.
   */
  inline IndexMapDevice<KEY_DIM, VALUE_DIM> get_device() {
    this->event_stack.wait();
    return this->index_map_device;
  }
};

extern template class IndexMap<2, 1>;

template <int KEY_DIM, int VALUE_DIM>
using IndexMapSharedPtr = std::shared_ptr<IndexMap<KEY_DIM, VALUE_DIM>>;

/**
 * ResourceStackInterface for IndexMap.
 */
template <int KEY_DIM, int VALUE_DIM>
struct ResourceStackInterfaceIndexMap
    : ResourceStackInterface<IndexMap<KEY_DIM, VALUE_DIM>> {

  SYCLTargetSharedPtr sycl_target;
  ResourceStackInterfaceIndexMap(SYCLTargetSharedPtr sycl_target)
      : sycl_target(sycl_target) {}

  virtual inline IndexMapSharedPtr<KEY_DIM, VALUE_DIM> construct() override {
    return std::make_shared<IndexMap<KEY_DIM, VALUE_DIM>>(this->sycl_target);
  }

  virtual inline void
  free([[maybe_unused]] IndexMapSharedPtr<KEY_DIM, VALUE_DIM> &resource)
      override {
    // These buffers are freed by their destructors hence we don't need to do
    // anything here.
  }

  virtual inline void
  clean([[maybe_unused]] IndexMapSharedPtr<KEY_DIM, VALUE_DIM> &resource)
      override {}
};

/**
 * ResourceStackMap key for ResourceStackInterfaceIndexMap.
 */
template <int KEY_DIM, int VALUE_DIM> struct ResourceStackKeyIndexMap {};

/**
 * Helper wrapper to create a IndexMap. restore_index_map must be called.
 *
 * @param sycl_target Compute device for map.
 * @returns Shared pointer to IndexMap.
 */
template <int KEY_DIM, int VALUE_DIM>
IndexMapSharedPtr<KEY_DIM, VALUE_DIM>
get_index_map(SYCLTargetSharedPtr sycl_target) {

  auto p = get_resource<IndexMap<KEY_DIM, VALUE_DIM>,
                        ResourceStackInterfaceIndexMap<KEY_DIM, VALUE_DIM>>(
      sycl_target->resource_stack_map,
      ResourceStackKeyIndexMap<KEY_DIM, VALUE_DIM>{}, sycl_target);

  return p;
}

/**
 * Helper wrapper to restore IndexMapSharedPtr created by get_index_map. This
 * function must be called for IndexMaps created by get_index_map.
 *
 * @param sycl_target Compute device for map.
 * @param index_map IndexMap to restore.
 */
template <int KEY_DIM, int VALUE_DIM>
void restore_index_map(SYCLTargetSharedPtr sycl_target,
                       IndexMapSharedPtr<KEY_DIM, VALUE_DIM> &index_map) {

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyIndexMap<KEY_DIM, VALUE_DIM>{}, index_map);
}

} // namespace NESO::Particles

#endif
