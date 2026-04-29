#ifndef _NESO_PARTICLES_CONTAINERS_INDEX_MAP_HPP_
#define _NESO_PARTICLES_CONTAINERS_INDEX_MAP_HPP_

#include "../algorithms/common.hpp"
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

  // Total number of entries.
  INT total_num_values{0};

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
   * Convert linear index an array index. Note this implementation is not a
   * particullary efficient (integer division) piece of code. Avoid calling
   * anywhere performance critical.
   *
   * @param[in] key_linear Linear key to convert.
   * @param[in, out] key_array Output array key.
   */
  inline void get_array_index(const INT key_linear, int *key_array) {
    INT l = key_linear;
    for (int dim = KEY_DIM - 1; dim >= 0; dim--) {
      const INT dk = l % this->key_strides[dim];
      key_array[dim] = static_cast<int>(dk);
      l -= dk;
      l /= this->key_strides[dim];
    }
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
  inline int &at(const int key[KEY_DIM], const int value_index,
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
  INT total_num_keys{0};

public:
  /// Disable (implicit) copies.
  IndexMap(const IndexMap &st) = delete;
  /// Disable (implicit) copies.
  IndexMap &operator=(IndexMap const &a) = delete;

  // Compute device.
  SYCLTargetSharedPtr sycl_target;

  // Version ID for map. This member is for downstream use.
  std::int64_t version{0};

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
    INT n = 1;
    for (int dx = 0; dx < KEY_DIM; dx++) {
      n *= this->index_map_device.key_strides[dx];
    }
    this->total_num_keys = n;
    this->d_offsets->realloc_no_copy(n + 1);
    this->index_map_device.d_offsets = this->d_offsets->ptr;
  }

  /**
   * @returns Total number of keys across all dimensions.
   */
  inline INT get_num_keys() const { return total_num_keys; }

  /**
   * @returns DeviceBuffer that can be used to accumulate counts per key entry.
   * restore_tmp_buffer_num_values must be called.
   */
  inline std::shared_ptr<BufferDevice<INT>> get_tmp_buffer_num_values() {
    auto d_buffer = get_resource<BufferDevice<INT>,
                                 ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);
    d_buffer->realloc_no_copy(this->get_num_keys() + 1);
    this->sycl_target->queue
        .template fill<INT>(d_buffer->ptr, static_cast<INT>(0),
                            static_cast<std::size_t>(this->get_num_keys() + 1))
        .wait_and_throw();
    return d_buffer;
  }

  /**
   * @param buffer DeviceBuffer that can be used to accumulate counts per key
   * entry.
   */
  inline void
  restore_tmp_buffer_num_values(std::shared_ptr<BufferDevice<INT>> &buffer) {
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, buffer);
  }

  /**
   * Populuate the offsets buffer using the provided number of values counts.
   * Reallocates the values buffer.
   *
   * @param d_num_values Device buffer, of size number of keys plus one, holding
   * the number of values for each key.
   */
  inline void populate_offsets_buffer(INT *d_num_values) {

    joint_exclusive_scan_blocking(
        this->sycl_target, static_cast<std::size_t>(this->get_num_keys() + 1),
        d_num_values, this->d_offsets->ptr);

    INT total_num_values = 0;
    sycl_target->queue
        .memcpy(&total_num_values, this->d_offsets->ptr + this->get_num_keys(),
                sizeof(INT))
        .wait_and_throw();

    this->d_values->realloc_no_copy(total_num_values * VALUE_DIM);
    for (int dx = 0; dx < VALUE_DIM; dx++) {
      this->index_map_device.d_values[dx] =
          this->d_values->ptr + dx * total_num_values;
    }
    this->index_map_device.total_num_values = total_num_values;
  }

  /**
   * @returns Device type for map.
   */
  inline IndexMapDevice<KEY_DIM, VALUE_DIM> get_device() {
    return this->index_map_device;
  }

  /**
   * @returns Host copy of stored map.
   */
  inline std::map<std::array<int, KEY_DIM>,
                  std::array<std::vector<int>, VALUE_DIM>>
  get_values() {

    std::map<std::array<int, KEY_DIM>, std::array<std::vector<int>, VALUE_DIM>>
        return_values;

    IndexMapDevice index_map_device = this->index_map_device;

    const INT total_num_values = index_map_device.total_num_values;

    std::vector<int, HostAllocator<int>> h_values(
        total_num_values * VALUE_DIM,
        HostAllocator<int>{this->sycl_target->queue});

    std::vector<INT, HostAllocator<INT>> h_offsets(
        this->get_num_keys() + 1, HostAllocator<INT>{this->sycl_target->queue});

    EventStack es;

    for (int dx = 0; dx < VALUE_DIM; dx++) {
      es.push(this->sycl_target->queue.memcpy(
          h_values.data() + total_num_values * dx,
          this->index_map_device.d_values[dx], total_num_values * sizeof(int)));
    }

    es.push(this->sycl_target->queue.memcpy(
        h_offsets.data(), this->index_map_device.d_offsets,
        (this->get_num_keys() + 1) * sizeof(INT)));

    es.wait();

    index_map_device.d_offsets = h_offsets.data();
    for (int dx = 0; dx < VALUE_DIM; dx++) {
      index_map_device.d_values[dx] = h_values.data() + total_num_values * dx;
    }

    const INT total_num_keys = this->get_num_keys();
    for (INT ex = 0; ex < total_num_keys; ex++) {
      std::array<int, KEY_DIM> key;
      index_map_device.get_array_index(ex, key.data());
      const int num_values = index_map_device.get_num_values(key.data());
      const INT offset = index_map_device.get_offset(key.data());

      for (int dx = 0; dx < VALUE_DIM; dx++) {
        return_values[key][dx].resize(num_values);
        std::memcpy(return_values.at(key).at(dx).data(),
                    index_map_device.d_values[dx] + offset,
                    num_values * sizeof(int));
      }
    }

    return return_values;
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
