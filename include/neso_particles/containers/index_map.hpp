#ifndef _NESO_PARTICLES_CONTAINERS_INDEX_MAP_HPP_
#define _NESO_PARTICLES_CONTAINERS_INDEX_MAP_HPP_

#include "../compute_target.hpp"

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
  inline int get_linear_index(const int key[KEY_DIM]) const {
    int index = key[KEY_DIM - 1];
    for (int dim = KEY_DIM - 2; dim >= 0; dim--) {
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
template <int KEY_DIM, int VALUE_DIM> struct IndexMap {};

template <int KEY_DIM, int VALUE_DIM>
using IndexMapSharedPtr = std::shared_ptr<IndexMap<KEY_DIM, VALUE_DIM>>;

/**
 * TODO
 */
template <int KEY_DIM, int VALUE_DIM>
IndexMapSharedPtr<KEY_DIM, VALUE_DIM>
create_index_map(SYCLTargetSharedPtr sycl_target);

/**
 * TODO
 */
template <int KEY_DIM, int VALUE_DIM>
void restore_index_map(SYCLTargetSharedPtr sycl_target,
                       IndexMapSharedPtr<KEY_DIM, VALUE_DIM> partition_context);
} // namespace NESO::Particles

#endif
