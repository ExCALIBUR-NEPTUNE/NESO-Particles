#ifndef _NESO_PARTICLES_LOOKUP_TABLE_H_
#define _NESO_PARTICLES_LOOKUP_TABLE_H_

#include "../compute_target.hpp"
#include <cstdint>
#include <memory>
#include <set>

namespace NESO::Particles {

/**
 * Device type for LookupTable.
 */
template <typename KEY_TYPE, typename VALUE_TYPE> struct LookupTableNode {
  int size;
  std::int8_t const *RESTRICT d_entry_exists;
  VALUE_TYPE const *RESTRICT d_entries;

  /**
   *  For a given key find and return the stored value.
   *
   *  @param[in] key Input global key to retrieve value for.
   *  @param[in, out] value Pointer to value, only valid if the key is found.
   *  @returns True if the key is found in the tree otherwise false.
   */
  inline bool get(const KEY_TYPE key, VALUE_TYPE *value) {
    const bool exists = d_entry_exists[key] > 0;
    if (exists) {
      *value = d_entries[key];
      return true;
    } else {
      return false;
    }
  }
};

/**
 * Simple lookup table type.
 */
template <typename KEY_TYPE, typename VALUE_TYPE> class LookupTable {
protected:
  std::set<KEY_TYPE> h_entry_exists;
  std::unique_ptr<BufferDevice<std::int8_t>> d_entry_exists;
  std::unique_ptr<BufferDevice<VALUE_TYPE>> d_entries;
  std::unique_ptr<BufferDevice<LookupTableNode<KEY_TYPE, VALUE_TYPE>>> d_root;

public:
  SYCLTargetSharedPtr sycl_target;
  int size;
  LookupTableNode<KEY_TYPE, VALUE_TYPE> *root;

  /**
   * Create new lookup table on device.
   *
   * @param sycl_target Device on which to create lookup table.
   * @param size Size of lookup table, entries are indexed [0, size).
   */
  LookupTable(SYCLTargetSharedPtr sycl_target, const int size)
      : sycl_target(sycl_target), size(size) {
    std::vector<std::int8_t> tmp_false(size);
    std::fill(tmp_false.begin(), tmp_false.end(), 0);
    this->d_entry_exists =
        std::make_unique<BufferDevice<std::int8_t>>(sycl_target, tmp_false);
    this->d_entries =
        std::make_unique<BufferDevice<VALUE_TYPE>>(sycl_target, size);
    std::vector<LookupTableNode<KEY_TYPE, VALUE_TYPE>> tmp_root(1);
    tmp_root.at(0).size = size;
    tmp_root.at(0).d_entry_exists = this->d_entry_exists->ptr;
    tmp_root.at(0).d_entries = this->d_entries->ptr;
    this->d_root =
        std::make_unique<BufferDevice<LookupTableNode<KEY_TYPE, VALUE_TYPE>>>(
            sycl_target, tmp_root);
    this->root = this->d_root->ptr;
  }

  /**
   * Add an entry to the lookup table.
   *
   * @param key Key for entry, must be in [0, size).
   * @param value Value to add for key.
   */
  inline void add(const KEY_TYPE key, const VALUE_TYPE value) {
    NESOASSERT((key > -1) && (key < this->size), "Bad key passed to add");
    const std::int8_t true_value = 1;
    auto e0 = this->sycl_target->queue.memcpy(this->d_entry_exists->ptr + key,
                                              &true_value, sizeof(std::int8_t));
    auto e1 = this->sycl_target->queue.memcpy(this->d_entries->ptr + key,
                                              &value, sizeof(VALUE_TYPE));
    e0.wait_and_throw();
    e1.wait_and_throw();
    this->h_entry_exists.insert(key);
  }

  /**
   * Host callable method to retrieve the value that corresponds to a given key.
   *
   * @param[in] key Key to retrieve value for.
   * @param[in, out] value Pointer to value type in which to place output value.
   * @returns True if the key is found in the container otherwise false.
   */
  inline bool host_get(const KEY_TYPE key, VALUE_TYPE *value) {
    NESOASSERT((key > -1) && (key < this->size), "Bad key passed to add");
    if (!this->h_entry_exists.count(key)) {
      return false;
    } else {
      this->sycl_target->queue
          .memcpy(value, this->d_entries->ptr + key, sizeof(VALUE_TYPE))
          .wait_and_throw();
      return true;
    }
  }
};

} // namespace NESO::Particles

#endif
