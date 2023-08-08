#ifndef _NESO_PARTICLES_INT_KEY_VALUE_MAP
#define _NESO_PARTICLES_INT_KEY_VALUE_MAP

#include "compute_target.hpp"
#include <map>
#include <set>
#include <type_traits>

namespace NESO::Particles {

/**
 *  TODO
 */
template <typename KEY_TYPE, typename VALUE_TYPE, INT WIDTH>
struct IntKeyValueNode {

  KEY_TYPE node_key;
  IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *lhs;
  IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *rhs;
  bool exists[WIDTH];
  VALUE_TYPE data[WIDTH];

  /**
   * TODO
   */
  inline void reset() {
    this->lhs = nullptr;
    this->rhs = nullptr;
    for (int ix = 0; ix < WIDTH; ix++) {
      this->exists[ix] = false;
      this->data[ix] = 0;
    }
  }
  /**
   * TODO
   */
  static inline KEY_TYPE get_node_key(const KEY_TYPE key) {
    const INT cast_key = static_cast<INT>(key);
    if (key < 0) {
      const INT mcast_key = ((-cast_key - 1) / WIDTH) + 1;
      return -mcast_key;
    } else {
      return cast_key / WIDTH;
    }
  }
  /**
   * TODO
   */
  static inline KEY_TYPE get_leaf_key(const KEY_TYPE key) {
    const INT cast_key = static_cast<INT>(key);
    if (cast_key < 0) {
      return WIDTH - 1 - ((-cast_key - 1) % WIDTH);
    } else {
      return cast_key % WIDTH;
    }
  }
  /**
   * TODO
   */
  inline bool is_node(const KEY_TYPE node_key) {
    return (node_key == this->node_key);
  }
  /**
   * TODO
   */
  inline bool is_leaf_set(const KEY_TYPE leaf_key) { return exists[leaf_key]; }
  /**
   * TODO
   */
  inline VALUE_TYPE get_value(const KEY_TYPE leaf_key) {
    return this->data[leaf_key];
  }
  /**
   * TODO
   */
  inline IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *
  next(const KEY_TYPE node_key) {
    if (node_key < this->node_key) {
      return lhs;
    } else if (node_key > this->node_key) {
      return rhs;
    } else {
      return this;
    }
  }
  /**
   * TODO
   */
  inline bool get_location(const KEY_TYPE key, bool **leaf_set,
                           VALUE_TYPE **value) {
    const KEY_TYPE node_key = get_node_key(key);
    const KEY_TYPE leaf_key = get_leaf_key(key);

    IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *next = this;

    bool node_found = this->is_node(node_key);
    while ((!node_found) && (next != nullptr)) {
      node_found = next->is_node(node_key);
      if (!node_found) {
        next = next->next(node_key);
      }
    }
    if (node_found) {
      *leaf_set = &(next->exists[leaf_key]);
      *value = &(next->data[leaf_key]);
      return true;
    } else {
      return false;
    }
  }

  inline bool get(const KEY_TYPE key, VALUE_TYPE *value) {
    VALUE_TYPE *value_location;
    bool *leaf_set;
    const bool exists = this->get_location(key, &leaf_set, &value_location);
    if (exists && (*leaf_set)) {
      *value = *value_location;
      return true;
    } else {
      return false;
    }
  }

  inline bool set(const KEY_TYPE key, const VALUE_TYPE value) {
    bool *leaf_set;
    VALUE_TYPE *value_location;
    const bool exists = this->get_location(key, &leaf_set, &value_location);
    if (exists) {
      *value_location = value;
      *leaf_set = true;
      return true;
    } else {
      return false;
    }
  }

  inline void add_node(const KEY_TYPE node_key,
                       IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *node) {

    IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *current_node = this;
    IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *next_node =
        current_node->next(node_key);

    // travel down the tree until we find a branch not populated
    while (next_node != nullptr) {
      current_node = next_node;
      next_node = current_node->next(node_key);
    }

    // add the new node to current_node either lhs or rhs
    if (node_key < current_node->node_key) {
      current_node->lhs = node;
    } else {
      current_node->rhs = node;
    }
  }
};

/**
 *  TODO
 */
template <typename KEY_TYPE, typename VALUE_TYPE, INT WIDTH>
class IntKeyValueMap {
protected:
  std::map<KEY_TYPE, IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *> nodes;

public:
  /// TODO
  SYCLTargetSharedPtr sycl_target;
  /// TODO
  IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *root;

  ~IntKeyValueMap() {
    for (auto &nx : this->nodes) {
      this->sycl_target->free(nx.second);
    }
  }

  /**
   *  TODO
   */
  IntKeyValueMap(SYCLTargetSharedPtr sycl_target) : sycl_target(sycl_target) {

    static_assert(std::is_trivially_copyable_v<
                      IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH>> == true,
                  "IntKeyValueNode is not trivially copyable to device");
    this->root = nullptr;
  }
  inline void add(const KEY_TYPE key, const VALUE_TYPE value) {

    const KEY_TYPE node_key =
        IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH>::get_node_key(key);
    const KEY_TYPE leaf_key =
        IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH>::get_leaf_key(key);
    const bool node_exists = static_cast<bool>(this->nodes.count(node_key));

    // need to allocate a new node
    if (!node_exists) {

      IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *new_node = nullptr;
      new_node = static_cast<IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH> *>(
          this->sycl_target->malloc_device(
              sizeof(IntKeyValueNode<KEY_TYPE, VALUE_TYPE, WIDTH>)));
      this->nodes[node_key] = new_node;

      bool add_to_tree = true;
      if (this->root == nullptr) {
        add_to_tree = false;
        this->root = new_node;
      }
      auto k_root = this->root;

      this->sycl_target->queue
          .submit([&](sycl::handler &cgh) {
            cgh.single_task<>([=]() {
              new_node->reset();
              new_node->node_key = node_key;
              new_node->exists[leaf_key] = true;
              new_node->data[leaf_key] = value;

              if (add_to_tree) {
                k_root->add_node(node_key, new_node);
              }
            });
          })
          .wait_and_throw();

    } else {

      auto k_root = this->root;
      this->sycl_target->queue
          .submit([&](sycl::handler &cgh) {
            cgh.single_task<>([=]() { k_root->set(key, value); });
          })
          .wait_and_throw();
    }
  }

  /**
   * TODO
   */
  inline bool host_get(const KEY_TYPE key, VALUE_TYPE *value) {
    if (this->root == nullptr) {
      return false;
    }
    BufferDeviceHost<VALUE_TYPE> output(this->sycl_target, 1);
    BufferDeviceHost<bool> exists(this->sycl_target, 1);

    auto k_output = output.d_buffer.ptr;
    auto k_exists = exists.d_buffer.ptr;

    auto k_root = this->root;
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task<>(
              [=]() { k_exists[0] = k_root->get(key, k_output); });
        })
        .wait_and_throw();
    output.device_to_host();
    exists.device_to_host();

    *value = output.h_buffer.ptr[0];
    return exists.h_buffer.ptr[0];
  }
};

}; // namespace NESO::Particles

#endif
