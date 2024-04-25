#ifndef _NESO_PARTICLES_BLOCKED_BINARY_TREE_H_
#define _NESO_PARTICLES_BLOCKED_BINARY_TREE_H_

#include "../compute_target.hpp"
#include <map>
#include <set>
#include <type_traits>

namespace NESO::Particles {

/**
 *  Generic node in a tree where each node stores an array of VALUE_TYPE
 *  elements of length WIDTH. `BlockedBinaryTree` is the class that actually
 *  creates the tree. This type should be trivially copyable to the device.
 */
template <typename KEY_TYPE, typename VALUE_TYPE, INT WIDTH>
struct BlockedBinaryNode {

  /// The starting key this node in the tree holds, i.e. with WIDTH=8 node 2
  /// holds keys [8,15].
  KEY_TYPE node_key;
  /// Pointer to the node which forms the left hand branch from this node. May
  /// be nullptr if this node does not exist.
  BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *lhs;
  /// Pointer to the node which forms the right hand branch from this node. May
  /// be nullptr if this node does not exist.
  BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *rhs;
  /// Bools that indicate if the entries in the data member are actual values.
  bool exists[WIDTH];
  /// The storage for the values the tree holds.
  VALUE_TYPE data[WIDTH];

  /**
   * Reset the node to a default state with no child nodes and no data held.
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
   * For a given input key (in the global key space) get the node key which
   * contains the values for the key.
   *
   * @param key Input key to index into the tree with.
   * @returns Node key that indicates the node in the tree which contains the
   * key.
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
   * For a given input key (in the global key space) get the leaf key which can
   * be used to index into the data member on the node which contains the key -
   * see `get_node_key`.
   *
   * @param key Input key to index into container.
   * @returns Index into data member that corresponds to the input key.
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
   * Test if the node corresponds to the input node key.#
   *
   * @param node_key Node key to test.
   * @returns True if this node corresponds to the input node key.
   */
  inline bool is_node(const KEY_TYPE node_key) {
    return (node_key == this->node_key);
  }

  /**
   * Assuming a node does not match a requested node key, return the child
   * branch which might contain the node. If the node key matches the current
   * node returns the current node.
   *
   * @param node_key Node key currently being searched for.
   * @returns Pointer to child node (may be nullptr).
   */
  inline BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *
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
   * For a given global key find the leaf location for the value and the bool
   * that indicates if the value is set. If the key is not in the tree return
   * false.
   *
   * @param[in] key Global key to find location of value for.
   * @param[in, out] leaf_set Return location that points to the flag that
   * indicates if the value is set.
   * @param[in, out] value Return location for a pointer to the value.
   * @returns True if the key is located in the tree otherwise false. The
   * leaf_set and value parameters only contain meaningful values if the return
   * value is true.
   */
  inline bool get_location(const KEY_TYPE key, bool **leaf_set,
                           VALUE_TYPE **value) {
    const KEY_TYPE node_key = get_node_key(key);
    const KEY_TYPE leaf_key = get_leaf_key(key);

    BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *next = this;

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

  /**
   *  For a given key find and return the stored value.
   *
   *  @param[in] key Input global key to retrieve value for.
   *  @param[in, out] value Pointer to value, only valid if the key is found.
   *  @returns True if the key is found in the tree otherwise false.
   */
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

  /**
   *  For a given key store the corresponding value in the tree. This function
   *  assumes that the node is already allocated and placed in the tree
   *  according to the node_key.
   *
   *  @param key Global key to store value against.
   *  @param value Input value to store pointed to by key.
   *  @param Returns true if the value was successfully stored. A return value
   *  of false indicates the node which should store the value is not present
   *  in the tree.
   */
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

  /**
   * Add an allocated node to the tree under a given node key. The `node_key`
   * member of the added node will be used to determine tree placement.
   *
   * @param node Node to place into the tree.
   */
  inline void add_node(BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *node) {

    const KEY_TYPE node_key = node->node_key;
    BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *current_node = this;
    BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *next_node =
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
 *  Create a blocked key-value map with a given block size. This class creates
 *  the tree and provides methods to get and set key-value pairs.
 */
template <typename KEY_TYPE, typename VALUE_TYPE, INT WIDTH>
class BlockedBinaryTree {
protected:
  std::map<KEY_TYPE, BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *> nodes;

public:
  /// The SYCLTarget on which the tree is allocated.
  SYCLTargetSharedPtr sycl_target;
  /// The root node of the tree.
  BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *root;

  ~BlockedBinaryTree() {
    for (auto &nx : this->nodes) {
      this->sycl_target->free(nx.second);
    }
  }

  /**
   *  Create a new tree on a given SYCL device.
   */
  BlockedBinaryTree(SYCLTargetSharedPtr sycl_target)
      : sycl_target(sycl_target) {

    static_assert(std::is_trivially_copyable_v<
                      BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH>> == true,
                  "BlockedBinaryNode is not trivially copyable to device");
    static_assert(std::is_integral<KEY_TYPE>::value,
                  "KEY_TYPE is not an integral type.");
    this->root = nullptr;
  }

  /**
   * Add a key-value pair to the container.
   *
   * @param key Key to add to container.
   * @param value Value to add to container under given key.
   */
  inline void add(const KEY_TYPE key, const VALUE_TYPE value) {

    const KEY_TYPE node_key =
        BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH>::get_node_key(key);
    const KEY_TYPE leaf_key =
        BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH>::get_leaf_key(key);
    const bool node_exists = static_cast<bool>(this->nodes.count(node_key));

    // need to allocate a new node
    if (!node_exists) {

      BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *new_node = nullptr;
      new_node = static_cast<BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH> *>(
          this->sycl_target->malloc_device(
              sizeof(BlockedBinaryNode<KEY_TYPE, VALUE_TYPE, WIDTH>)));
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
                k_root->add_node(new_node);
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
   * Host callable method to retrieve the value that corresponds to a given key.
   *
   * @param[in] key Key to retrieve value for.
   * @param[in, out] value Pointer to value type in which to place output value.
   * @returns True if the key is found in the container otherwise false.
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
