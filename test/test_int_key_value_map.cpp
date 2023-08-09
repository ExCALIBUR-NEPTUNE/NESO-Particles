#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <map>
#include <neso_particles.hpp>
#include <random>
#include <set>

using namespace NESO::Particles;

TEST(BlockedBinaryTree, Indexing) {
  INT to_test;

  // test node key indexing
  to_test = BlockedBinaryNode<INT, double, 8>::get_node_key(16);
  ASSERT_EQ(to_test, 2);
  to_test = BlockedBinaryNode<INT, double, 8>::get_node_key(15);
  ASSERT_EQ(to_test, 1);
  to_test = BlockedBinaryNode<INT, double, 8>::get_node_key(8);
  ASSERT_EQ(to_test, 1);
  to_test = BlockedBinaryNode<INT, double, 8>::get_node_key(7);
  ASSERT_EQ(to_test, 0);
  to_test = BlockedBinaryNode<INT, double, 8>::get_node_key(0);
  ASSERT_EQ(to_test, 0);
  to_test = BlockedBinaryNode<INT, double, 8>::get_node_key(-1);
  ASSERT_EQ(to_test, -1);
  to_test = BlockedBinaryNode<INT, double, 8>::get_node_key(-8);
  ASSERT_EQ(to_test, -1);
  to_test = BlockedBinaryNode<INT, double, 8>::get_node_key(-9);
  ASSERT_EQ(to_test, -2);
  to_test = BlockedBinaryNode<INT, double, 8>::get_node_key(-16);
  ASSERT_EQ(to_test, -2);

  // test leaf key indexing
  to_test = BlockedBinaryNode<INT, double, 8>::get_leaf_key(16);
  ASSERT_EQ(to_test, 0);
  to_test = BlockedBinaryNode<INT, double, 8>::get_leaf_key(15);
  ASSERT_EQ(to_test, 7);
  to_test = BlockedBinaryNode<INT, double, 8>::get_leaf_key(8);
  ASSERT_EQ(to_test, 0);
  to_test = BlockedBinaryNode<INT, double, 8>::get_leaf_key(7);
  ASSERT_EQ(to_test, 7);
  to_test = BlockedBinaryNode<INT, double, 8>::get_leaf_key(0);
  ASSERT_EQ(to_test, 0);
  to_test = BlockedBinaryNode<INT, double, 8>::get_leaf_key(-1);
  ASSERT_EQ(to_test, 7);
  to_test = BlockedBinaryNode<INT, double, 8>::get_leaf_key(-8);
  ASSERT_EQ(to_test, 0);
  to_test = BlockedBinaryNode<INT, double, 8>::get_leaf_key(-9);
  ASSERT_EQ(to_test, 7);
  to_test = BlockedBinaryNode<INT, double, 8>::get_leaf_key(-16);
  ASSERT_EQ(to_test, 0);
}

template <INT WIDTH> static inline void tree_wrapper() {

  auto sycl_target = std::make_shared<SYCLTarget>(GPU_SELECTOR, MPI_COMM_WORLD);

  auto map =
      std::make_shared<BlockedBinaryTree<INT, double, WIDTH>>(sycl_target);

  double output;
  ASSERT_EQ(map->host_get(1, &output), false);

  const double v0 = 1.1123;
  map->add(42, v0);
  bool exists;
  exists = map->host_get(43, &output);
  ASSERT_EQ(exists, false);
  exists = map->host_get(42, &output);
  ASSERT_EQ(exists, true);
  ASSERT_EQ(output, v0);

  const int N = 1024;

  std::mt19937 rng(124318);
  std::mt19937 rngf(4318);
  std::uniform_int_distribution<INT> dist(-32, 32);
  std::uniform_real_distribution<double> distf{};
  std::map<INT, double> correct;

  std::vector<INT> keys_unordered;
  for (int ix = 0; ix < N; ix++) {
    keys_unordered.push_back(dist(rng));
  }
  for (auto key : keys_unordered) {
    correct[key] = distf(rngf);
  }
  for (auto key : keys_unordered) {
    const double value = correct[key];
    map->add(key, value);
  }

  for (auto key_value : correct) {
    double value;
    const bool found = map->host_get(key_value.first, &value);
    ASSERT_TRUE(found);
    ASSERT_EQ(value, key_value.second);
  }
}

TEST(BlockedBinaryTree, BinaryTree) {
  // width = 1 is a binary tree
  tree_wrapper<1>();
}

TEST(BlockedBinaryTree, EightBlockTree) { tree_wrapper<8>(); }
