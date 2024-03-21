#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <mesh_hierarchy_data/mesh_hierarchy_data.hpp>
#include <neso_particles.hpp>
#include <string>
#include <vector>

using namespace NESO::Particles;
using namespace MeshHierarchyData;

namespace {

struct EmptySerialise : public SerialInterface {
  virtual inline std::size_t get_num_bytes() const override { return 0; }
  virtual inline void
  serialise([[maybe_unused]] std::byte *buffer,
            [[maybe_unused]] const std::size_t num_bytes) const override {}
  virtual inline void
  deserialise([[maybe_unused]] const std::byte *buffer,
              [[maybe_unused]] const std::size_t num_bytes) override {}
};

struct IntSerialise : public SerialInterface {
  int a;

  virtual inline std::size_t get_num_bytes() const override {
    return sizeof(int);
  }
  virtual inline void
  serialise([[maybe_unused]] std::byte *buffer,
            [[maybe_unused]] const std::size_t num_bytes) const override {
    ASSERT_EQ(num_bytes, sizeof(int));
    std::memcpy(buffer, &this->a, sizeof(int));
  }
  virtual inline void
  deserialise([[maybe_unused]] const std::byte *buffer,
              [[maybe_unused]] const std::size_t num_bytes) override {
    ASSERT_EQ(num_bytes, sizeof(int));
    std::memcpy(&this->a, buffer, sizeof(int));
  }
};

} // namespace

TEST(SerialContainer, size_zero) {
  const int N = 16;
  std::vector<EmptySerialise> empty_objs(N);
  SerialContainer<EmptySerialise> sce(empty_objs);
  ASSERT_EQ(sce.buffer.size(), N * sizeof(std::size_t));
}

TEST(SerialContainer, size_int) {
  const int N = 16;
  std::vector<IntSerialise> int_objs(N);
  for (int ix = 0; ix < N; ix++) {
    int_objs.at(ix).a = ix + 1;
  }

  SerialContainer<IntSerialise> sce(int_objs);
  ASSERT_EQ(sce.buffer.size(), N * (sizeof(std::size_t) + sizeof(int)));

  // copy the packed objs into a new instance
  SerialContainer<IntSerialise> sc_unpack(sce.buffer.size());

  std::size_t first_size;
  std::memcpy(&first_size, sce.buffer.data(), sizeof(std::size_t));
  ASSERT_EQ(first_size, sizeof(int));

  ASSERT_EQ(sc_unpack.buffer.size(), sce.buffer.size());
  std::memcpy(sc_unpack.buffer.data(), sce.buffer.data(), sce.buffer.size());

  // unpack by deserialising
  std::vector<IntSerialise> unpacked_int_objs;
  sc_unpack.get(unpacked_int_objs);
  ASSERT_EQ(unpacked_int_objs.size(), N);
  for (int ix = 0; ix < N; ix++) {
    ASSERT_EQ(unpacked_int_objs.at(ix).a, ix + 1);
  }
}

TEST(SerialContainer, append) {
  const int N = 16;
  const int M = 32;
  std::vector<IntSerialise> a(N);
  std::vector<IntSerialise> b(M);
  for (int ix = 0; ix < N; ix++) {
    a.at(ix).a = ix + 1;
  }
  for (int ix = 0; ix < M; ix++) {
    b.at(ix).a = ix + 100;
  }

  SerialContainer<IntSerialise> sa(a);
  SerialContainer<IntSerialise> sb(b);
  SerialContainer<IntSerialise> se{};
  se.append(sa);
  ASSERT_EQ(se.buffer.size(), N * (sizeof(std::size_t) + sizeof(int)));
  se.append(sb);
  ASSERT_EQ(se.buffer.size(), (N + M) * (sizeof(std::size_t) + sizeof(int)));

  std::vector<IntSerialise> c;
  se.get(c);
  for (int ix = 0; ix < N; ix++) {
    ASSERT_EQ(c.at(ix).a, ix + 1);
  }
  for (int ix = 0; ix < M; ix++) {
    ASSERT_EQ(c.at(ix + N).a, ix + 100);
  }
}
