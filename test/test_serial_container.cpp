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

struct IntTuple : public SerialInterface {
  int rank_source;
  int rank_destination;
  int a;
  virtual inline std::size_t get_num_bytes() const override {
    return 3 * sizeof(int);
  }
  virtual inline void
  serialise([[maybe_unused]] std::byte *buffer,
            [[maybe_unused]] const std::size_t num_bytes) const override {
    ASSERT_EQ(num_bytes, 3 * sizeof(int));
    std::memcpy(buffer, &this->rank_source, sizeof(int));
    buffer += sizeof(int);
    std::memcpy(buffer, &this->rank_destination, sizeof(int));
    buffer += sizeof(int);
    std::memcpy(buffer, &this->a, sizeof(int));
  }
  virtual inline void
  deserialise([[maybe_unused]] const std::byte *buffer,
              [[maybe_unused]] const std::size_t num_bytes) override {
    ASSERT_EQ(num_bytes, 3 * sizeof(int));
    std::memcpy(&this->rank_source, buffer, sizeof(int));
    buffer += sizeof(int);
    std::memcpy(&this->rank_destination, buffer, sizeof(int));
    buffer += sizeof(int);
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

TEST(MeshHierarchyData, init) {
  
  int size, rank;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  const int ndim = 2;
  const int mesh_size = size;
  std::vector<int> dims = {mesh_size, mesh_size};
  std::vector<double> origin = {0.0, 0.0};
  const double extent = 1.0;
  const int subdivision_order = 0;
  auto mesh_hierarchy = std::make_shared<MeshHierarchy>(
    MPI_COMM_WORLD,
    ndim,
    dims,
    origin,
    extent,
    subdivision_order
  );

  mesh_hierarchy->claim_initialise();
  mesh_hierarchy->claim_cell(rank, 1);
  mesh_hierarchy->claim_finalise();

  const int owner = mesh_hierarchy->get_owner(rank);
  ASSERT_EQ(owner, rank);

  const int N = 1;
  std::map<INT, std::vector<IntTuple>> sources;

  for(int rx=0 ; rx<size ; rx++){
    sources[rx] = std::vector<IntTuple>(N);
    for(int ix=0 ; ix<N ; ix++){
      sources.at(rx).at(ix).rank_source = rank;
      sources.at(rx).at(ix).rank_destination = rx;
      sources.at(rx).at(ix).a = ix+1;
    }
  }

  MeshHierarchyContainer mhc(mesh_hierarchy, sources);


  mesh_hierarchy->free();
}







