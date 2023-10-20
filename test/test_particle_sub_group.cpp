#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>
#include <type_traits>

using namespace NESO::Particles;

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common() {
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;

  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  const int cell_count = mesh->get_cell_count();

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<INT>("MARKER"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int N = 1093; // prime
  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;
  const INT id_offset = rank * N;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px + id_offset;
  }

  A->add_particles_local(initial_distribution);
  parallel_advection_initialisation(A, 16);

  auto ccb = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, A->position_dat, A->cell_id_dat);

  ccb->execute();
  A->cell_move();

  return A;
}

class TestParticleSubGroup : public ParticleSubGroup {
public:
  template <typename KERNEL, typename... ARGS>
  TestParticleSubGroup(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                       ARGS... args)
      : ParticleSubGroup(particle_group, kernel, args...) {}

  inline int test_get_cells_layers(std::vector<INT> &cells,
                                   std::vector<INT> &layers) {
    return get_cells_layers(cells, layers);
  }
};

} // namespace

TEST(ParticleSubGroup, selector) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  TestParticleSubGroup aa(
      A, [=](auto ID) { return ((ID[0] % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  aa.create();

  std::vector<INT> cells;
  std::vector<INT> layers;

  const int num_particles = aa.test_get_cells_layers(cells, layers);
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);

    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    EXPECT_TRUE((*id)[0][layerx] % 2 == 0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, particle_loop) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] == 42; }, Access::read(Sym<INT>("ID")));

  GlobalArray<int> counter(sycl_target, 1, 0);

  auto pl_counter = particle_loop(
      aa, [=](Access::GlobalArray::Add<int> G1) { G1(0, 1); },
      Access::add(counter));

  pl_counter->execute();

  auto vector_c = counter.get();
  EXPECT_EQ(vector_c.at(0), 1);

  auto bb = std::make_shared<TestParticleSubGroup>(
      A, [=](auto ID) { return ((ID[0] % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  LocalArray<int> counter2(sycl_target, 2, 0);

  auto pl_counter2 = particle_loop(
      std::dynamic_pointer_cast<ParticleSubGroup>(bb),
      [=](auto G1, auto ID, auto MARKER) {
        G1(0, 1);
        G1(1, ID[0]);
        MARKER[0] = 1;
      },
      Access::add(counter2), Access::read(Sym<INT>("ID")),
      Access::write(Sym<INT>("MARKER")));

  pl_counter2->execute();
  auto vector_b = counter2.get();

  std::vector<INT> cells;
  std::vector<INT> layers;
  const int num_particles = bb->test_get_cells_layers(cells, layers);

  EXPECT_EQ(num_particles, vector_b.at(0));

  int id_counter = 0;
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);

    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    id_counter += (*id)[0][layerx];
  }

  EXPECT_EQ(id_counter, vector_b.at(1));

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    const int nrow = id->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      if ((*id)[0][rowx] % 2 == 0) {
        EXPECT_EQ((*marker)[0][rowx], 1);
      } else {
        EXPECT_EQ((*marker)[0][rowx], 0);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
