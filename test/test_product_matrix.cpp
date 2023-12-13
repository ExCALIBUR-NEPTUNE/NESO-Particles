#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>
#include <type_traits>

using namespace NESO::Particles;

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common(const int N = 1093) {
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;

  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("MARKER"), 2),
                             ParticleProp(Sym<INT>("PARENT"), 2),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int rank = sycl_target->comm_pair.rank_parent;
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

} // namespace

TEST(ProductMatrix, product_matrix_spec) {

  auto product_spec = product_matrix_spec(ParticleSpec(
      ParticleProp(Sym<REAL>("P2"), ndim), ParticleProp(Sym<INT>("MARKER"), 2),
      ParticleProp(Sym<INT>("PARENT"), 2)));

  ASSERT_EQ(product_spec->num_properties_real, 1);
  ASSERT_EQ(product_spec->num_properties_int, 2);
  ASSERT_EQ(product_spec->num_components_real, ndim);
  ASSERT_EQ(product_spec->num_components_int, 4);
  ASSERT_EQ(product_spec->syms_real.at(0), Sym<REAL>("P2"));
  ASSERT_EQ(product_spec->syms_int.at(0), Sym<INT>("MARKER"));
  ASSERT_EQ(product_spec->syms_int.at(1), Sym<INT>("PARENT"));
  ASSERT_EQ(product_spec->components_real.at(0), ndim);
  ASSERT_EQ(product_spec->components_int.at(0), 2);
  ASSERT_EQ(product_spec->components_int.at(1), 2);
}

namespace {

class ProductMatrixTest : public NESO::Particles::ProductMatrix {

public:
  ProductMatrixTest(SYCLTargetSharedPtr sycl_target,
                    std::shared_ptr<ProductMatrixSpec> spec)
      : ProductMatrix(sycl_target, spec) {}

  inline void get(std::vector<REAL> &out_real, std::vector<INT> &out_int) {

    const size_t num_real =
        this->num_products * this->spec->num_components_real;
    const size_t num_int = this->num_products * this->spec->num_components_int;
    out_real.resize(num_real);
    out_int.resize(num_int);
    this->sycl_target->queue
        .memcpy(out_real.data(), this->d_data_real->ptr,
                sizeof(REAL) * num_real)
        .wait_and_throw();
    this->sycl_target->queue
        .memcpy(out_int.data(), this->d_data_int->ptr, sizeof(INT) * num_int)
        .wait_and_throw();
  }
};

} // namespace

TEST(ProductMatrix, base) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto product_spec = product_matrix_spec(ParticleSpec(
      ParticleProp(Sym<REAL>("P2"), ndim), ParticleProp(Sym<INT>("MARKER"), 2),
      ParticleProp(Sym<INT>("PARENT"), 2)));

  product_spec->set_default_value(Sym<INT>("MARKER"), 1, 43);

  auto pm = std::make_shared<ProductMatrixTest>(sycl_target, product_spec);

  auto loop = particle_loop(
      A,
      [=](auto index, auto PM, auto P) {
        const INT particle_index = index.get_local_linear_index();
        for (int dx = 0; dx < ndim; dx++) {
          PM.at_real(particle_index, 0, dx) = P.at(dx);
        }
        PM.at_int(particle_index, 0, 0) = 42;
        PM.at_int(particle_index, 1, 0) = index.cell;
        PM.at_int(particle_index, 1, 1) = index.layer;
      },
      Access::read(ParticleLoopIndex{}),
      Access::write(std::dynamic_pointer_cast<ProductMatrix>(pm)),
      Access::read(Sym<REAL>("P")));

  const int npart_local = A->get_npart_local();
  pm->reset(npart_local);
  ASSERT_EQ(pm->num_products, npart_local);
  loop->execute();

  std::vector<REAL> out_real;
  std::vector<INT> out_int;
  pm->get(out_real, out_int);

  std::map<std::tuple<int, int>, std::vector<REAL>> positions;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto p = A->get_dat(Sym<REAL>("P"))->cell_dat.get_cell(cellx);
    const int nrow = p->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      // for each dimension
      std::vector<REAL> pv(ndim);
      for (int dimx = 0; dimx < ndim; dimx++) {
        pv.at(dimx) = (*p)[dimx][rowx];
      }
      positions[{cellx, rowx}] = pv;
    }
  }

  for (int px = 0; px < npart_local; px++) {
    const int marker0 = out_int.at(0 * npart_local + px);
    const int marker1 = out_int.at(1 * npart_local + px);
    const int parent_cell = out_int.at(2 * npart_local + px);
    const int parent_layer = out_int.at(3 * npart_local + px);
    ASSERT_EQ(marker0, 42);
    ASSERT_EQ(marker1, 43);

    std::vector<REAL> to_test(ndim);
    for (int dx = 0; dx < ndim; dx++) {
      to_test.at(dx) = out_real.at(dx * npart_local + px);
    }
    auto correct = positions.at({parent_cell, parent_layer});
    for (int dx = 0; dx < ndim; dx++) {
      ASSERT_EQ(to_test.at(dx), correct.at(dx));
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

/**
 * Implicitly places the new particles in cell 0
 */
TEST(ProductMatrix, add_particles_local_0) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto product_spec = product_matrix_spec(ParticleSpec(
      ParticleProp(Sym<REAL>("P2"), ndim), ParticleProp(Sym<INT>("MARKER"), 2),
      ParticleProp(Sym<INT>("PARENT"), 2)));

  product_spec->set_default_value(Sym<INT>("MARKER"), 1, 43);

  auto pm = std::make_shared<ProductMatrixTest>(sycl_target, product_spec);

  auto loop = particle_loop(
      A,
      [=](auto index, auto PM, auto P) {
        const INT particle_index = index.get_local_linear_index();
        for (int dx = 0; dx < ndim; dx++) {
          PM.at_real(particle_index, 0, dx) = P.at(dx);
        }
        PM.at_int(particle_index, 0, 0) = 42;
        PM.at_int(particle_index, 1, 0) = index.cell;
        PM.at_int(particle_index, 1, 1) = index.layer;
      },
      Access::read(ParticleLoopIndex{}),
      Access::write(std::dynamic_pointer_cast<ProductMatrix>(pm)),
      Access::read(Sym<REAL>("P")));

  const int npart_local = A->get_npart_local();
  pm->reset(npart_local);
  ASSERT_EQ(pm->num_products, npart_local);
  loop->execute();

  std::vector<REAL> out_real;
  std::vector<INT> out_int;
  pm->get(out_real, out_int);

  std::map<std::tuple<int, int>, std::vector<REAL>> positions;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto p = A->get_dat(Sym<REAL>("P"))->cell_dat.get_cell(cellx);
    const int nrow = p->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      // for each dimension
      std::vector<REAL> pv(ndim);
      for (int dimx = 0; dimx < ndim; dimx++) {
        pv.at(dimx) = (*p)[dimx][rowx];
      }
      positions[{cellx, rowx}] = pv;
    }
  }

  particle_loop(
      A,
      [](auto PARENT) {
        PARENT.at(0) = -1;
        PARENT.at(1) = -1;
      },
      Access::write(Sym<INT>("PARENT")))
      ->execute();

  const INT npart_local_before = A->get_npart_local();
  std::vector<INT> npart_cell(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    npart_cell.at(cx) = A->get_npart_cell(cx);
  }

  A->add_particles_local(std::dynamic_pointer_cast<ProductMatrix>(pm));

  // we did not set the cells -> the new particles should all be in cell 0
  ASSERT_EQ(A->get_npart_local(), 2 * npart_local_before);
  ASSERT_EQ(npart_cell.at(0) + npart_local_before, A->get_npart_cell(0));
  for (int cx = 1; cx < cell_count; cx++) {
    ASSERT_EQ(npart_cell.at(cx), A->get_npart_cell(cx));
  }

  auto P = A->get_dat(Sym<REAL>("P"))->cell_dat.get_cell(0);
  auto P2 = A->get_dat(Sym<REAL>("P2"))->cell_dat.get_cell(0);
  auto PARENT = A->get_dat(Sym<INT>("PARENT"))->cell_dat.get_cell(0);
  auto MARKER = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(0);
  const int nrow = P2->nrow;

  for (int rowx = 0; rowx < nrow; rowx++) {
    const INT parent_cell = PARENT->at(rowx, 0);
    const INT parent_row = PARENT->at(rowx, 1);
    if (parent_row > -1) {
      auto correct = positions.at({parent_cell, parent_row});
      for (int dx = 0; dx < ndim; dx++) {
        ASSERT_EQ(P2->at(rowx, dx), correct.at(dx));
      }
      ASSERT_EQ(MARKER->at(rowx, 0), 42);
      ASSERT_EQ(MARKER->at(rowx, 1), 43);
      for (int dx = 0; dx < ndim; dx++) {
        ASSERT_EQ(P->at(rowx, dx), 0.0);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ProductMatrix, add_particles_local_1) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  std::vector<ParticleProp<REAL>> properties_real = {
      ParticleProp(Sym<REAL>("P2"), ndim)};
  std::vector<ParticleProp<INT>> properties_int = {
      ParticleProp(Sym<INT>("MARKER"), 2), ParticleProp(Sym<INT>("PARENT"), 2),
      ParticleProp(Sym<INT>("CELL_ID"), 1)};

  auto product_spec =
      product_matrix_spec(ParticleSpec(properties_real, properties_int));

  product_spec->set_default_value(Sym<INT>("MARKER"), 1, 43);

  auto pm = std::make_shared<ProductMatrixTest>(sycl_target, product_spec);

  auto loop = particle_loop(
      A,
      [=](auto index, auto PM, auto P, auto CELL_ID) {
        const INT particle_index = index.get_local_linear_index();
        for (int dx = 0; dx < ndim; dx++) {
          PM.at_real(particle_index, 0, dx) = P.at(dx);
        }
        PM.at_int(particle_index, 0, 0) = 42;
        PM.at_int(particle_index, 1, 0) = index.cell;
        PM.at_int(particle_index, 1, 1) = index.layer;
        // place the particle in the correct cell
        PM.at_int(particle_index, 2, 0) = CELL_ID.at(0);
      },
      Access::read(ParticleLoopIndex{}),
      Access::write(std::dynamic_pointer_cast<ProductMatrix>(pm)),
      Access::read(Sym<REAL>("P")), Access::read(Sym<INT>("CELL_ID")));

  const int npart_local = A->get_npart_local();
  pm->reset(npart_local);
  ASSERT_EQ(pm->num_products, npart_local);
  loop->execute();

  std::vector<REAL> out_real;
  std::vector<INT> out_int;
  pm->get(out_real, out_int);

  std::map<std::tuple<int, int>, std::vector<REAL>> positions;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto p = A->get_dat(Sym<REAL>("P"))->cell_dat.get_cell(cellx);
    const int nrow = p->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      // for each dimension
      std::vector<REAL> pv(ndim);
      for (int dimx = 0; dimx < ndim; dimx++) {
        pv.at(dimx) = (*p)[dimx][rowx];
      }
      positions[{cellx, rowx}] = pv;
    }
  }

  particle_loop(
      A,
      [](auto PARENT) {
        PARENT.at(0) = -1;
        PARENT.at(1) = -1;
      },
      Access::write(Sym<INT>("PARENT")))
      ->execute();

  const INT npart_local_before = A->get_npart_local();
  std::vector<INT> npart_cell(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    npart_cell.at(cx) = A->get_npart_cell(cx);
  }

  A->add_particles_local(std::dynamic_pointer_cast<ProductMatrix>(pm));

  ASSERT_EQ(A->get_npart_local(), 2 * npart_local_before);
  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(npart_cell.at(cx) * 2, A->get_npart_cell(cx));
  }

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto P = A->get_dat(Sym<REAL>("P"))->cell_dat.get_cell(cellx);
    auto P2 = A->get_dat(Sym<REAL>("P2"))->cell_dat.get_cell(cellx);
    auto PARENT = A->get_dat(Sym<INT>("PARENT"))->cell_dat.get_cell(cellx);
    auto MARKER = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    const int nrow = P2->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT parent_cell = PARENT->at(rowx, 0);
      const INT parent_row = PARENT->at(rowx, 1);
      if (parent_row > -1) {
        auto correct = positions.at({parent_cell, parent_row});
        for (int dx = 0; dx < ndim; dx++) {
          ASSERT_EQ(P2->at(rowx, dx), correct.at(dx));
        }
        ASSERT_EQ(PARENT->at(rowx, 0), cellx);
        ASSERT_EQ(MARKER->at(rowx, 0), 42);
        ASSERT_EQ(MARKER->at(rowx, 1), 43);
        for (int dx = 0; dx < ndim; dx++) {
          ASSERT_EQ(P->at(rowx, dx), 0.0);
        }
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
