#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/external_interfaces/common/overlay_cartesian_mesh.hpp>
#include <string>
#include <vector>

using namespace NESO::Particles;

TEST(ExternalCommon, overlay_cartesian_mesh) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int ndim = 2;
  std::vector<REAL> origin = {1.0, 2.0};
  std::vector<REAL> extents = {4.0, 8.0};
  std::vector<int> cell_counts = {8, 16};

  ExternalCommon::OverlayCartesianMesh ocm(sycl_target, ndim, origin, extents,
                                           cell_counts);

  ASSERT_EQ(ocm.origin, origin);
  ASSERT_EQ(ocm.extents, extents);
  ASSERT_EQ(ocm.cell_counts, cell_counts);

  for (int dimx = 0; dimx < ndim; dimx++) {
    const REAL cell_width = extents.at(dimx) / cell_counts.at(dimx);
    const REAL inverse_cell_width = 1.0 / cell_width;
    ASSERT_NEAR(cell_width, ocm.cell_widths.at(dimx), 1.0e-15);
    ASSERT_NEAR(inverse_cell_width, ocm.inverse_cell_widths.at(dimx), 1.0e-15);
  }

  ASSERT_EQ(ocm.get_cell_in_dimension(0, 1.0), 0);
  ASSERT_EQ(ocm.get_cell_in_dimension(0, 5.0), 7);
  ASSERT_EQ(ocm.get_cell_in_dimension(1, 2.0), 0);
  ASSERT_EQ(ocm.get_cell_in_dimension(1, 10.0), 15);
  ASSERT_EQ(ocm.get_cell_count(), 8 * 16);

  std::vector<int> index = {0, 0};
  ASSERT_EQ(ocm.get_linear_cell_index(index), 0);
  index = {0, 1};
  ASSERT_EQ(ocm.get_linear_cell_index(index), 8);

  index = {1, 2};
  auto bb = ocm.get_bounding_box(index);

  ASSERT_NEAR(bb->lower(0), 1.5, 1.0e-14);
  ASSERT_NEAR(bb->lower(1), 3.0, 1.0e-14);
  ASSERT_NEAR(bb->upper(0), 2.0, 1.0e-14);
  ASSERT_NEAR(bb->upper(1), 3.5, 1.0e-14);

  sycl_target->free();
}

TEST(ExternalCommon, overlay_cartesian_mesh_bb) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int ndim = 2;

  std::vector<REAL> bb = {1.0, 2.0, 0.0, 5.0, 10.0, 0.0};

  auto bounding_box = std::make_shared<ExternalCommon::BoundingBox>(bb);

  auto ocm = create_overlay_mesh(sycl_target, ndim, bounding_box, 32);

  ASSERT_TRUE(ocm->get_cell_count() >= 32);
  ASSERT_EQ(ocm->ndim, 2);

  sycl_target->free();
}

TEST(ExternalCommon, overlay_cartesian_mesh_bb_intersection) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int ndim = 2;
  std::vector<REAL> origin = {0.0, 0.0};
  std::vector<REAL> extents = {8.0, 16.0};
  std::vector<int> cell_counts = {8, 16};

  ExternalCommon::OverlayCartesianMesh ocm(sycl_target, ndim, origin, extents,
                                           cell_counts);

  std::vector<int> cells;

  {
    std::vector<REAL> bbv = {0.0, 0.0, 0.0, 0.1, 0.1, 0.1};
    auto bb = std::make_shared<ExternalCommon::BoundingBox>(bbv);
    ocm.get_intersecting_cells(bb, cells);
    ASSERT_EQ(cells.size(), 1);
    ASSERT_EQ(cells.at(0), 0);
  }
  {
    std::vector<REAL> bbv = {0.5, 0.5, 0.0, 1.5, 1.1, 0.1};
    auto bb = std::make_shared<ExternalCommon::BoundingBox>(bbv);
    ocm.get_intersecting_cells(bb, cells);
    ASSERT_EQ(cells.size(), 4);
    ASSERT_EQ(cells.at(0), 0);
    ASSERT_EQ(cells.at(1), 8);
    ASSERT_EQ(cells.at(2), 1);
    ASSERT_EQ(cells.at(3), 9);
  }
  {
    std::vector<REAL> bbv = {1.0, 1.0, 0.0, 1.5, 1.1, 0.1};
    auto bb = std::make_shared<ExternalCommon::BoundingBox>(bbv);
    ocm.get_intersecting_cells(bb, cells);
    ASSERT_EQ(cells.size(), 4);
    ASSERT_EQ(cells.at(0), 0);
    ASSERT_EQ(cells.at(1), 8);
    ASSERT_EQ(cells.at(2), 1);
    ASSERT_EQ(cells.at(3), 9);
  }
  {
    std::vector<REAL> bbv = {0.5, 0.2, 1.0, 1.0, 1.0, 0.1};
    auto bb = std::make_shared<ExternalCommon::BoundingBox>(bbv);
    ocm.get_intersecting_cells(bb, cells);
    ASSERT_EQ(cells.size(), 4);
    ASSERT_EQ(cells.at(0), 0);
    ASSERT_EQ(cells.at(1), 8);
    ASSERT_EQ(cells.at(2), 1);
    ASSERT_EQ(cells.at(3), 9);
  }
  {
    std::vector<REAL> bbv = {-0.5, -0.2, 1.0, 1.0, 1.0, 0.1};
    auto bb = std::make_shared<ExternalCommon::BoundingBox>(bbv);
    ocm.get_intersecting_cells(bb, cells);
    ASSERT_EQ(cells.size(), 4);
    ASSERT_EQ(cells.at(0), 0);
    ASSERT_EQ(cells.at(1), 8);
    ASSERT_EQ(cells.at(2), 1);
    ASSERT_EQ(cells.at(3), 9);
  }
  {
    std::vector<REAL> bbv = {7.9, 15.9, 0, 8.2, 16.1, 0.1};
    auto bb = std::make_shared<ExternalCommon::BoundingBox>(bbv);
    ocm.get_intersecting_cells(bb, cells);
    ASSERT_EQ(cells.size(), 1);
    ASSERT_EQ(cells.at(0), 8 * 16 - 1);
  }

  sycl_target->free();
}

TEST(ExternalCommon, bounding_box_expand) {
  std::vector<REAL> bbv0 = {0.0, 0.0, 0.0, 1.0, 2.0, 0.0};
  std::vector<REAL> bbv1 = {-1.0, 1.0, 0.0, 2.0, 1.5, 0.0};
  auto bb0 = std::make_shared<ExternalCommon::BoundingBox>(bbv0);
  auto bb1 = std::make_shared<ExternalCommon::BoundingBox>(bbv1);
  auto bb = std::make_shared<ExternalCommon::BoundingBox>();

  bb->expand(bb0);
  ASSERT_EQ(bb->lower(0), 0.0);
  ASSERT_EQ(bb->lower(1), 0.0);
  ASSERT_EQ(bb->upper(0), 1.0);
  ASSERT_EQ(bb->upper(1), 2.0);
  bb->expand(bb1);
  ASSERT_EQ(bb->lower(0), -1.0);
  ASSERT_EQ(bb->lower(1), 0.0);
  ASSERT_EQ(bb->upper(0), 2.0);
  ASSERT_EQ(bb->upper(1), 2.0);
}

TEST(ExternalCommon, dof_mapper_dg) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int num_cells = 15;
  const int num_dofs = 3;
  const int num_dofs_total = num_cells * num_dofs;
  auto cell_dat_const =
      std::make_shared<CellDatConst<REAL>>(sycl_target, num_cells, num_dofs, 1);

  auto mapper = std::make_shared<ExternalCommon::DOFMapperDG>(
      sycl_target, num_cells, num_dofs);

  auto lambda_host_order = [&](const int cell, const int dof) {
    // reverse the order as a test
    return num_dofs_total - 1 - (cell * num_dofs + dof);
  };

  for (int cx = 0; cx < num_cells; cx++) {
    for (int dx = 0; dx < num_dofs; dx++) {
      mapper->set(cx, dx, lambda_host_order(cx, dx));
    }
  }

  for (int cx = 0; cx < num_cells; cx++) {
    for (int dx = 0; dx < num_dofs; dx++) {
      ASSERT_EQ(mapper->get(cx, dx), lambda_host_order(cx, dx));
    }
  }

  REAL value = 100.0;
  for (int cx = 0; cx < num_cells; cx++) {
    for (int dx = 0; dx < num_dofs; dx++) {
      cell_dat_const->set_value(cx, dx, 0, value++);
    }
  }

  value = 100.0;
  for (int cx = 0; cx < num_cells; cx++) {
    for (int dx = 0; dx < num_dofs; dx++) {
      ASSERT_EQ(cell_dat_const->get_value(cx, dx, 0), value++);
    }
  }

  std::vector<REAL> ext(num_cells * num_dofs);

  EventStack es;
  mapper->copy_to_external(cell_dat_const, ext.data(), es);

  es.wait();
  value = 100.0;
  for (int cx = 0; cx < num_cells; cx++) {
    for (int dx = 0; dx < num_dofs; dx++) {
      ASSERT_EQ(ext.at(lambda_host_order(cx, dx)), value++);
      ext.at(lambda_host_order(cx, dx)) += 100.0;
    }
  }

  mapper->copy_from_external(cell_dat_const, ext.data(), es);
  es.wait();
  for (int cx = 0; cx < num_cells; cx++) {
    for (int dx = 0; dx < num_dofs; dx++) {
      ASSERT_EQ(ext.at(lambda_host_order(cx, dx)),
                cell_dat_const->get_value(cx, dx, 0));
    }
  }

  sycl_target->free();
}

TEST(ExternalCommon, quadrature_point_mapper) {

  const int N = 32;
  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);
  auto qpm = std::make_shared<ExternalCommon::QuadraturePointMapper>(
      sycl_target, domain);

  auto owned_cells = mesh->get_owned_cells();
  ASSERT_EQ(owned_cells.size(), mesh->get_cell_count());

  std::vector<REAL> qpoints;
  const int NT = owned_cells.size() + N;
  qpoints.reserve(NT * ndim);
  const REAL h = mesh->cell_width_fine;
  for (auto &ox : owned_cells) {
    for (int dx = 0; dx < ndim; dx++) {
      qpoints.push_back((ox[dx] + 0.5) * h);
    }
  }

  std::mt19937 rng_pos(52234234);
  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  for (int px = 0; px < N; px++) {
    for (int dx = 0; dx < ndim; dx++) {
      qpoints.push_back(positions[dx][px]);
    }
  }

  ASSERT_FALSE(qpm->points_added());
  qpm->add_points_initialise();
  std::vector<REAL> point(ndim);
  for (int ix = 0; ix < NT * ndim; ix += ndim) {
    for (int dx = 0; dx < ndim; dx++) {
      point.at(dx) = qpoints.at(ix + dx);
    }
    qpm->add_point(point.data());
  }
  qpm->add_points_finalise();
  ASSERT_TRUE(qpm->points_added());

  // Create some data
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<INT>("GLOBAL_CELL_ID"), 1)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  const int cell_count = owned_cells.size();
  ParticleSet initial_distribution(cell_count, particle_spec);
  for (int px = 0; px < cell_count; px++) {
    for (int dx = 0; dx < ndim; dx++) {
      initial_distribution[Sym<REAL>("P")][px][dx] = qpoints.at(px * ndim + dx);
    }
  }
  A->add_particles_local(initial_distribution);
  A->cell_move();

  // Assemble some known data onto a CellDatConst in each cell
  auto mapper = std::make_shared<MeshHierarchyMapper>(
      sycl_target, mesh->get_mesh_hierarchy());
  auto k_mapper = mapper->get_device_mapper();
  auto cac_global_cell_index =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 1, 1);
  cac_global_cell_index->fill(0);
  particle_loop(
      A,
      [=](auto P, auto G, auto C) {
        REAL position[3];
        for (int dx = 0; dx < ndim; dx++) {
          position[dx] = P.at(dx);
        }
        INT global_cell[6];
        k_mapper.map_to_tuple(position, global_cell);
        const INT linear_global_cell =
            k_mapper.tuple_to_linear_global(global_cell);
        C.fetch_add(0, 0, (REAL)linear_global_cell);
        G.at(0) = linear_global_cell;
      },
      Access::read(Sym<REAL>("P")), Access::write(Sym<INT>("GLOBAL_CELL_ID")),
      Access::add(cac_global_cell_index))
      ->execute();

  // Copy the data into the quadrature point values
  particle_loop(
      qpm->particle_group, [=](auto Q, auto C) { Q.at(0) = C.at(0, 0); },
      Access::write(qpm->get_sym(1)), Access::read(cac_global_cell_index))
      ->execute();

  std::vector<REAL> output(NT);
  qpm->get(1, output);

  // For each point compute the global mesh hiererachy cell and hence get the
  // value that should have been returned.

  auto h_mapper = mapper->get_host_mapper();
  REAL position[3];
  INT global_index[6];

  for (int px = 0; px < NT; px++) {
    for (int dx = 0; dx < ndim; dx++) {
      position[dx] = qpoints.at(px * ndim + dx);
    }
    h_mapper.map_to_tuple(position, global_index);
    const REAL index = h_mapper.tuple_to_linear_global(global_index);
    const REAL to_test = output.at(px);

    ASSERT_NEAR(to_test, index, 1.0e-15);
  }

  // test the set call
  for (auto &ox : output) {
    ox *= 3.14;
  }
  qpm->set(1, output);

  // write to the CellDatConst for each cell.
  cac_global_cell_index->fill(0);
  particle_loop(
      qpm->particle_group,
      [=](auto Q, auto O, auto C) {
        if (O.at(3) == 0) {
          C.at(0, 0) = Q.at(0);
        }
      },
      Access::read(qpm->get_sym(1)),
      Access::read(Sym<INT>("ADDING_RANK_INDEX")),
      Access::write(cac_global_cell_index))
      ->execute();

  // Read from the test ParticleGroup
  particle_loop(
      A, [=](auto Q, auto C) { Q.at(0) = C.at(0, 0); },
      Access::write(Sym<REAL>("Q")), Access::read(cac_global_cell_index))
      ->execute();

  for (int cx = 0; cx < cell_count; cx++) {
    auto Q = A->get_cell(Sym<REAL>("Q"), cx);
    auto G = A->get_cell(Sym<INT>("GLOBAL_CELL_ID"), cx);
    const auto nrow = Q->nrow;
    for (int rx = 0; rx < nrow; rx++) {
      ASSERT_NEAR(Q->at(rx, 0), ((REAL)G->at(rx, 0)) * 3.14, 1.0e-14);
    }
  }

  qpm->free();
  mesh->free();
  sycl_target->free();
}

TEST(ExternalCommon, cartesian_to_barycentric_triangle) {
  {
    const REAL x1 = 0.0;
    const REAL y1 = 0.0;
    const REAL x2 = 2.0;
    const REAL y2 = 0.0;
    const REAL x3 = 0.0;
    const REAL y3 = 3.0;
    REAL l1, l2, l3, xx, yy;
    {
      const REAL x = 0.0;
      const REAL y = 0.0;
      ExternalCommon::triangle_cartesian_to_barycentric(x1, y1, x2, y2, x3, y3,
                                                        x, y, &l1, &l2, &l3);
      ASSERT_NEAR(l1, 1.0, 1.0e-10);
      ASSERT_NEAR(l2, 0.0, 1.0e-10);
      ASSERT_NEAR(l3, 0.0, 1.0e-10);
      ExternalCommon::triangle_barycentric_to_cartesian(x1, y1, x2, y2, x3, y3,
                                                        l1, l2, l3, &xx, &yy);
      ASSERT_NEAR(x, xx, 1.0e-10);
      ASSERT_NEAR(y, yy, 1.0e-10);
    }
    {
      const REAL x = 2.0;
      const REAL y = 0.0;
      ExternalCommon::triangle_cartesian_to_barycentric(x1, y1, x2, y2, x3, y3,
                                                        x, y, &l1, &l2, &l3);
      ASSERT_NEAR(l1, 0.0, 1.0e-10);
      ASSERT_NEAR(l2, 1.0, 1.0e-10);
      ASSERT_NEAR(l3, 0.0, 1.0e-10);
      ExternalCommon::triangle_barycentric_to_cartesian(x1, y1, x2, y2, x3, y3,
                                                        l1, l2, l3, &xx, &yy);
      ASSERT_NEAR(x, xx, 1.0e-10);
      ASSERT_NEAR(y, yy, 1.0e-10);
    }
    {
      const REAL x = 0.0;
      const REAL y = 3.0;
      ExternalCommon::triangle_cartesian_to_barycentric(x1, y1, x2, y2, x3, y3,
                                                        x, y, &l1, &l2, &l3);
      ASSERT_NEAR(l1, 0.0, 1.0e-10);
      ASSERT_NEAR(l2, 0.0, 1.0e-10);
      ASSERT_NEAR(l3, 1.0, 1.0e-10);
      ExternalCommon::triangle_barycentric_to_cartesian(x1, y1, x2, y2, x3, y3,
                                                        l1, l2, l3, &xx, &yy);
      ASSERT_NEAR(x, xx, 1.0e-10);
      ASSERT_NEAR(y, yy, 1.0e-10);
    }
    {
      const REAL x = 0.2;
      const REAL y = 0.3;
      ExternalCommon::triangle_cartesian_to_barycentric(x1, y1, x2, y2, x3, y3,
                                                        x, y, &l1, &l2, &l3);
      ASSERT_NEAR(std::abs(l1) + std::abs(l2) + std::abs(l3), 1.0, 1.0e-10);
      ExternalCommon::triangle_barycentric_to_cartesian(x1, y1, x2, y2, x3, y3,
                                                        l1, l2, l3, &xx, &yy);
      ASSERT_NEAR(x, xx, 1.0e-10);
      ASSERT_NEAR(y, yy, 1.0e-10);
    }

    // rotated vertices
    {
      const REAL x = x1;
      const REAL y = y1;
      ExternalCommon::triangle_cartesian_to_barycentric(x1, y1, x2, y2, x3, y3,
                                                        x, y, &l1, &l2, &l3);
      ASSERT_NEAR(l1, 1.0, 1.0e-10);
      ASSERT_NEAR(l2, 0.0, 1.0e-10);
      ASSERT_NEAR(l3, 0.0, 1.0e-10);
    }
    {
      const REAL x = x1;
      const REAL y = y1;
      ExternalCommon::triangle_cartesian_to_barycentric(x3, y3, x1, y1, x2, y2,
                                                        x, y, &l1, &l2, &l3);
      ASSERT_NEAR(l1, 0.0, 1.0e-10);
      ASSERT_NEAR(l2, 1.0, 1.0e-10);
      ASSERT_NEAR(l3, 0.0, 1.0e-10);
    }
    {
      const REAL x = x1;
      const REAL y = y1;
      ExternalCommon::triangle_cartesian_to_barycentric(x2, y2, x3, y3, x1, y1,
                                                        x, y, &l1, &l2, &l3);
      ASSERT_NEAR(l1, 0.0, 1.0e-10);
      ASSERT_NEAR(l2, 0.0, 1.0e-10);
      ASSERT_NEAR(l3, 1.0, 1.0e-10);
    }
  }
}

TEST(ExternalCommon, cartesian_to_collapsed_quad) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  auto lambda_wrap_call_inner = [&](const REAL x0, const REAL y0, const REAL x1,
                                    const REAL y1, const REAL x2, const REAL y2,
                                    const REAL x3, const REAL y3, const REAL x,
                                    const REAL y, REAL *RESTRICT eta0,
                                    REAL *RESTRICT eta1) {
    ExternalCommon::quad_cartesian_to_collapsed(x0, y0, x1, y1, x2, y2, x3, y3,
                                                x, y, eta0, eta1);
  };

  std::array<REAL, 2> v0;
  std::array<REAL, 2> v1;
  std::array<REAL, 2> v2;
  std::array<REAL, 2> v3;
  auto lambda_wrap_call = [&](const REAL x, const REAL y, REAL *RESTRICT eta0,
                              REAL *RESTRICT eta1) {
    lambda_wrap_call_inner(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], v3[0],
                           v3[1], x, y, eta0, eta1);
  };

  auto lambda_wrap_forward_call = [&](const REAL eta0, const REAL eta1, REAL *x,
                                      REAL *y) {
    ExternalCommon::quad_collapsed_to_cartesian(v0[0], v0[1], v1[0], v1[1],
                                                v2[0], v2[1], v3[0], v3[1],
                                                eta0, eta1, x, y);
  };

  // rotate vertices
  auto lambda_test_rotate = [&]() {
    const REAL x0 = v0[0];
    const REAL y0 = v0[1];
    const REAL x1 = v1[0];
    const REAL y1 = v1[1];
    const REAL x2 = v2[0];
    const REAL y2 = v2[1];
    const REAL x3 = v3[0];
    const REAL y3 = v3[1];
    const REAL x = x0;
    const REAL y = y0;
    REAL eta0, eta1;
    ExternalCommon::quad_cartesian_to_collapsed(x0, y0, x1, y1, x2, y2, x3, y3,
                                                x, y, &eta0, &eta1);
    ASSERT_NEAR(eta0, -1.0, 1.0e-15);
    ASSERT_NEAR(eta1, -1.0, 1.0e-15);

    ExternalCommon::quad_cartesian_to_collapsed(x3, y3, x0, y0, x1, y1, x2, y2,
                                                x, y, &eta0, &eta1);
    ASSERT_NEAR(eta0, 1.0, 1.0e-15);
    ASSERT_NEAR(eta1, -1.0, 1.0e-15);

    ExternalCommon::quad_cartesian_to_collapsed(x2, y2, x3, y3, x0, y0, x1, y1,
                                                x, y, &eta0, &eta1);
    ASSERT_NEAR(eta0, 1.0, 1.0e-15);
    ASSERT_NEAR(eta1, 1.0, 1.0e-15);

    ExternalCommon::quad_cartesian_to_collapsed(x1, y1, x2, y2, x3, y3, x0, y0,
                                                x, y, &eta0, &eta1);
    ASSERT_NEAR(eta0, -1.0, 1.0e-15);
    ASSERT_NEAR(eta1, 1.0, 1.0e-15);
  };

  REAL eta0 = -100.0;
  REAL eta1 = -100.0;

  v0 = {0.0, 0.0};
  v1 = {1.0, 0.0};
  v2 = {1.0, 1.0};
  v3 = {0.0, 1.0};
  lambda_wrap_call(0.0, 0.0, &eta0, &eta1);
  ASSERT_NEAR(eta0, -1.0, 1.0e-15);
  ASSERT_NEAR(eta1, -1.0, 1.0e-15);
  lambda_wrap_call(1.0, 0.0, &eta0, &eta1);
  ASSERT_NEAR(eta0, 1.0, 1.0e-15);
  ASSERT_NEAR(eta1, -1.0, 1.0e-15);
  lambda_wrap_call(1.0, 1.0, &eta0, &eta1);
  ASSERT_NEAR(eta0, 1.0, 1.0e-15);
  ASSERT_NEAR(eta1, 1.0, 1.0e-15);
  lambda_wrap_call(0.0, 1.0, &eta0, &eta1);
  ASSERT_NEAR(eta0, -1.0, 1.0e-15);
  ASSERT_NEAR(eta1, 1.0, 1.0e-15);
  lambda_wrap_call(0.5, 0.5, &eta0, &eta1);
  ASSERT_NEAR(eta0, 0.0, 1.0e-15);
  ASSERT_NEAR(eta1, 0.0, 1.0e-15);

  // slightly out of the element
  v0 = {-1.0, -1.0};
  v1 = {1.0, -1.0};
  v2 = {1.0, 1.0};
  v3 = {-1.0, 1.0};
  lambda_wrap_call(1.1, 1.1, &eta0, &eta1);
  ASSERT_NEAR(eta0, 1.1, 1.0e-15);
  ASSERT_NEAR(eta1, 1.1, 1.0e-15);
  lambda_wrap_call(-1.1, -1.1, &eta0, &eta1);
  ASSERT_NEAR(eta0, -1.1, 1.0e-15);
  ASSERT_NEAR(eta1, -1.1, 1.0e-15);

  // random tests
  std::uniform_real_distribution<double> uniform_rng(-1.0, 1.0);
  std::mt19937 rng = std::mt19937(34235481);

  auto lambda_test = [&]() {
    for (int testx = 0; testx < 10; testx++) {
      const REAL eta0 = uniform_rng(rng);
      const REAL eta1 = uniform_rng(rng);
      REAL eta0t, eta1t;
      REAL x, y;
      lambda_wrap_forward_call(eta0, eta1, &x, &y);
      lambda_wrap_call(x, y, &eta0t, &eta1t);
      ASSERT_NEAR(eta0, eta0t, 1.0e-15);
      ASSERT_NEAR(eta1, eta1t, 1.0e-15);
    }
  };

  // test on the reference element
  lambda_test();
  lambda_test_rotate();

  // test on a more complicated element
  v0 = {-4.0, -3.0};
  v1 = {2.3, -4.2};
  v2 = {3.4, 1.3};
  v3 = {-3.0, 2.3};
  lambda_test();
  lambda_test_rotate();
}

TEST(ExternalCommon, cartesian_to_barycentric_quad) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::array<REAL, 2> v0;
  std::array<REAL, 2> v1;
  std::array<REAL, 2> v2;
  std::array<REAL, 2> v3;

  auto lambda_wrap_call_inner = [&](const REAL x0, const REAL y0, const REAL x1,
                                    const REAL y1, const REAL x2, const REAL y2,
                                    const REAL x3, const REAL y3, const REAL x,
                                    const REAL y, REAL *RESTRICT l0,
                                    REAL *RESTRICT l1, REAL *RESTRICT l2,
                                    REAL *RESTRICT l3) {
    ExternalCommon::quad_cartesian_to_barycentric(x0, y0, x1, y1, x2, y2, x3,
                                                  y3, x, y, l0, l1, l2, l3);
  };

  auto lambda_wrap_call = [&](const REAL x, const REAL y, REAL *RESTRICT l0,
                              REAL *RESTRICT l1, REAL *RESTRICT l2,
                              REAL *RESTRICT l3) {
    lambda_wrap_call_inner(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], v3[0],
                           v3[1], x, y, l0, l1, l2, l3);
  };

  v0 = {0.0, 0.0};
  v1 = {1.0, 0.0};
  v2 = {1.0, 1.0};
  v3 = {0.0, 1.0};
  REAL l0, l1, l2, l3;
  lambda_wrap_call(0.0, 0.0, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 1.0, 1.0e-15);
  ASSERT_NEAR(l1, 0.0, 1.0e-15);
  ASSERT_NEAR(l2, 0.0, 1.0e-15);
  ASSERT_NEAR(l3, 0.0, 1.0e-15);
  lambda_wrap_call(1.0, 0.0, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 0.0, 1.0e-15);
  ASSERT_NEAR(l1, 1.0, 1.0e-15);
  ASSERT_NEAR(l2, 0.0, 1.0e-15);
  ASSERT_NEAR(l3, 0.0, 1.0e-15);
  lambda_wrap_call(1.0, 1.0, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 0.0, 1.0e-15);
  ASSERT_NEAR(l1, 0.0, 1.0e-15);
  ASSERT_NEAR(l2, 1.0, 1.0e-15);
  ASSERT_NEAR(l3, 0.0, 1.0e-15);
  lambda_wrap_call(0.0, 1.0, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 0.0, 1.0e-15);
  ASSERT_NEAR(l1, 0.0, 1.0e-15);
  ASSERT_NEAR(l2, 0.0, 1.0e-15);
  ASSERT_NEAR(l3, 1.0, 1.0e-15);
  lambda_wrap_call(0.5, 0.5, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 0.25, 1.0e-15);
  ASSERT_NEAR(l1, 0.25, 1.0e-15);
  ASSERT_NEAR(l2, 0.25, 1.0e-15);
  ASSERT_NEAR(l3, 0.25, 1.0e-15);
  lambda_wrap_call(0.0, 0.5, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 0.5, 1.0e-15);
  ASSERT_NEAR(l1, 0.0, 1.0e-15);
  ASSERT_NEAR(l2, 0.0, 1.0e-15);
  ASSERT_NEAR(l3, 0.5, 1.0e-15);
  lambda_wrap_call(1.0, 0.5, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 0.0, 1.0e-15);
  ASSERT_NEAR(l1, 0.5, 1.0e-15);
  ASSERT_NEAR(l2, 0.5, 1.0e-15);
  ASSERT_NEAR(l3, 0.0, 1.0e-15);
  lambda_wrap_call(0.5, 0.0, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 0.5, 1.0e-15);
  ASSERT_NEAR(l1, 0.5, 1.0e-15);
  ASSERT_NEAR(l2, 0.0, 1.0e-15);
  ASSERT_NEAR(l3, 0.0, 1.0e-15);
  lambda_wrap_call(0.5, 1.0, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 0.0, 1.0e-15);
  ASSERT_NEAR(l1, 0.0, 1.0e-15);
  ASSERT_NEAR(l2, 0.5, 1.0e-15);
  ASSERT_NEAR(l3, 0.5, 1.0e-15);
  lambda_wrap_call(0.75, 0.80, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 0.25 * 0.2, 1.0e-15);
  ASSERT_NEAR(l1, 0.75 * 0.2, 1.0e-15);
  ASSERT_NEAR(l2, 0.75 * 0.8, 1.0e-15);
  ASSERT_NEAR(l3, 0.25 * 0.8, 1.0e-15);

  v0 = {1.0, -0.25};
  v1 = {1.0, -0.125};
  v2 = {0.8, -0.125};
  v3 = {0.8, -0.25};

  REAL x = 1.0;
  REAL y = -0.25;

  lambda_wrap_call(x, y, &l0, &l1, &l2, &l3);
  ASSERT_NEAR(l0, 1.0, 1.0e-15);
  ASSERT_NEAR(l1, 0.0, 1.0e-15);
  ASSERT_NEAR(l2, 0.0, 1.0e-15);
  ASSERT_NEAR(l3, 0.0, 1.0e-15);
}

TEST(VTK, VTKHDF) {
  VTK::VTKHDF vtkhdf("test_external_common.vtkhdf", MPI_COMM_WORLD);
  int rank;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  std::vector<VTK::UnstructuredCell> data(2);
  {
    data.at(0).cell_type = VTK::CellType::triangle;
    std::vector<double> points = {0.0, 0.0, static_cast<double>(rank),
                                  1.0, 0.0, static_cast<double>(rank),
                                  0.0, 1.0, static_cast<double>(rank)};
    std::vector<double> point_data = {
        1.0 + rank,
        2.0 + rank,
        3.0 + rank,
    };
    data.at(0).num_points = 3;
    data.at(0).points = points;
    data.at(0).point_data["value"] = point_data;
  }

  {
    data.at(1).cell_type = VTK::CellType::quadrilateral;
    std::vector<double> points = {1.0, 0.0, static_cast<double>(rank),
                                  0.0, 1.0, static_cast<double>(rank),
                                  1.0, 2.0, static_cast<double>(rank),
                                  2.0, 1.0, static_cast<double>(rank)};
    std::vector<double> point_data = {
        1.0 + rank,
        2.0 + rank,
        3.0 + rank,
        4.0 + rank,
    };
    data.at(1).num_points = 4;
    data.at(1).points = points;
    data.at(1).point_data["value"] = point_data;
  }

  vtkhdf.write(data);
  vtkhdf.close();
}
