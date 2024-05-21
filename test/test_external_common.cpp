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
