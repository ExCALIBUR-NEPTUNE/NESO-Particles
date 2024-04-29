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
