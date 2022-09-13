#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
using namespace NESO::Particles;

TEST(LocalDecompositionHMesh, test_no_decomp_mesh) {

  const int ndim = 2;

  std::vector<double> extents(ndim);
  extents[0] = 2;
  extents[1] = 3;

  std::vector<double> origin(ndim);
  origin[0] = -1.0;
  origin[1] = -2.0;

  const int cell_count = 20 * 20;

  LocalDecompositionHMesh mesh(ndim, origin, extents, cell_count,
                               MPI_COMM_WORLD);

  ASSERT_EQ(mesh.get_cell_count(), cell_count);
  ASSERT_EQ(mesh.global_origin[0], -1.0);
  ASSERT_EQ(mesh.global_origin[1], -2.0);
  ASSERT_EQ(mesh.global_extents[0], 2.0);
  ASSERT_EQ(mesh.global_extents[1], 3.0);

  mesh.free();
}
