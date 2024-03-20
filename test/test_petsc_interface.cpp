#include <CL/sycl.hpp>
#include <external_interfaces/petsc/petsc_interface.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <string>

using namespace NESO::Particles;

TEST(PETSC, init) {

  int argc;
  char **argv;
  PETSCCHK(PetscInitialize(&argc, &argv, nullptr, nullptr));
  // TODO
  std::string gmsh_filename = "/home/js0259/git-ukaea/NESO-Particles-paper/"
                              "resources/mesh_ring/mesh_ring.msh";
  DM dm;
  // PETSCCHK(DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, gmsh_filename.c_str(),
  //                                   static_cast<PetscBool>(1), &dm));

  const int mesh_size = 32;
  PetscInt faces[2] = {mesh_size, mesh_size};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, faces,
                               /* lower */ NULL,
                               /* upper */ NULL,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  PetscInt start, end;
  PETSCCHK(DMPlexGetHeightStratum(dm, 0, &start, &end));
  ASSERT_EQ(mesh->get_cell_count(), end - start);
  ASSERT_EQ(mesh->get_ndim(), 2);

  auto mh = mesh->get_mesh_hierarchy();

  ASSERT_EQ(mh->dims[0], 1);
  ASSERT_EQ(mh->dims[1], 1);
  ASSERT_EQ(mh->ncells_dim_fine, mesh_size);

  // The mesh hierarchy ownership should match the mesh ownership as it is a
  // cartesian mesh.
  int rank;
  MPICHK(MPI_Comm_rank(mesh->get_comm(), &rank));

  const int cell_count = mesh->get_cell_count();
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  auto mh_mapper = std::make_shared<MeshHierarchyMapper>(sycl_target, mh);
  auto mapper = mh_mapper->get_host_mapper();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    // Test the bounding box contains the average of the vertices
    std::vector<REAL> cell_average(2);
    mesh->dmh->get_cell_vertex_average(cellx, cell_average);
    auto bb = mesh->dmh->get_cell_bounding_box(cellx);
    ASSERT_TRUE(bb->contains_point(2, cell_average));

    // Check cell claimed the corresponding MH cell
    INT mh_tuple[6];
    mapper.map_to_tuple(cell_average.data(), mh_tuple);
    const INT mh_linear_index = mapper.tuple_to_linear_global(mh_tuple);
    const int owner = mh->get_owner(mh_linear_index);
    ASSERT_EQ(owner, rank);
  }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
