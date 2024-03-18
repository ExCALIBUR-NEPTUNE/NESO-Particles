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
  DM dm_gmsh;
  PETSCCHK(DMPlexCreateGmshFromFile(PETSC_COMM_WORLD, gmsh_filename.c_str(),
                                    static_cast<PetscBool>(1), &dm_gmsh));

  DM dm;
  PetscSF sf;
  int size;
  MPICHK(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  if (size == 1) {
    PETSCCHK(DMClone(dm_gmsh, &dm));
  } else {
    PETSCCHK(DMPlexDistribute(dm_gmsh, 0, &sf, &dm));
  }
  PETSCCHK(DMDestroy(&dm_gmsh));

  auto mesh =
      std::make_shared<PETScInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  PetscInt start, end;
  PETSCCHK(DMPlexGetHeightStratum(dm, 0, &start, &end));
  ASSERT_EQ(mesh->get_cell_count(), end);
  ASSERT_EQ(mesh->get_ndim(), 2);

  mesh->free();
  PETSCCHK(PetscSFDestroy(&sf));
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
