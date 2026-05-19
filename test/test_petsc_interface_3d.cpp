#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>
#include <string>

using namespace NESO::Particles;

TEST(PETSc, foo) {
  std::filesystem::path gmsh_filepath;
  // GET_TEST_RESOURCE(gmsh_filepath,
  // "gmsh/reference_all_types_square_0.2.msh");

  nprint("TODO commit a mesh");
  gmsh_filepath = get_env_string("GMSH_TMP", "");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  int rank = -1;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  auto mesh_helper =
      std::make_shared<PetscInterface::DMPlexHelper>(MPI_COMM_WORLD, dm);

  auto vtk_data = mesh_helper->get_vtk_cell_data();

  VTK::VTKHDF vtkhdf("foo.vtkhdf", MPI_COMM_WORLD);
  vtkhdf.write(vtk_data);
  vtkhdf.close();

  // auto mesh =
  //     std::make_shared<PetscInterface::DMPlexInterface>(dm, 0,
  //     MPI_COMM_WORLD);
  //
  // const double volume = mesh->dmh->get_volume();
  // ASSERT_NEAR(volume, 4.0, 1.0e-10);
  //
  // mesh->free();

  mesh_helper->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
#endif
