#ifdef NESO_PARTICLES_PETSC
#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>

using namespace NESO::Particles;

TEST(PETSc, matrix_invert_3) {
  PETSCCHK(PetscInitializeNoArguments());

  const REAL M[9] = {
      0.6210163141855796, 0.3190160150713188, 0.2571805706556594,
      0.5936442375675933, 0.991744920278965,  0.6897101781042996,
      0.5889379079562244, 0.8198441731417937, 0.8429227586052362};

  const REAL Lcorrect[9] = {
      2.396111040073248,   -0.5142632694765471, -0.3102782441062093,
      -0.834399945091885,  3.2951384763344302,  -2.441622402938384,
      -0.8625733291639001, -2.8456117965337913, 3.777907865839368};

  REAL L[9];
  PetscInterface::invert_matrix(3, M, L);

  for (int ix = 0; ix < 9; ix++) {
    ASSERT_NEAR(L[ix], Lcorrect[ix], 1.0e-14);
  }

  PETSCCHK(PetscFinalize());
}

TEST(PETSc, matrix_invert_4) {
  PETSCCHK(PetscInitializeNoArguments());

  const REAL M[16] = {
      0.0946422072387985, 0.8120429190962001, 0.4942494461121053,
      0.9178125430602081, 0.2850375256128153, 0.1463457132200754,
      0.4322885831714008, 0.4254072817319338, 0.7713729647584051,
      0.9431779836320303, 0.0572777823803825, 0.3918158930684735,
      0.1392738874243453, 0.1285321501053717, 0.8270943309693088,
      0.4699009249254635};

  const REAL Lcorrect[16] = {
      -1.0656931725086511, 2.527708148587856,  0.6244657934921137,
      -0.727545948845295,  0.4636364699948946, -3.6742381021991775,
      1.0168260293649087,  1.572901090240109,  -0.4912991088190402,
      -2.238254316420641,  0.4099585614121688, 2.64409330408957,
      1.0538003226407844,  4.1954832071005805, -1.1848051089178082,
      -2.7404814941081126};

  REAL L[16];
  PetscInterface::invert_matrix(4, M, L);

  for (int ix = 0; ix < 16; ix++) {
    ASSERT_NEAR(L[ix], Lcorrect[ix], 1.0e-14);
  }

  PETSCCHK(PetscFinalize());
}

TEST(PETSc, weighted_distribute) {
  PETSCCHK(PetscInitializeNoArguments());

  int size = 0;
  int rank = 0;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  const int ndim = 2;
  const int mesh_size = 128;
  PetscInt faces[3] = {mesh_size, 1};

  DM dm;
  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
      /* lower */ NULL,
      /* upper */ NULL,
      /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscSection weights_section;
  PETSCCHK(PetscSectionCreate(MPI_COMM_WORLD, &weights_section));

  PetscInt point_start = 0;
  PetscInt point_end = 0;
  PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));
  PETSCCHK(PetscSectionSetChart(weights_section, point_start, point_end));

  if (!rank) {

    for (PetscInt px = point_start; px < point_end; px++) {
      PETSCCHK(PetscSectionSetDof(weights_section, px, 0));
    }

    PETSCCHK(DMPlexGetHeightStratum(dm, 0, &point_start, &point_end));

    for (PetscInt px = point_start; px < point_end; px++) {
      PETSCCHK(PetscSectionSetDof(weights_section, px, 1));
    }
    PETSCCHK(PetscSectionSetDof(weights_section, point_start, 10));
  }
  PETSCCHK(PetscSectionSetUp(weights_section));
  PETSCCHK(DMSetLocalSection(dm, weights_section));

  PetscInterface::generic_distribute(&dm);

  PETSCCHK(DMPlexGetHeightStratum(dm, 0, &point_start, &point_end));
  const PetscInt cell_count = point_end - point_start;

  PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));
  const PetscInt point_count = point_end - point_start;

  for (int rx = 0; rx < size; rx++) {

    if (rx == rank) {
      nprint(rank, cell_count, point_count);
      std::cout << std::flush;
    }
    MPICHK(MPI_Barrier(MPI_COMM_WORLD));
  }

  PETSCCHK(PetscSectionDestroy(&weights_section));
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
