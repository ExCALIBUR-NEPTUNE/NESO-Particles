#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>
#include <string>

using namespace NESO::Particles;

TEST(PETSc, dmplex_interface_3d_base) {
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

  nprint("TODO fix ordering");
  // auto vtk_data = mesh_helper->get_vtk_cell_data();
  // VTK::VTKHDF vtkhdf("foo.vtkhdf", MPI_COMM_WORLD);
  // vtkhdf.write(vtk_data);
  // vtkhdf.close();

  std::vector<PetscScalar> point{0.1, 0.1, 0.1};
  mesh_helper->cell_contains_point_3d(0, point);

  nprint("TODO cleanup");
  for (int cellx = 0; cellx < mesh_helper->get_cell_count(); cellx++) {
    const bool contained = mesh_helper->cell_contains_point_3d(cellx, point);
    if (contained) {
      nprint("FOUND", cellx);
    }
  }

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  // ASSERT_TRUE(mesh->validate_halos(false));
  nprint("TODO revert to false");
  ASSERT_TRUE(mesh->validate_halos(true));

  const double volume = mesh->dmh->get_volume();
  ASSERT_NEAR(volume, 8.0, 1.0e-10);

  mesh->free();

  mesh_helper->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSc, dmplex_3d_mapper) {
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

  const int ndim = 3;
  int rank = -1;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  std::mt19937 rng_pos(52234234 + rank);

  REAL extents[3] = {2.0, 2.0, 2.0};

  const int cell_count = mesh->get_cell_count();
  const int N = 200 * cell_count;
  auto positions = uniform_within_extents(N, ndim, extents, rng_pos);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions[dimx][px] - 1.0;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }
  A->add_particles_local(initial_distribution);

  A->hybrid_move();
  A->cell_move();

  // auto vtk_data = mesh->dmh->get_vtk_cell_data();
  // VTK::VTKHDF vtkhdf("foo.vtkhdf", MPI_COMM_WORLD);
  // for(int cellx=0 ; cellx<cell_count ; cellx++){
  //   vtk_data.at(cellx).cell_data["CELL_ID"] = cellx;
  // }
  // vtkhdf.write(vtk_data);
  // vtkhdf.close();
  // H5Part h5part("bar.h5part", A, Sym<INT>("CELL_ID"));
  // const int Nsteps = 1;
  // for(int stepx=0 ; stepx<Nsteps ; stepx++){
  //   h5part.write();
  //   h5part.close();
  //   A->hybrid_move();
  //   A->cell_move();
  // }

  std::vector<PetscScalar> point(3);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto CELL_ID = A->get_cell(Sym<INT>("CELL_ID"), cellx);
    auto P = A->get_cell(Sym<REAL>("P"), cellx);
    const int nrow = CELL_ID->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      point[0] = P->at(rowx, 0);
      point[1] = P->at(rowx, 1);
      point[2] = P->at(rowx, 2);

      const bool host_bool = mesh->dmh->cell_contains_point(cellx, point);
      ASSERT_TRUE(host_bool);
    }
  }

  sycl_target->free();
  mesh->free();

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

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

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  {
    auto vtk_data = mesh->dmh->get_vtk_cell_data();

    for (auto &element : vtk_data) {
      const int num_vertices = element.points.size() / 3;
      for (int vx = 0; vx < num_vertices; vx++) {
        const REAL x = element.points.at(3 * vx + 0);
        const REAL y = element.points.at(3 * vx + 1);
        const REAL z = element.points.at(3 * vx + 2);
        element.point_data["x"].push_back(x);
        element.point_data["y"].push_back(y);
        element.point_data["z"].push_back(z);
      }
      element.cell_data["rank"] = rank;
    }

    VTK::VTKHDF vtkhdf("foo.vtkhdf", MPI_COMM_WORLD);
    vtkhdf.write(vtk_data);
    vtkhdf.close();
  }

  nprint("BEFORE HALO VTK");

  {
    auto vtk_data = mesh->dmh_halo->get_vtk_cell_data();

    for (auto &element : vtk_data) {
      element.cell_data["rank"] = rank;
    }

    VTK::VTKHDF vtkhdf("foo_halo_" + std::to_string(rank) + ".vtkhdf",
                       MPI_COMM_SELF);
    vtkhdf.write(vtk_data);
    vtkhdf.close();
  }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
