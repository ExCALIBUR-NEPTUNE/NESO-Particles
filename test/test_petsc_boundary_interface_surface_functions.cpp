#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>

TEST(PETScBoundary2D, setup) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  const int ndim = 2;
  const int mesh_size = 8;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces, lower, upper,
      /* periodicity */ NULL, PETSC_TRUE, &dm));
  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2};
  boundary_groups[2] = {3, 4};

  auto b2d = std::make_shared<PetscInterface::BoundaryInteraction2D>(
      sycl_target, mesh, boundary_groups);

  auto f1 = b2d->create_function(1, "DG", 0);
  auto f2 = b2d->create_function(2, "DG", 0);

  std::set<int> s1 = {1, 2};
  std::set<int> s2 = {3, 4};
  auto face_sets = mesh->dmh->get_face_sets();

  std::set<PetscInt> cg1;
  std::set<PetscInt> cg2;

  for (auto labelx_indices : face_sets) {
    const auto label = labelx_indices.first;
    const auto &indices = labelx_indices.second;
    if (label > 0) {
      for (auto index : indices) {
        if (s1.count(label)) {
          cg1.insert(mesh->dmh->get_point_global_index(index));
        } else {
          ASSERT_TRUE(s2.count(label));
          cg2.insert(mesh->dmh->get_point_global_index(index));
        }
      }
    }
  }

  std::set<PetscInt> t1;
  std::set<PetscInt> t2;

  for (auto cx : f1->cells) {
    t1.insert(cx);
  }
  for (auto cx : f2->cells) {
    t2.insert(cx);
  }

  ASSERT_EQ(cg1, t1);
  ASSERT_EQ(cg2, t2);

  ASSERT_EQ(static_cast<std::size_t>(f1->local_dof_count), cg1.size());
  ASSERT_EQ(static_cast<std::size_t>(f2->local_dof_count), cg2.size());

  auto h_dofs1 = f1->get_dofs();
  ASSERT_EQ(h_dofs1.size(), cg1.size());
  auto h_dofs2 = f2->get_dofs();
  ASSERT_EQ(h_dofs2.size(), cg2.size());

  // int index = 0;
  // for (auto cx : f1->cells) {
  //   h_dofs1.at(index++) = cx;
  // }
  // f1->set_dofs(h_dofs1);
  // f1->write_vtkhdf("f1.vtkhdf");

  b2d->free();
  sycl_target->free();
  mesh->free();

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
