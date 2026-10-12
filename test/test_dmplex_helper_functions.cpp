#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>
#include <neso_particles/external_interfaces/vtk/vtk.hpp>

using namespace NESO::Particles;

TEST(DMPlexHelper, set_partitioner_cell_ownership_map) {
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = (ndim == 2) ? 32 : 16;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  int size = -1;
  int rank = -1;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
      /* lower */ NULL,
      /* upper */ NULL,
      /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInt cell_start = -1;
  PetscInt cell_end = -1;
  PETSCCHK(DMPlexGetHeightStratum(dm, 0, &cell_start, &cell_end));

  std::map<PetscInt, int> map_cell_points_to_ranks;
  // We expect the box mesh to only exist on rank 0 at construction.

  PetscInt point_start = 0;
  PetscInt point_end = 0;
  PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));

  IS global_point_numbers;
  PETSCCHK(DMPlexCreatePointNumbering(dm, &global_point_numbers));
  const PetscInt *ptr;
  PETSCCHK(ISGetIndices(global_point_numbers, &ptr));

  std::vector<int> v_global_cell_points_to_ranks;

  auto lambda_new_rank = [&](auto global_point) {
    const int owning_rank = global_point % size;
    return owning_rank;
  };

  if (rank == 0) {

    std::map<PetscInt, int> map_global_cell_points_to_ranks;

    PetscInt max_global_point = 0;
    for (PetscInt cellx = cell_start; cellx < cell_end; cellx++) {
      const PetscInt global_point = ptr[cellx - point_start];
      const auto owning_rank = lambda_new_rank(global_point);
      map_cell_points_to_ranks[cellx] = owning_rank;
      max_global_point = std::max(global_point, max_global_point);
      map_global_cell_points_to_ranks[global_point] = owning_rank;
    }

    v_global_cell_points_to_ranks.resize(max_global_point + 1);

    for (auto &m : map_global_cell_points_to_ranks) {
      v_global_cell_points_to_ranks.at(m.first) = m.second;
    }
  } else {
    ASSERT_EQ(cell_start, 0);
    ASSERT_EQ(cell_end, 0);
  }

  PETSCCHK(ISRestoreIndices(global_point_numbers, &ptr));
  PETSCCHK(ISDestroy(&global_point_numbers));

  {
    int num_cells = v_global_cell_points_to_ranks.size();
    MPICHK(MPI_Bcast(&num_cells, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (rank) {
      v_global_cell_points_to_ranks.resize(num_cells);
    }
    MPICHK(MPI_Bcast(v_global_cell_points_to_ranks.data(), num_cells, MPI_INT,
                     0, MPI_COMM_WORLD));
  }

  PetscInterface::set_partitioner_cell_ownership(&dm, map_cell_points_to_ranks);

  PetscSF sf = nullptr;
  PetscInterface::generic_distribute(&dm, MPI_COMM_WORLD, 0, &sf);
  auto points_map = PetscInterface::get_global_distributed_points_map(dm, sf);

  std::vector<PetscInt> map_new_to_old(points_map.size());
  const PetscInt num_points = points_map.size();
  for (PetscInt px = 0; px < num_points; px++) {
    const PetscInt old = points_map.at(px);
    if (old > -1) {
      map_new_to_old.at(points_map.at(px)) = px;
    }
  }

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  const int cell_count = mesh->get_cell_count();
  //{
  //  auto vtk_cell_data = mesh->dmh->get_vtk_cell_data();
  //  for (int cellx = 0; cellx < cell_count; cellx++) {
  //    vtk_cell_data.at(cellx).cell_data["rank"] = rank;
  //  }
  //
  //  VTK::VTKHDF vtkhdf("owning_ranks.vtkhdf", MPI_COMM_WORLD);
  //  vtkhdf.write(vtk_cell_data);
  //  vtkhdf.close();
  //}

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const PetscInt point_index = mesh->dmh->get_cell_point_index(cellx);
    const PetscInt global_index_new =
        mesh->dmh->get_point_global_index(point_index);
    const PetscInt global_index_old = map_new_to_old.at(global_index_new);
    const auto owning_rank = lambda_new_rank(global_index_old);
    ASSERT_EQ(owning_rank, rank);
  }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(DMPlexHelper, set_partitioner_cell_ownership_label) {
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = (ndim == 2) ? 32 : 16;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};

  int size = -1;
  int rank = -1;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
      /* lower */ NULL,
      /* upper */ NULL,
      /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInt cell_start = -1;
  PetscInt cell_end = -1;
  PETSCCHK(DMPlexGetHeightStratum(dm, 0, &cell_start, &cell_end));

  PetscInt point_start = 0;
  PetscInt point_end = 0;
  PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));

  IS global_point_numbers;
  PETSCCHK(DMPlexCreatePointNumbering(dm, &global_point_numbers));
  const PetscInt *ptr;
  PETSCCHK(ISGetIndices(global_point_numbers, &ptr));

  std::vector<int> v_global_cell_points_to_ranks;

  auto lambda_new_rank = [&](auto global_point) {
    const int owning_rank = global_point % size;
    return owning_rank;
  };

  PETSCCHK(DMCreateLabel(dm, "test_partition"));

  if (rank == 0) {

    std::map<PetscInt, int> map_global_cell_points_to_ranks;

    DMLabel label;
    PETSCCHK(DMGetLabel(dm, "test_partition", &label));

    PetscInt max_global_point = 0;
    for (PetscInt cellx = cell_start; cellx < cell_end; cellx++) {
      const PetscInt global_point = ptr[cellx - point_start];
      const PetscInt owning_rank = lambda_new_rank(global_point);
      PETSCCHK(DMLabelSetValue(label, cellx, owning_rank));
      max_global_point = std::max(global_point, max_global_point);
      map_global_cell_points_to_ranks[global_point] = owning_rank;
    }

    v_global_cell_points_to_ranks.resize(max_global_point + 1);

    for (auto &m : map_global_cell_points_to_ranks) {
      v_global_cell_points_to_ranks.at(m.first) = m.second;
    }
  } else {
    ASSERT_EQ(cell_start, 0);
    ASSERT_EQ(cell_end, 0);
  }

  PETSCCHK(ISRestoreIndices(global_point_numbers, &ptr));
  PETSCCHK(ISDestroy(&global_point_numbers));

  {
    int num_cells = v_global_cell_points_to_ranks.size();
    MPICHK(MPI_Bcast(&num_cells, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (rank) {
      v_global_cell_points_to_ranks.resize(num_cells);
    }
    MPICHK(MPI_Bcast(v_global_cell_points_to_ranks.data(), num_cells, MPI_INT,
                     0, MPI_COMM_WORLD));
  }

  PetscInterface::set_partitioner_from_label(&dm, "test_partition");

  PetscSF sf = nullptr;
  PetscInterface::generic_distribute(&dm, MPI_COMM_WORLD, 0, &sf);
  auto points_map = PetscInterface::get_global_distributed_points_map(dm, sf);

  std::vector<PetscInt> map_new_to_old(points_map.size());
  const PetscInt num_points = points_map.size();
  for (PetscInt px = 0; px < num_points; px++) {
    const PetscInt old = points_map.at(px);
    if (old > -1) {
      map_new_to_old.at(points_map.at(px)) = px;
    }
  }

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  const int cell_count = mesh->get_cell_count();
  //{
  //  auto vtk_cell_data = mesh->dmh->get_vtk_cell_data();
  //  for (int cellx = 0; cellx < cell_count; cellx++) {
  //    vtk_cell_data.at(cellx).cell_data["rank"] = rank;
  //  }
  //
  //  VTK::VTKHDF vtkhdf("owning_ranks.vtkhdf", MPI_COMM_WORLD);
  //  vtkhdf.write(vtk_cell_data);
  //  vtkhdf.close();
  //}

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const PetscInt point_index = mesh->dmh->get_cell_point_index(cellx);
    const PetscInt global_index_new =
        mesh->dmh->get_point_global_index(point_index);
    const PetscInt global_index_old = map_new_to_old.at(global_index_new);
    const auto owning_rank = lambda_new_rank(global_index_old);
    ASSERT_EQ(owning_rank, rank);
  }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
