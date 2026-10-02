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

/**
 * Get the point indices of edges that form a loop around the boundary. This
 * functions finds an edge loop of the mesh on this MPI rank, i.e. the mesh is
 * not distributed.
 *
 * @param dm Input DMPlex to find boundary of.
 * @returns Vector that forms the edge loop.
 */
std::vector<PetscInt> get_boundary_edge_loop(DM &dm) {

  PetscInt edge_start = -1;
  PetscInt edge_end = -1;
  PETSCCHK(DMPlexGetDepthStratum(dm, 1, &edge_start, &edge_end));

  // Find a starting edge on the boundary.
  PetscInt start_edge = -1;
  for (PetscInt edgex = edge_start; edgex < edge_end; edgex++) {
    PetscInt support_size = 0;
    PETSCCHK(DMPlexGetSupportSize(dm, edgex, &support_size));
    // The boundary edges have support size of 1
    if (support_size == 1) {
      start_edge = edgex;
      break;
    }
  }

  NESOASSERT(start_edge != -1, "Failed to find a starting edge.");

  PetscInt current_edge = start_edge;
  PetscInt prev_vertex = -1;

  std::vector<PetscInt> edge_loop = {start_edge};
  std::set<PetscInt> seen_points;
  seen_points.insert(start_edge);

  {
    PetscInt cone_size = 0;
    PETSCCHK(DMPlexGetConeSize(dm, current_edge, &cone_size));
    NESOASSERT(cone_size == 2, "Expected cone to be of size two.");
    const PetscInt *cone = nullptr;
    PETSCCHK(DMPlexGetCone(dm, current_edge, &cone));
    prev_vertex = cone[0];
  }

  const PetscInt total_num_edges = edge_end - edge_start;

  for (PetscInt outer_edge = 0; outer_edge < total_num_edges; outer_edge++) {
    PetscInt cone_size = 0;
    PETSCCHK(DMPlexGetConeSize(dm, current_edge, &cone_size));
    NESOASSERT(cone_size == 2, "Expected cone to be of size two.");
    const PetscInt *cone = nullptr;
    PETSCCHK(DMPlexGetCone(dm, current_edge, &cone));
    const PetscInt next_vertex = (cone[0] == prev_vertex) ? cone[1] : cone[0];

    PetscInt next_edge = -1;
    PetscInt support_size = -1;
    PETSCCHK(DMPlexGetSupportSize(dm, next_vertex, &support_size));
    const PetscInt *support = nullptr;
    PETSCCHK(DMPlexGetSupport(dm, next_vertex, &support));

    for (PetscInt edgei = 0; edgei < support_size; edgei++) {
      // One of the edges that touches this vertex is the previous edge and
      // another is the edge we want to traverse along. All other edges have
      // support size two.
      const PetscInt edgex = support[edgei];
      PetscInt support_size_edge = 0;
      PETSCCHK(DMPlexGetSupportSize(dm, edgex, &support_size_edge));

      if (support_size_edge == 1) {
        const PetscInt *cone = nullptr;
        PETSCCHK(DMPlexGetCone(dm, edgex, &cone));
        const bool edge_points_backwards =
            (cone[0] == prev_vertex) || (cone[1] == prev_vertex);

        // If the edge has support size 1 and the cone does not have the
        // previous vertex then this is the next edge.
        if (!edge_points_backwards) {
          next_edge = edgex;
          break;
        }
      }
    }

    NESOASSERT(next_edge != -1, "Failed to find the next edge.");

    prev_vertex = next_vertex;
    current_edge = next_edge;

    NESOASSERT(seen_points.count(next_edge) == 0 ||
                   (current_edge == start_edge),
               "This edge has already been seen.");
    seen_points.insert(next_edge);

    if (current_edge == start_edge) {
      break;
    } else {
      edge_loop.push_back(next_edge);
    }
  }

  return edge_loop;
}

std::vector<PetscInt>
get_boundary_cell_loop(DM &dm, std::vector<PetscInt> &boundary_edge_loop) {

  std::set<PetscInt> seen_cells;
  std::vector<PetscInt> cell_loop;
  cell_loop.reserve(boundary_edge_loop.size());

  for (PetscInt edgex : boundary_edge_loop) {
    PetscInt support_size = -1;
    PETSCCHK(DMPlexGetSupportSize(dm, edgex, &support_size));
    NESOASSERT(support_size == 1, "This edge is not a boundary edge.");
    const PetscInt *support = nullptr;
    PETSCCHK(DMPlexGetSupport(dm, edgex, &support));
    const PetscInt cell = support[0];

    if (!seen_cells.count(cell)) {
      cell_loop.push_back(cell);
    }
  }

  return cell_loop;
}

std::map<PetscInt, int>
partition_with_uniform_boundary(DM &dm, const int num_partitions) {

  MPI_Comm comm;
  PETSCCHK(PetscObjectGetComm((PetscObject)dm, &comm));
  int rank = 0;
  MPICHK(MPI_Comm_rank(comm, &rank));

  PetscInt cell_start = -1;
  PetscInt cell_end = -1;
  PETSCCHK(DMPlexGetHeightStratum(dm, 0, &cell_start, &cell_end));

  std::map<PetscInt, int> map_cell_points_to_ranks;
  for (PetscInt cellx = cell_start; cellx < cell_end; cellx++) {
    map_cell_points_to_ranks[cellx] = 0;
  }

  const PetscInt num_cells = cell_end - cell_start;
  std::vector<int> cell_owning_ranks(num_cells);
  std::fill(cell_owning_ranks.begin(), cell_owning_ranks.end(), 0);
  std::map<int, std::set<PetscInt>> map_ranks_to_cells;

  if (rank == 0) {
    auto edge_loop = get_boundary_edge_loop(dm);

    const std::size_t edge_loop_size = edge_loop.size();
    NESOASSERT(edge_loop_size > 0, "No edge loop found.");
    auto cell_loop = get_boundary_cell_loop(dm, edge_loop);
    const int cell_loop_size = cell_loop.size();

    for (int partitionx = 0; partitionx < num_partitions; partitionx++) {
      int start = 0;
      int end = 0;
      get_decomp_1d(num_partitions, cell_loop_size, partitionx, &start, &end);
      for (int cellx = start; cellx < end; cellx++) {
        const PetscInt cell_point = cell_loop.at(cellx);
        map_cell_points_to_ranks[cell_point] = partitionx;
      }
    }
  }

  return map_cell_points_to_ranks;
}

TEST(DMPlexHelper, partition_even_boundary) {
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

  auto m = partition_with_uniform_boundary(dm, size);

  PetscInterface::set_partitioner_cell_ownership(&dm, m);
  PetscInterface::generic_distribute(&dm, MPI_COMM_WORLD, 0);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  const int cell_count = mesh->get_cell_count();
  {
    auto vtk_cell_data = mesh->dmh->get_vtk_cell_data();
    for (int cellx = 0; cellx < cell_count; cellx++) {
      vtk_cell_data.at(cellx).cell_data["rank"] = rank;
    }

    VTK::VTKHDF vtkhdf("owning_ranks_partitioner.vtkhdf", MPI_COMM_WORLD);
    vtkhdf.write(vtk_cell_data);
    vtkhdf.close();
  }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
