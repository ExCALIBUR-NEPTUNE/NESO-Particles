#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"

/**
 * Create a 2D square mesh of NxN cells each 1x1. N is the number of MPI ranks.
 * MPI rank i owns the i-th row. All objects are indexed lexicographically from
 * zero row wise.
 */
TEST(PETSc, dmplex_from_existing_mesh_quads) {
  PETSCCHK(PetscInitializeNoArguments());
  auto sycl_target = std::make_shared<SYCLTarget>(0, PETSC_COMM_WORLD);
  const int mpi_size = sycl_target->comm_pair.size_parent;
  const int mpi_rank = sycl_target->comm_pair.rank_parent;

  // First we setup the topology of the mesh.
  PetscInt num_cells_owned = mpi_size;

  std::vector<PetscInt> cells;
  cells.reserve(num_cells_owned * 4);

  // We are careful to list the vertices in counter clock-wise order, this might
  // matter.
  for (PetscInt cx = 0; cx < num_cells_owned; cx++) {
    // These are global indices not local indices.
    PetscInt vertex_sw = cx + mpi_rank * (mpi_size + 1);
    PetscInt vertex_se = vertex_sw + 1;
    PetscInt vertex_ne = vertex_se + mpi_size + 1;
    PetscInt vertex_nw = vertex_ne - 1;
    cells.push_back(vertex_sw);
    cells.push_back(vertex_se);
    cells.push_back(vertex_ne);
    cells.push_back(vertex_nw);
  }

  /*
   * Each rank owns a contiguous block of global indices. We label our indices
   * lexicographically (row-wise). Sorting out the global vertex indexing is
   * probably one of the more tedious parts.
   */
  PetscInt num_vertices_owned = mpi_size + 1;
  if (mpi_rank == mpi_size - 1) {
    // Make the last rank own the top edge.
    num_vertices_owned += mpi_size + 1;
  }

  /*
   * Create the coordinates for the block of vertices we pass to petsc. For an
   * existing mesh in memory this step will probably involve some MPI
   * communication to gather the blocks of coordinates on the ranks which pass
   * them to PETSc.
   */
  std::vector<PetscScalar> vertex_coords(num_vertices_owned * 2);
  for (int px = 0; px < mpi_size + 1; px++) {
    // our cell extent is 1.0
    vertex_coords.at(px * 2 + 0) = px;
    vertex_coords.at(px * 2 + 1) = mpi_rank;
  }
  if (mpi_rank == mpi_size - 1) {
    for (int px = 0; px < mpi_size + 1; px++) {
      // our cell extent is 1.0
      vertex_coords.at((mpi_size + 1 + px) * 2 + 0) = px;
      vertex_coords.at((mpi_size + 1 + px) * 2 + 1) = mpi_rank + 1;
    }
  }

  // This DM will contain the DMPlex after we call the creation routine.
  DM dm;
  // Create the DMPlex from the cells and coordinates.
  PETSCCHK(DMPlexCreateFromCellListParallelPetsc(
      PETSC_COMM_WORLD, 2, num_cells_owned, num_vertices_owned, PETSC_DECIDE, 4,
      PETSC_TRUE, cells.data(), 2, vertex_coords.data(), NULL, NULL, &dm));

  /*
   *
   *
   *
   *
   *
   * Below here is testing of the DMPlex
   *
   *
   *
   *
   *
   *
   */

  // Create the map from local point numbering to global point numbering.
  IS global_point_numbers;
  PETSCCHK(DMPlexCreatePointNumbering(dm, &global_point_numbers));

  // Get the DMPlex point start/end for the cells this process owns in the
  // DMPlex
  PetscInt cell_local_start, cell_local_end;
  PETSCCHK(DMPlexGetDepthStratum(dm, 2, &cell_local_start, &cell_local_end));
  // Map the local cell indices to global indices
  PetscInt point_start, point_end;
  PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));
  const PetscInt *ptr;
  PETSCCHK(ISGetIndices(global_point_numbers, &ptr));
  PetscInt local_index = 0;
  PetscInt correct_cell_index_start = mpi_rank * mpi_size;
  for (PetscInt point = cell_local_start; point < cell_local_end; point++) {
    PetscInt cell_index = ptr[point - point_start];
    ASSERT_EQ(correct_cell_index_start + local_index, cell_index);
    local_index++;
  }
  PETSCCHK(ISRestoreIndices(global_point_numbers, &ptr));

  // Check the vertices of the local cells are the vertices we expect.
  // This also checks that the ordering and parallel decomposition is what we
  // expect
  PetscInterface::DMPlexHelper dmh(PETSC_COMM_WORLD, dm);
  std::vector<std::vector<REAL>> vertices;
  std::set<std::vector<REAL>> vertices_to_test;
  std::set<std::vector<REAL>> vertices_correct;

  for (int cx = 0; cx < mpi_size; cx++) {
    dmh.get_cell_vertices(cx, vertices);
    vertices_to_test.clear();
    vertices_correct.clear();

    for (auto vx : vertices) {
      vertices_to_test.insert(vx);
    }

    vertices_correct.insert(
        {static_cast<REAL>(cx), static_cast<REAL>(mpi_rank)});
    vertices_correct.insert(
        {static_cast<REAL>(cx + 1), static_cast<REAL>(mpi_rank)});
    vertices_correct.insert(
        {static_cast<REAL>(cx), static_cast<REAL>(mpi_rank + 1)});
    vertices_correct.insert(
        {static_cast<REAL>(cx + 1), static_cast<REAL>(mpi_rank + 1)});
    ASSERT_EQ(vertices_correct, vertices_to_test);
  }

  dmh.free();

  PETSCCHK(ISDestroy(&global_point_numbers));
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
  sycl_target->free();
}

#endif
