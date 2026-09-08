#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>

TEST(PETScDSMC, voronoi_cell_volume_cartesian_2d) {
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

  const int max_sub_division = 3;
  const int cell_count = mesh->get_cell_count();

  int global_offset = 0;
  MPICHK(MPI_Scan(&cell_count, &global_offset, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD));
  global_offset -= cell_count;

  int max_num_voronoi_cells = 0;
  auto points = std::make_shared<CellDat<REAL>>(sycl_target, cell_count, ndim);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int num_cells = ((global_offset + cellx) % max_sub_division) + 1;
    points->set_nrow(cellx, std::pow(num_cells, ndim));
    max_num_voronoi_cells = std::max(max_num_voronoi_cells, num_cells);
  }
  points->wait_set_nrow();

  EventStack event_stack;
  std::vector<CellData<REAL>> cell_data;
  cell_data.reserve(cell_count);
  std::vector<std::vector<REAL>> vertices;

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int num_cells = ((global_offset + cellx) % max_sub_division) + 1;
    const REAL h_inner = h / num_cells;
    const REAL offset = h_inner * 0.5;
    auto cellx_data = points->get_cell(cellx);
    cell_data.push_back(cellx_data);
    mesh->dmh->get_cell_vertices(cellx, vertices);

    std::array<REAL, 3> origin = {std::numeric_limits<REAL>::max(),
                                  std::numeric_limits<REAL>::max(),
                                  std::numeric_limits<REAL>::max()};
    for (auto &vx : vertices) {
      for (std::size_t dx = 0; dx < vx.size(); dx++) {
        origin.at(dx) = std::min(origin.at(dx), vx.at(dx));
      }
    }

    int point_index = 0;
    auto lambda_create_point = [&](auto idx) {
      for (int dx = 0; dx < ndim; dx++) {
        const REAL px = origin.at(dx) + offset + h_inner * idx.at(dx);
        cellx_data->at(point_index, dx) = px;
      }
      point_index++;
    };

    if (ndim == 2) {
      for (int iy = 0; iy < num_cells; iy++) {
        for (int ix = 0; ix < num_cells; ix++) {
          std::array<int, 2> idx = {ix, iy};
          lambda_create_point(idx);
        }
      }
    } else {
      for (int iz = 0; iz < num_cells; iz++) {
        for (int iy = 0; iy < num_cells; iy++) {
          for (int ix = 0; ix < num_cells; ix++) {
            std::array<int, 3> idx = {ix, iy, iz};
            lambda_create_point(idx);
          }
        }
      }
    }

    points->set_cell_async(cellx, cellx_data, event_stack);
  }
  event_stack.wait();

  auto subdivide_cells_voronoi =
      std::make_shared<SubdivideCellsVoronoi>(sycl_target, points);

  NDLocalArraySharedPtr<REAL, 2> volumes = nullptr;

  std::size_t min_num_samples = 10000;
  std::size_t num_samples = min_num_samples;
  const std::size_t max_num_samples = 1000000;
  REAL stol = 0.01;

  PetscInterface::estimate_voronoi_cell_volume(
      mesh, subdivide_cells_voronoi, num_samples, stol, max_num_samples,
      volumes, nullptr, 1024);
  ASSERT_TRUE(num_samples >=
              min_num_samples * std::pow(max_num_voronoi_cells, ndim));

  auto h_volumes = volumes->get();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int num_cells = ((global_offset + cellx) % max_sub_division) + 1;
    const int vcell_start = cellx * std::pow(max_num_voronoi_cells, ndim);
    const int vcell_end0 = vcell_start + std::pow(num_cells, ndim);
    const int vcell_end1 = vcell_start + std::pow(max_num_voronoi_cells, ndim);
    for (int vcellx = vcell_start; vcellx < vcell_end0; vcellx++) {
      const REAL volume_correct = std::pow((h / num_cells), ndim);
      const REAL volume_to_test = h_volumes.at(vcellx);
      const REAL err = relative_error(volume_correct, volume_to_test);
      ASSERT_TRUE(err < 0.10);
    }
    for (int vcellx = vcell_end0; vcellx < vcell_end1; vcellx++) {
      ASSERT_EQ(h_volumes.at(vcellx), -1.0);
    }
  }

  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
#endif
