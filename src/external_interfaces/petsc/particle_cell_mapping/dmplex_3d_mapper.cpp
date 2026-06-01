#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/external_interfaces/petsc/particle_cell_mapping/dmplex_3d_mapper.hpp>

namespace NESO::Particles::PetscInterface {

DMPlex3DMapper::DMPlex3DMapper(SYCLTargetSharedPtr sycl_target,
                               DMPlexInterfaceSharedPtr dmplex_interface)
    : sycl_target(sycl_target), dmplex_interface(dmplex_interface) {

  constexpr int ndim = 3;
  auto dmh = dmplex_interface->dmh;
  auto dmh_halo = dmplex_interface->dmh_halo;

  const int num_local_cells = dmh->get_cell_count();
  const int num_halo_cells = dmh_halo ? dmh_halo->get_cell_count() : 0;
  const int num_total_cells = num_local_cells + num_halo_cells;

  // Create a bounding box for halo cells and local cells.
  auto bounding_box = dmplex_interface->dmh->get_bounding_box();
  if (dmh_halo) {
    bounding_box->expand(dmh_halo->get_bounding_box());
  }

  // Create an overlayed Cartesian mesh.
  this->overlay_mesh = ExternalCommon::create_overlay_mesh(
      this->sycl_target, ndim, bounding_box, num_total_cells);

  // Make the lookup table for the vertex data
  this->cell_data =
      std::make_unique<LookupTable<int, Implementation3DLinear::Linear3DData>>(
          this->sycl_target, num_total_cells);

  // For each local and halo cell find the overlay cells they intersect with
  std::map<int, std::list<int>> map_overlay_cells;

  // Helper lambda to populate the cell data
  auto lambda_populate_cell_data =
      [&](DM &dm, PetscInt petsc_index, const int owning_rank,
          const int local_id) -> Implementation3DLinear::Linear3DData {
    Implementation3DLinear::Linear3DData tmp_data;

    PetscInt num_faces = 0;
    PETSCCHK(DMPlexGetConeSize(dm, petsc_index, &num_faces));

    const PetscScalar *tmp;
    PetscScalar *vertices = nullptr;
    PetscInt num_crossings = 0, num_coords;
    PetscBool is_dg;

    PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                      &tmp, &vertices));
    const int num_vertices = num_coords / 3;
    std::vector<PetscScalar> h_vertices;
    h_vertices.reserve(num_coords);
    for (PetscInt ix = 0; ix < num_coords; ix++) {
      h_vertices.push_back(vertices[ix]);
    }
    PETSCCHK(DMPlexRestoreCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                          &tmp, &vertices));

    const PetscInt *cone = nullptr;
    PETSCCHK(DMPlexGetCone(dm, petsc_index, &cone));

    for (PetscInt facex = 0; facex < num_faces; facex++) {
      const PetscInt face_petsc_index = cone[facex];

      PETSCCHK(DMPlexGetCellCoordinates(dm, face_petsc_index, &is_dg,
                                        &num_coords, &tmp, &vertices));
      NESOASSERT(num_coords == 12 || num_coords == 9, "Unexpected num coords.");
      const PetscInt c0 = 0;
      const PetscInt c1 = 1;
      const PetscInt c2 = num_coords == 12 ? 3 : 2;

      const PetscScalar v0[3] = {vertices[3 * c1 + 0] - vertices[3 * c0 + 0],
                                 vertices[3 * c1 + 1] - vertices[3 * c0 + 1],
                                 vertices[3 * c1 + 2] - vertices[3 * c0 + 2]};

      const PetscScalar v1[3] = {vertices[3 * c2 + 0] - vertices[3 * c0 + 0],
                                 vertices[3 * c2 + 1] - vertices[3 * c0 + 1],
                                 vertices[3 * c2 + 2] - vertices[3 * c0 + 2]};

      PetscScalar n[3] = {0.0, 0.0, 0.0};
      KERNEL_CROSS_PRODUCT_3D(v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], n[0],
                              n[1], n[2]);

      int direction = 0;

      for (PetscInt vx = 0; vx < num_vertices; vx++) {

        // vector from first vertex to test point
        const PetscScalar t0[3] = {h_vertices.at(3 * vx + 0) - vertices[0],
                                   h_vertices.at(3 * vx + 1) - vertices[1],
                                   h_vertices.at(3 * vx + 2) - vertices[2]};

        const PetscScalar t0_dot_n =
            KERNEL_DOT_PRODUCT_3D(t0[0], t0[1], t0[2], n[0], n[1], n[2]);

        if (std::fabs(t0_dot_n) > 1.0e-6) {
          const int to_test_direction = t0_dot_n > 0.0 ? 1 : -1;
          if (direction == 0) {
            direction = to_test_direction;
          } else {
            NESOASSERT(direction == to_test_direction,
                       "Inconsistent directions.");
          }
        }
      }

      NESOASSERT(direction != 0, "Could not determine direction");
      // normal points inwards
      if (direction > 0) {
        n[0] *= -1;
        n[1] *= -1;
        n[2] *= -1;
      }

      const REAL normalisation =
          1.0 / std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

      tmp_data.normal_origin[facex * 6 + 0 + 0] = normalisation * n[0];
      tmp_data.normal_origin[facex * 6 + 0 + 1] = normalisation * n[1];
      tmp_data.normal_origin[facex * 6 + 0 + 2] = normalisation * n[2];
      tmp_data.normal_origin[facex * 6 + 3 + 0] = vertices[0];
      tmp_data.normal_origin[facex * 6 + 3 + 1] = vertices[1];
      tmp_data.normal_origin[facex * 6 + 3 + 2] = vertices[2];

      PETSCCHK(DMPlexRestoreCellCoordinates(dm, face_petsc_index, &is_dg,
                                            &num_coords, &tmp, &vertices));
    }

    tmp_data.owning_rank = owning_rank;
    tmp_data.local_id = local_id;
    tmp_data.num_faces = num_faces;

    return tmp_data;
  };

  // Local cells
  int index = 0;
  const int num_cells_local = dmplex_interface->dmh->get_cell_count();
  std::vector<int> overlay_cells;
  for (int cx = 0; cx < num_cells_local; cx++) {
    auto bb = dmplex_interface->dmh->get_cell_bounding_box(cx);
    this->overlay_mesh->get_intersecting_cells(bb, overlay_cells);
    for (auto &ox : overlay_cells) {
      map_overlay_cells[ox].push_back(index);
    }
    // Record the description of this cell
    const PetscInt petsc_index =
        dmplex_interface->dmh->get_dmplex_cell_index(cx);
    auto tmp_data =
        lambda_populate_cell_data(dmplex_interface->dmh->dm, petsc_index,
                                  sycl_target->comm_pair.rank_parent, index);
    this->cell_data->add(index, tmp_data);
    index++;
  }

  // Halo cells
  if (dmh_halo) {
    const int num_cells_halo = dmplex_interface->dmh_halo->get_cell_count();
    for (int cx = 0; cx < num_cells_halo; cx++) {
      auto bb = dmh_halo->get_cell_bounding_box(cx);
      this->overlay_mesh->get_intersecting_cells(bb, overlay_cells);
      for (auto &ox : overlay_cells) {
        map_overlay_cells[ox].push_back(index);
      }
      // Record the description of this cell

      const PetscInt point_index =
          dmplex_interface->dmh_halo->get_dmplex_cell_index(cx);
      auto id_rank = dmplex_interface->map_local_lid_remote_lid.at(point_index);
      const PetscInt petsc_index =
          dmplex_interface->dmh_halo->get_dmplex_cell_index(cx);

      auto tmp_data =
          lambda_populate_cell_data(dmh_halo->dm, petsc_index,
                                    std::get<0>(id_rank), std::get<1>(id_rank));
      this->cell_data->add(index, tmp_data);
      index++;
    }
  }

  // Create the map from overlay cartesian cells to DMPlex cells
  const int overlay_cell_count = this->overlay_mesh->get_cell_count();
  this->map_sizes = std::make_unique<LookupTable<int, int>>(this->sycl_target,
                                                            overlay_cell_count);

  for (int cx = 0; cx < overlay_cell_count; cx++) {
    // The keys untouched above will have a default initialised list of size
    // 0.
    this->map_sizes->add(cx, map_overlay_cells[cx].size());
  }

  // Create the lookup table for candidate cells
  this->map_candidates = std::make_unique<LookupTable<int, int *>>(
      this->sycl_target, overlay_cell_count);

  // push the maps from overlay cells to candidate cells onto device
  std::vector<int> candidate_cells;
  for (int cx = 0; cx < overlay_cell_count; cx++) {
    const int num_candidates = map_overlay_cells.at(cx).size();
    if (num_candidates) {
      candidate_cells.clear();
      // Use .at now has the previous loop default initialised all the cells.
      candidate_cells.reserve(num_candidates);
      // Convert to std vector to make copying to device easier.
      candidate_cells.insert(candidate_cells.end(),
                             map_overlay_cells.at(cx).begin(),
                             map_overlay_cells.at(cx).end());
      // copy candidate cells to device
      auto tmp_ptr = std::make_unique<BufferDevice<int>>(this->sycl_target,
                                                         candidate_cells);
      // push the device pointer onto the lookup table
      this->map_candidates->add(cx, tmp_ptr->ptr);
      // push this unique ptr onto a stack to keep it in scope
      this->map_stack.push(std::move(tmp_ptr));
    }
  }

  // Checking loop that all particles were binned into cells.
  this->ep = std::make_unique<ErrorPropagate>(this->sycl_target);
}

void DMPlex3DMapper::map(ParticleGroup &particle_group, const int map_cell) {

  auto dat_positions = particle_group.position_dat;
  auto dat_cells = particle_group.cell_id_dat;
  auto dat_ranks = particle_group.mpi_rank_dat;

  auto k_overlay_mapper = this->overlay_mesh->get_device_mapper();
  auto k_map_sizes = this->map_sizes->root;
  auto k_map_candidates = this->map_candidates->root;
  auto k_cell_data = this->cell_data->root;

  auto map_loop = particle_loop(
      "DMPlex3DMapper::map", dat_positions,
      [=](auto P, auto CELL, auto RANK) {
        // Find the cell in the overlayed mesh
        const REAL x0 = P.at(0);
        const REAL x1 = P.at(1);
        const REAL x2 = P.at(2);
        int cell_tuple[3] = {k_overlay_mapper.get_cell_in_dimension(0, x0),
                             k_overlay_mapper.get_cell_in_dimension(1, x1),
                             k_overlay_mapper.get_cell_in_dimension(2, x2)};
        const int overlay_cell =
            k_overlay_mapper.get_linear_cell_index(cell_tuple);
        // Get the number of candidate cells
        int num_candidates = 0;
        k_map_sizes->get(overlay_cell, &num_candidates);
        int *candidates = nullptr;
        k_map_candidates->get(overlay_cell, &candidates);
        // loop over candidates and test if point in cell
        for (int cx = 0; cx < num_candidates; cx++) {

          const int candidate = candidates[cx];
          // Get the cell data for this candidate cell
          Implementation3DLinear::Linear3DData const *cell_data = nullptr;
          k_cell_data->get(candidate, &cell_data);

          // Test if point in candidate cell
          int num_crossings = 0;
          const int num_faces = cell_data->num_faces;
          const REAL *normal_origin = cell_data->normal_origin;

          bool contained = true;
          for (int facex = 0; facex < num_faces; facex++) {
            const REAL normal[3] = {normal_origin[facex * 6 + 0],
                                    normal_origin[facex * 6 + 1],
                                    normal_origin[facex * 6 + 2]};
            const REAL origin[3] = {normal_origin[facex * 6 + 3 + 0],
                                    normal_origin[facex * 6 + 3 + 1],
                                    normal_origin[facex * 6 + 3 + 2]};
            const REAL t0[3] = {x0 - origin[0], x1 - origin[1], x2 - origin[2]};

            const PetscScalar t0_dot_n = KERNEL_DOT_PRODUCT_3D(
                t0[0], t0[1], t0[2], normal[0], normal[1], normal[2]);

            if (t0_dot_n > 0.0) {
              contained = false;
            }
          }

          if (contained) {
            CELL.at(0) = cell_data->local_id;
            RANK.at(1) = cell_data->owning_rank;
          }
        }
      },
      Access::read(dat_positions), Access::write(dat_cells),
      Access::write(dat_ranks));

  if (map_cell > -1) {
    map_loop->execute(map_cell);
  } else {
    map_loop->execute();
  }

  if (map_cell > -1) {
    auto k_ep = this->ep->device_ptr();
    particle_loop(
        "DMPlex3DMapper::check", dat_positions,
        [=](auto RANK) { NESO_KERNEL_ASSERT(RANK.at(1) > -1, k_ep); },
        Access::read(dat_ranks))
        ->execute(map_cell);
    if (this->ep->get_flag()) {
      auto ranks = dat_ranks->cell_dat.get_cell(map_cell);
      auto positions = dat_positions->cell_dat.get_cell(map_cell);
      for (int rx = 0; rx < ranks->nrow; rx++) {
        if (ranks->at(rx, 1) < 0) {
          nprint("-----------Failing particle info------------");
          particle_group.print_particle(map_cell, rx);
          nprint("--------------------------------------------");
        }
      }
    }
    this->ep->check_and_throw("DMPlex3DMapper Failed to find local cell for "
                              "one or more particles.");
  }
}

} // namespace NESO::Particles::PetscInterface
#endif
