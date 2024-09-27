#ifndef _NESO_PARTICLES_DMPLEX_2D_MAPPER_H_
#define _NESO_PARTICLES_DMPLEX_2D_MAPPER_H_

#include "../../../compute_target.hpp"
#include "../../../containers/lookup_table.hpp"
#include "../../../error_propagate.hpp"
#include "../../../local_mapping.hpp"
#include "../../../loop/particle_loop.hpp"
#include "../../../particle_group_impl.hpp"
#include "../../common/overlay_cartesian_mesh.hpp"
#include "../dmplex_interface.hpp"
#include <list>
#include <memory>
#include <stack>

namespace NESO::Particles::PetscInterface {

namespace Implementation2DLinear {

struct Linear2DData {
  int owning_rank;
  int local_id;
  int num_vertices;
  REAL vertices[8];
  int faces[8];
};

} // namespace Implementation2DLinear

/**
 * Class to implement binning particles into cells in linear 2D DMPlex meshes.
 */
class DMPlex2DMapper {
protected:
  std::unique_ptr<LookupTable<int, Implementation2DLinear::Linear2DData>>
      cell_data;
  std::shared_ptr<ExternalCommon::OverlayCartesianMesh> overlay_mesh;

  // size of candidate maps for each overlay cell
  std::unique_ptr<LookupTable<int, int>> map_sizes;
  // candidate maps for each overlay cell
  std::unique_ptr<LookupTable<int, int *>> map_candidates;
  // stack for device map of candidate cells
  std::stack<std::unique_ptr<BufferDevice<int>>> map_stack;
  std::unique_ptr<ErrorPropagate> ep;

public:
  SYCLTargetSharedPtr sycl_target;
  DMPlexInterfaceSharedPtr dmplex_interface;

  /**
   * Create mapper for a compute target and 2D DMPlex.
   *
   * @param sycl_target Compute target to create mapper on.
   * @param dmplex_interface DMPlexInterface containing 2D DMPlex to create
   * mapper for.
   */
  DMPlex2DMapper(SYCLTargetSharedPtr sycl_target,
                 DMPlexInterfaceSharedPtr dmplex_interface)
      : sycl_target(sycl_target), dmplex_interface(dmplex_interface) {

    constexpr int ndim = 2;
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
    this->cell_data = std::make_unique<
        LookupTable<int, Implementation2DLinear::Linear2DData>>(
        this->sycl_target, num_total_cells);

    // For each local and halo cell find the overlay cells they intersect with
    std::map<int, std::list<int>> map_overlay_cells;

    // Helper lambda to populate the cell data
    auto lambda_populate_cell_data =
        [&](DM &dm, PetscInt petsc_index, const int owning_rank,
            const int local_id) -> Implementation2DLinear::Linear2DData {
      Implementation2DLinear::Linear2DData tmp_data;
      PetscBool is_dg;
      PetscInt num_coords;
      const PetscScalar *array;
      PetscScalar *coords = nullptr;

      PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                        &array, &coords));
      for (int cx = 0; cx < num_coords; cx++) {
        tmp_data.vertices[cx] = coords[cx];
      }
      NESOASSERT((num_coords == 6) || (num_coords == 8),
                 "Unexpected number of coordinates.");
      if (num_coords == 8) {
        tmp_data.num_vertices = 4;
        // {0, 1, 1, 2, 2, 3, 3, 0};
        tmp_data.faces[0] = 0;
        tmp_data.faces[1] = 1;
        tmp_data.faces[2] = 1;
        tmp_data.faces[3] = 2;
        tmp_data.faces[4] = 2;
        tmp_data.faces[5] = 3;
        tmp_data.faces[6] = 3;
        tmp_data.faces[7] = 0;
      } else {
        tmp_data.num_vertices = 3;
        // {0, 1, 1, 2, 2, 0};
        tmp_data.faces[0] = 0;
        tmp_data.faces[1] = 1;
        tmp_data.faces[2] = 1;
        tmp_data.faces[3] = 2;
        tmp_data.faces[4] = 2;
        tmp_data.faces[5] = 0;
      }
      tmp_data.owning_rank = owning_rank;
      tmp_data.local_id = local_id;
      PETSCCHK(DMPlexRestoreCellCoordinates(dm, petsc_index, &is_dg,
                                            &num_coords, &array, &coords));
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
        auto id_rank =
            dmplex_interface->map_local_lid_remote_lid.at(point_index);
        const PetscInt petsc_index =
            dmplex_interface->dmh_halo->get_dmplex_cell_index(cx);

        auto tmp_data = lambda_populate_cell_data(dmh_halo->dm, petsc_index,
                                                  std::get<0>(id_rank),
                                                  std::get<1>(id_rank));
        this->cell_data->add(index, tmp_data);
        index++;
      }
    }

    // Create the map from overlay cartesian cells to DMPlex cells
    const int overlay_cell_count = this->overlay_mesh->get_cell_count();
    this->map_sizes = std::make_unique<LookupTable<int, int>>(
        this->sycl_target, overlay_cell_count);

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

  /**
   * Map particles into cells.
   *
   * @param particle_group Particles to map into cells.
   * @param map_cell Cell to explicitly map. Values less than zero imply map
   * all cells.
   */
  inline void map(ParticleGroup &particle_group, const int map_cell) {

    auto dat_positions = particle_group.position_dat;
    auto dat_cells = particle_group.cell_id_dat;
    auto dat_ranks = particle_group.mpi_rank_dat;

    auto k_overlay_mapper = this->overlay_mesh->get_device_mapper();
    auto k_map_sizes = this->map_sizes->root;
    auto k_map_candidates = this->map_candidates->root;
    auto k_cell_data = this->cell_data->root;

    auto map_loop = particle_loop(
        "DMPlex2DMapper::map", dat_positions,
        [=](auto P, auto CELL, auto RANK) {
          // Find the cell in the overlayed mesh
          const REAL x0 = P.at(0);
          const REAL x1 = P.at(1);
          int cell_tuple[2] = {k_overlay_mapper.get_cell_in_dimension(0, x0),
                               k_overlay_mapper.get_cell_in_dimension(1, x1)};
          const int overlay_cell =
              k_overlay_mapper.get_linear_cell_index(cell_tuple);
          // Get the number of candidate cells
          int num_candidates;
          k_map_sizes->get(overlay_cell, &num_candidates);
          int *candidates;
          k_map_candidates->get(overlay_cell, &candidates);
          // loop over candidates and test if point in cell
          for (int cx = 0; cx < num_candidates; cx++) {
            const int candidate = candidates[cx];
            // Get the cell data for this candidate cell
            Implementation2DLinear::Linear2DData const *cell_data;
            k_cell_data->get(candidate, &cell_data);
            // Test if point in candidate cell
            int num_crossings = 0;
            const int num_vertices = cell_data->num_vertices;
            const int *faces = cell_data->faces;
            const REAL *vertices = cell_data->vertices;

            for (int facex = 0; facex < num_vertices; facex++) {
              REAL xi = vertices[faces[2 * facex + 0] * 2 + 0];
              REAL yi = vertices[faces[2 * facex + 0] * 2 + 1];
              REAL xj = vertices[faces[2 * facex + 1] * 2 + 0];
              REAL yj = vertices[faces[2 * facex + 1] * 2 + 1];
              // Is the point in a corner
              if ((x0 == xj) && (x1 == yj)) {
                num_crossings = 1;
                break;
              }
              if ((yj > x1) != (yi > x1)) {
                REAL determinate =
                    (x0 - xj) * (yi - yj) - (xi - xj) * (x1 - yj);
                if (determinate == 0) {
                  // Point is on line
                  num_crossings = 1;
                  break;
                }
                if ((determinate < 0) != (yi < yj)) {
                  num_crossings++;
                }
              }
            }
            // If the number of crossings is odd then the point is in the cell.
            if (num_crossings % 2) {
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
          "DMPlex2DMapper::check", dat_positions,
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
      this->ep->check_and_throw("DMPlex2DMapper Failed to find local cell for "
                                "one or more particles.");
    }
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
