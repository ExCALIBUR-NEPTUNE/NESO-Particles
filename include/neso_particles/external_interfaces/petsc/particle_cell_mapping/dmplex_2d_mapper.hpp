#ifndef _NESO_PARTICLES_DMPLEX_2D_MAPPER_H_
#define _NESO_PARTICLES_DMPLEX_2D_MAPPER_H_

#include "../../../compute_target.hpp"
#include "../../../containers/lookup_table.hpp"
#include "../../../local_mapping.hpp"
#include "../../../loop/particle_loop.hpp"
#include "../../../particle_group.hpp"
#include "../../common/overlay_cartesian_mesh.hpp"
#include "../dmplex_interface.hpp"
#include <list>
#include <memory>

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

public:
  SYCLTargetSharedPtr sycl_target;
  DMPlexInterfaceSharedPtr dmplex_interface;

  /**
   * TODO
   */
  DMPlex2DMapper(SYCLTargetSharedPtr sycl_target,
                 DMPlexInterfaceSharedPtr dmplex_interface)
      : sycl_target(sycl_target), dmplex_interface(dmplex_interface) {

    constexpr int ndim = 2;
    const int num_local_cells = dmplex_interface->dmh->get_num_cells();
    const int num_halo_cells = dmplex_interface->dmh_halo->get_num_cells();
    const int num_total_cells = num_local_cells + num_halo_cells;

    // Create a bounding box for halo cells and local cells.
    auto bounding_box = dmplex_interface->dmh->get_bounding_box();
    bounding_box->expand(dmplex_interface->dmh_halo->get_bounding_box());

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
        [&](DM &dm, PetscInt c, const int owning_rank,
            const int local_id) -> Implementation2DLinear::Linear2DData {
      Implementation2DLinear::Linear2DData tmp_data;
      PetscBool is_dg;
      PetscInt num_coords;
      const PetscScalar *array;
      PetscScalar *coords = nullptr;
      PETSCCHK(DMPlexGetCellCoordinates(dm, c, &is_dg, &num_coords, &array,
                                        &coords));
      for (int cx = 0; cx < num_coords * ndim; cx++) {
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
      return tmp_data;
    };

    // Local cells
    int index = 0;
    int cell_start = dmplex_interface->dmh->cell_start;
    int cell_end = dmplex_interface->dmh->cell_end;
    std::vector<int> overlay_cells;
    for (int cx = cell_start; cx < cell_end; cx++) {
      auto bb = dmplex_interface->dmh->get_cell_bounding_box(cx);
      this->overlay_mesh->get_intersecting_cells(bb, overlay_cells);
      for (auto &ox : overlay_cells) {
        map_overlay_cells[ox].push_back(index);
      }
      // Record the description of this cell
      auto tmp_data =
          lambda_populate_cell_data(dmplex_interface->dmh->dm, cx,
                                    sycl_target->comm_pair.rank_parent, index);
      this->cell_data->add(index, tmp_data);
      index++;
    }
    // Halo cells
    cell_start = dmplex_interface->dmh_halo->cell_start;
    cell_end = dmplex_interface->dmh_halo->cell_end;
    for (int cx = cell_start; cx < cell_end; cx++) {
      auto bb = dmplex_interface->dmh_halo->get_cell_bounding_box(cx);
      this->overlay_mesh->get_intersecting_cells(bb, overlay_cells);
      for (auto &ox : overlay_cells) {
        map_overlay_cells[ox].push_back(index);
      }
      // Record the description of this cell
      auto id_rank = dmplex_interface->map_local_lid_remote_lid.at(cx);
      auto tmp_data =
          lambda_populate_cell_data(dmplex_interface->dmh_halo->dm, cx,
                                    std::get<0>(id_rank), std::get<1>(id_rank));
      this->cell_data->add(index, tmp_data);
      index++;
    }

    // Create the map from overlay cartesian cells to DMPlex cells
  }

  /**
   * TODO
   */
  inline void map(ParticleGroup &particle_group, const int map_cell) {}
};

} // namespace NESO::Particles::PetscInterface

#endif
