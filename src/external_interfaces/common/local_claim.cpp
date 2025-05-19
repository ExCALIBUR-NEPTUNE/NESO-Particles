#include <neso_particles/external_interfaces/common/local_claim.hpp>

namespace NESO::Particles::ExternalCommon {

void LocalClaim::claim(const int64_t index, const int weight,
                       const double weightf) {
  if (weight > 0.0) {
    this->claim_cells.insert(index);
    auto current_claim = this->claim_weights[index];
    if (weight > current_claim.weight) {
      this->claim_weights[index].weight = weight;
      this->claim_weights[index].weightf = weightf;
    }
  }
}

double overlap_1d(const double lhs, const double rhs, const int cell,
                  const double cell_width_fine) {

  const double cell_start = cell * cell_width_fine;
  const double cell_end = cell_start + cell_width_fine;

  // if the overlap is empty then the area is 0.
  if (rhs <= cell_start) {
    return 0.0;
  } else if (lhs >= cell_end) {
    return 0.0;
  }

  const double interval_start = std::max(cell_start, lhs);
  const double interval_end = std::min(cell_end, rhs);
  const double area = interval_end - interval_start;

  return (area > 0.0) ? area : 0.0;
}

void mesh_tuple_to_mh_tuple(const int ndim, const int64_t *index_mesh,
                            std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                            INT *index_mh) {
  for (int dimx = 0; dimx < ndim; dimx++) {
    auto pq = std::div((long long)index_mesh[dimx],
                       (long long)mesh_hierarchy->ncells_dim_fine);
    index_mh[dimx] = pq.quot;
    index_mh[dimx + ndim] = pq.rem;
  }
}

void bounding_box_map(BoundingBoxSharedPtr element_bounding_box,
                      std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                      std::deque<std::pair<INT, double>> &cells) {
  cells.clear();

  const int ndim = mesh_hierarchy->ndim;
  auto origin = mesh_hierarchy->origin;

  int cell_starts[3] = {0, 0, 0};
  int cell_ends[3] = {1, 1, 1};
  double shifted_bounding_box[6];

  // For each dimension compute the starting and ending cells overlapped by
  // this element by using the bounding box. This gives an iteration set of
  // cells touched by this element's bounding box.
  for (int dimx = 0; dimx < ndim; dimx++) {
    const double lhs_point = element_bounding_box->lower(dimx) - origin[dimx];
    const double rhs_point = element_bounding_box->upper(dimx) - origin[dimx];
    shifted_bounding_box[dimx] = lhs_point;
    shifted_bounding_box[dimx + 3] = rhs_point;
    int lhs_cell = lhs_point * mesh_hierarchy->inverse_cell_width_fine;
    int rhs_cell = rhs_point * mesh_hierarchy->inverse_cell_width_fine + 1;

    const int64_t ncells_dim_fine =
        mesh_hierarchy->ncells_dim_fine * mesh_hierarchy->dims[dimx];

    lhs_cell = (lhs_cell < 0) ? 0 : lhs_cell;
    lhs_cell = (lhs_cell >= ncells_dim_fine) ? ncells_dim_fine : lhs_cell;
    rhs_cell = (rhs_cell < 0) ? 0 : rhs_cell;
    rhs_cell = (rhs_cell > ncells_dim_fine) ? ncells_dim_fine : rhs_cell;

    cell_starts[dimx] = lhs_cell;
    cell_ends[dimx] = rhs_cell;
  }

  const double cell_width_fine = mesh_hierarchy->cell_width_fine;

  // mesh tuple index
  int64_t index_mesh[3];
  // mesh_hierarchy tuple index
  INT index_mh[6];

  // For each cell compute the overlap with the element and use the overlap
  // volume to compute a claim weight (as a ratio of the volume of the cell).
  for (int cz = cell_starts[2]; cz < cell_ends[2]; cz++) {
    index_mesh[2] = cz;
    double area_z = 1.0;
    if (ndim > 2) {
      area_z = overlap_1d(shifted_bounding_box[2], shifted_bounding_box[2 + 3],
                          cz, cell_width_fine);
    }

    for (int cy = cell_starts[1]; cy < cell_ends[1]; cy++) {
      index_mesh[1] = cy;
      double area_y = 1.0;
      if (ndim > 1) {
        area_y = overlap_1d(shifted_bounding_box[1],
                            shifted_bounding_box[1 + 3], cy, cell_width_fine);
      }

      for (int cx = cell_starts[0]; cx < cell_ends[0]; cx++) {
        index_mesh[0] = cx;
        const double area_x =
            overlap_1d(shifted_bounding_box[0], shifted_bounding_box[0 + 3], cx,
                       cell_width_fine);

        const double volume = area_x * area_y * area_z;

        if (volume > 0.0) {
          mesh_tuple_to_mh_tuple(ndim, index_mesh, mesh_hierarchy, index_mh);
          const INT index_global =
              mesh_hierarchy->tuple_to_linear_global(index_mh);
          cells.push_back({index_global, volume});
        }
      }
    }
  }
}

void bounding_box_claim(int element_id, BoundingBoxSharedPtr bounding_box,
                        std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                        LocalClaim &local_claim, MHGeomMap &mh_geom_map) {

  std::deque<std::pair<INT, double>> cells;
  bounding_box_map(bounding_box, mesh_hierarchy, cells);

  const int ndim = mesh_hierarchy->ndim;
  const double cell_width_fine = mesh_hierarchy->cell_width_fine;
  const double inverse_cell_volume = 1.0 / std::pow(cell_width_fine, ndim);

  for (const auto &cell_volume : cells) {
    const INT index_global = cell_volume.first;
    const double volume = cell_volume.second;
    const double ratio = volume * inverse_cell_volume;
    const int weight = 1000000.0 * ratio;

    local_claim.claim(index_global, weight, ratio);
    mh_geom_map[index_global].push_back(element_id);
  }
}

} // namespace NESO::Particles::ExternalCommon
