#include <neso_particles/utility.hpp>

namespace NESO::Particles {

std::vector<std::vector<double>>
uniform_within_extents(const int N, const int ndim, const double *extents) {
  std::mt19937 rng = std::mt19937(std::random_device{}());
  return uniform_within_extents(N, ndim, extents, rng);
}

std::vector<std::vector<double>> normal_distribution(const int N,
                                                     const int ndim,
                                                     const double mu,
                                                     const double sigma) {
  std::mt19937 rng = std::mt19937(std::random_device{}());
  return normal_distribution(N, ndim, mu, sigma, rng);
}

void uniform_within_cartesian_cells(CartesianHMeshSharedPtr mesh,
                                    const int npart_per_cell,
                                    std::vector<std::vector<double>> &positions,
                                    std::vector<int> &cells,
                                    std::optional<std::mt19937> rng_in) {

  std::mt19937 rng;
  if (!rng_in) {
    rng = std::mt19937(std::random_device{}());
  } else {
    rng = rng_in.value();
  }
  const int ndim = mesh->get_ndim();
  std::vector<double> extents(ndim);

  const double cell_width_fine = mesh->get_cell_width_fine();
  for (int dx = 0; dx < ndim; dx++) {
    extents.at(dx) = cell_width_fine;
  }
  const int cell_count = mesh->get_cart_cell_count();
  const int npart_total = npart_per_cell * cell_count;
  positions.resize(ndim);
  cells.resize(npart_total);
  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(npart_total);
  }
  std::vector<double> origin(ndim);
  for (int dx = 0; dx < ndim; dx++) {
    origin.at(dx) = mesh->get_mesh_hierarchy()->origin.at(dx);
  }

  auto mesh_cells = mesh->get_owned_cells();
  const bool single_cell_mode = mesh->single_cell_mode;

  for (int cx = 0; cx < cell_count; cx++) {
    const int index_start = cx * npart_per_cell;
    const int index_end = (cx + 1) * npart_per_cell;
    for (int ex = index_start; ex < index_end; ex++) {
      cells.at(ex) = single_cell_mode ? 0 : cx;
    }
    auto positions_ref_cell =
        uniform_within_extents(npart_per_cell, ndim, extents.data(), rng);

    std::vector<double> offset(ndim);
    for (int dx = 0; dx < ndim; dx++) {
      offset.at(dx) =
          origin.at(dx) + mesh_cells.at(cx).at(dx) * cell_width_fine;
    }

    int index = 0;
    for (int ex = index_start; ex < index_end; ex++) {
      for (int dx = 0; dx < ndim; dx++) {
        positions.at(dx).at(ex) =
            offset.at(dx) + positions_ref_cell.at(dx).at(index);
      }
      index++;
    }
  }
}

} // namespace NESO::Particles
