#ifndef _NESO_PARTICLES_DMPLEX_UTILITY_HPP_
#define _NESO_PARTICLES_DMPLEX_UTILITY_HPP_

#include "../common/quadrature_point_mapper.hpp"
#include "dmplex_cell_serialise.hpp"
#include "dmplex_interface.hpp"
#include "petsc_common.hpp"
#include "project_evaluate/dmplex_project_evaluate_barycentric.hpp"
#include "project_evaluate/dmplex_project_evaluate_dg.hpp"
#include <petscviewerhdf5.h>
#include <string>

namespace NESO::Particles::PetscInterface {

/**
 * Helper function to quickly initialise a uniform distribution of particles on
 * a DMPlex mesh cell.
 *
 * @param[in] mesh DMPlexInterface on which to spawn particles.
 * @param[in] cell Cell to sample particle positions in.
 * @param[in] npart Number of particle positions to sample.
 * @param[in] offset Offset into the positions and cells in which to start
 * placing samples.
 * @param[in, out] positions Ouput particle positions indexed by dimension then
 * particle. First index is dimension second index is particle. The vectors must
 * be of length at least offset + npart.
 * @param[in, out] cells Ouput particle cell ids indexed by particle. The vector
 * must be of length at least offset + npart.
 * @param[in] rng_in Optional input RNG to use.
 * @param[in] attempt_max Optional maximum number of attempts to place particles
 * in each cell (default 1E8).
 */
inline int uniform_within_dmplex_cell(
    DMPlexInterfaceSharedPtr mesh, const int cell, const int npart,
    const int offset, std::vector<std::vector<double>> &positions,
    std::vector<int> &cells, std::mt19937 *rng_in = nullptr,
    const int attempt_max = 1e8) {

  std::mt19937 rng;
  if (rng_in == nullptr) {
    rng = std::mt19937(std::random_device{}());
    rng_in = &rng;
  }

  const int ndim = mesh->get_ndim();

  for (int dx = 0; dx < ndim; dx++) {
    NESOASSERT(positions.at(dx).size() >= offset + npart,
               "Positions vector is too small.");
  }
  NESOASSERT(cells.size() >= offset + npart, "Cells vector is too small.");

  auto bounding_box = mesh->dmh->get_cell_bounding_box(cell);
  std::vector<std::uniform_real_distribution<double>> dists;
  dists.reserve(ndim);
  for (int dx = 0; dx < ndim; dx++) {
    dists.push_back(std::uniform_real_distribution<double>{
        bounding_box->lower(dx), bounding_box->upper(dx)});
  }

  int index = offset;
  std::vector<PetscScalar> proposed_position(ndim);
  for (int px = 0; px < npart; px++) {
    bool contained = false;
    int attempt_count = 0;
    while ((!contained) && (attempt_count < attempt_max)) {
      for (int dx = 0; dx < ndim; dx++) {
        proposed_position.at(dx) = dists.at(dx)(*rng_in);
      }
      contained = mesh->dmh->cell_contains_point(cell, proposed_position);
      attempt_count++;
    }
    NESOASSERT(attempt_count < attempt_max, "Maximum attempt count reached.");
    for (int dx = 0; dx < ndim; dx++) {
      positions.at(dx).at(index) = proposed_position.at(dx);
    }
    cells.at(index) = cell;
    index++;
  }

  return index;
}

/**
 * Helper function to quickly initialise a uniform distribution of particles on
 * a DMPlex mesh. Samples a fixed number of particles per cell (particle number
 * density will only be uniform if the cell volumes are uniform).
 *
 * @param[in] mesh DMPlexInterface on which to spawn particles.
 * @param[in] npart_per_cell Number of particle positions to sample for each
 * mesh cell.
 * @param[in, out] positions Ouput particle positions indexed by dimension then
 * particle.
 * @param[in, out] cells Ouput particle cell ids indexed by particle.
 * @param[in] rng_in Optional input RNG to use.
 * @param[in] attempt_max Optional maximum number of attempts to place particles
 * in each cell (default 1E8).
 */
inline void uniform_within_dmplex_cells(
    DMPlexInterfaceSharedPtr mesh, const int npart_per_cell,
    std::vector<std::vector<double>> &positions, std::vector<int> &cells,
    std::mt19937 *rng_in = nullptr, const int attempt_max = 1e8) {

  std::mt19937 rng;
  if (rng_in == nullptr) {
    rng = std::mt19937(std::random_device{}());
    rng_in = &rng;
  }
  const int ndim = mesh->get_ndim();
  const int cell_count = mesh->get_cell_count();
  const int npart_total = npart_per_cell * cell_count;

  // resize output space
  positions.resize(ndim);
  for (int dx = 0; dx < ndim; dx++) {
    positions[dx] = std::vector<double>(npart_total);
  }
  cells.resize(npart_total);

  // for each cell make particle positions
  int index = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    index = uniform_within_dmplex_cell(mesh, cx, npart_per_cell, index,
                                       positions, cells, rng_in, attempt_max);
  }
  NESOASSERT(index == npart_total, "Error creating particle positions.");
}

/**
 * Helper function to quickly initialise a uniform distribution of particles on
 * a DMPlex mesh. Note that this function will probably not exactly match the
 * specified number density as particles are discrete and hence the user should
 * check the number added, see return value. The number of particles for a cell
 * is computed as round(number_density * cell_volume) and hence may be 0 or
 * slightly more than expected.
 *
 * @param[in] mesh DMPlexInterface on which to spawn particles.
 * @param[in] number_density Target number density for each cell.
 * @param[in, out] positions Ouput particle positions indexed by dimension then
 * particle.
 * @param[in, out] cells Ouput particle cell ids indexed by particle.
 * @param[in] rng_in Optional input RNG to use.
 * @param[in] attempt_max Optional maximum number of attempts to place particles
 * in each cell (default 1E8).
 * @returns Number of particles added on this rank. Equal to the sizes of the
 * positions and cells vectors on return.
 */
inline int uniform_density_within_dmplex_cells(
    DMPlexInterfaceSharedPtr mesh, const REAL number_density,
    std::vector<std::vector<double>> &positions, std::vector<int> &cells,
    std::mt19937 *rng_in = nullptr, const int attempt_max = 1e8) {

  std::mt19937 rng;
  if (rng_in == nullptr) {
    rng = std::mt19937(std::random_device{}());
    rng_in = &rng;
  }
  const int ndim = mesh->get_ndim();
  const int cell_count = mesh->get_cell_count();

  std::vector<int> npart_per_cell(cell_count);
  int npart_total = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    const int tmp = std::round(number_density * mesh->dmh->get_cell_volume(cx));
    npart_per_cell.at(cx) = tmp;
    npart_total += tmp;
  }

  // resize output space
  positions.resize(ndim);
  for (int dx = 0; dx < ndim; dx++) {
    positions[dx] = std::vector<double>(npart_total);
  }
  cells.resize(npart_total);

  // for each cell make particle positions
  int index = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    index = uniform_within_dmplex_cell(mesh, cx, npart_per_cell.at(cx), index,
                                       positions, cells, rng_in, attempt_max);
  }
  NESOASSERT(index == npart_total, "Error creating particle positions.");
  return npart_total;
}

/**
 * Create a QuadraturePointMapper with a point in the vertex of each cell. In
 * the DG dense with a full set of point for each cell.
 *
 * @param sycl_target Compute device on which to create the mapper.
 * @param domain Domain for mapper.
 */
inline std::shared_ptr<ExternalCommon::QuadraturePointMapper>
make_quadrature_point_mapper_vertex(SYCLTargetSharedPtr sycl_target,
                                    DomainSharedPtr domain) {

  auto mesh =
      std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(domain->mesh);
  auto cell_vertices_info = get_cell_vertices_cdc(sycl_target, mesh->dmh);
  auto cdc_num_vertices = std::get<0>(cell_vertices_info);
  auto cdc_vertices = std::get<1>(cell_vertices_info);

  const int cell_count = domain->mesh->get_cell_count();
  auto qpm = std::make_shared<ExternalCommon::QuadraturePointMapper>(
      sycl_target, domain);
  std::vector<std::vector<double>> positions;
  std::vector<int> cells;

  const int k_ndim = mesh->get_ndim();
  // Add a point per vertex in the middle of each cell
  qpm->add_points_initialise();
  std::vector<REAL> average(k_ndim);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int num_vertices = cdc_num_vertices->get_value(cellx, 0, 0);
    mesh->dmh->get_cell_vertex_average(cellx, average);
    for (int vx = 0; vx < num_vertices; vx++) {
      qpm->add_point(average.data());
    }
  }
  qpm->add_points_finalise();

  // move the points to the vertices
  particle_loop(
      qpm->particle_group,
      [=](auto INDEX, auto VERTICES, auto POS) {
        for (int dx = 0; dx < k_ndim; dx++) {
          POS.at(dx) = VERTICES.at(INDEX.layer, dx);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(cdc_vertices),
      Access::write(qpm->particle_group->position_dat))
      ->execute();

  return qpm;
}

/**
 * Create a QuadraturePointMapper with a point at the average of each cell.
 *
 * @param sycl_target Compute device on which to create the mapper.
 * @param domain Domain for mapper.
 */
inline std::shared_ptr<ExternalCommon::QuadraturePointMapper>
make_quadrature_point_mapper_average(SYCLTargetSharedPtr sycl_target,
                                     DomainSharedPtr domain) {

  auto mesh =
      std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(domain->mesh);
  const int cell_count = domain->mesh->get_cell_count();
  auto qpm = std::make_shared<ExternalCommon::QuadraturePointMapper>(
      sycl_target, domain);

  const int k_ndim = mesh->get_ndim();
  qpm->add_points_initialise();
  std::vector<REAL> average(k_ndim);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    mesh->dmh->get_cell_vertex_average(cellx, average);
    qpm->add_point(average.data());
  }
  qpm->add_points_finalise();

  return qpm;
}

/**
 *
 */
template <std::size_t NDIM, typename T>
inline std::array<int, 3> sample_points_for_distribution(
    SYCLTargetSharedPtr sycl_target, DomainSharedPtr domain,
    const REAL particle_weight, T func,
    std::vector<std::vector<double>> &positions, std::vector<int> &cells,
    std::mt19937 *rng_in = nullptr, const int attempt_max = 1e8) {

  auto mesh =
      std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(domain->mesh);
  const int cell_count = domain->mesh->get_cell_count();
  auto qpm_vertices = make_quadrature_point_mapper_vertex(sycl_target, domain);
  // Evaluate the function at the corners of the cells
  ExternalCommon::interpolate<NDIM, 1>(qpm_vertices, func);

  auto cdc_inverse_num_vertices =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 1, 1);
  auto cdc_average =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 1, 1);
  cdc_average->fill(0.0);

  std::vector<std::vector<REAL>> vertices;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    mesh->dmh->get_cell_vertices(cellx, vertices);
    cdc_inverse_num_vertices->set_value(cellx, 0, 0, 1.0 / vertices.size());
  }

  particle_loop(
      qpm_vertices->particle_group,
      [=](auto Q, auto NUM_VERTICES, auto AVERAGE) {
        AVERAGE.fetch_add(0, 0, Q.at(0) * NUM_VERTICES.at(0, 0));
      },
      Access::read(qpm_vertices->get_sym(1)),
      Access::read(cdc_inverse_num_vertices), Access::add(cdc_average))
      ->execute();
  qpm_vertices->free();

  std::vector<int> npart_per_cell(cell_count);
  int npart_total = 0;
  int npart_min = std::numeric_limits<int>::max();
  int npart_max = 0;

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const REAL volume = mesh->dmh->get_cell_volume(cellx);
    auto weight = cdc_average->get_value(cellx, 0, 0);
    const REAL mass = std::max(weight * volume, 0.0);

    const int num_particles = std::round(mass / particle_weight);
    npart_per_cell.at(cellx) = num_particles;
    npart_total += num_particles;
    npart_min = std::min(npart_min, num_particles);
    npart_max = std::max(npart_max, num_particles);
  }

  const int ndim = mesh->get_ndim();

  // resize output space
  positions.resize(ndim);
  for (int dx = 0; dx < ndim; dx++) {
    positions[dx] = std::vector<double>(npart_total);
  }
  cells.resize(npart_total);

  // for each cell make particle positions
  int index = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    index = uniform_within_dmplex_cell(mesh, cx, npart_per_cell.at(cx), index,
                                       positions, cells, rng_in, attempt_max);
  }

  return {npart_min, npart_max, npart_total};
}

} // namespace NESO::Particles::PetscInterface

#endif
