#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/common_impl.hpp>
#include <neso_particles/external_interfaces/petsc/dmplex_utility.hpp>

namespace NESO::Particles::PetscInterface {

int uniform_within_dmplex_cell(DMPlexInterfaceSharedPtr mesh, const int cell,
                               const int npart, const int offset,
                               std::vector<std::vector<double>> &positions,
                               std::vector<int> &cells, std::mt19937 *rng_in,
                               const int attempt_max) {

  std::mt19937 rng;
  if (rng_in == nullptr) {
    rng = std::mt19937(std::random_device{}());
    rng_in = &rng;
  }

  const int ndim = mesh->get_ndim();

  for (int dx = 0; dx < ndim; dx++) {
    NESOASSERT(positions.at(dx).size() >=
                   static_cast<std::size_t>(offset + npart),
               "Positions vector is too small.");
  }
  NESOASSERT(cells.size() >= static_cast<std::size_t>(offset + npart),
             "Cells vector is too small.");

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

void uniform_within_dmplex_cells(DMPlexInterfaceSharedPtr mesh,
                                 const int npart_per_cell,
                                 std::vector<std::vector<double>> &positions,
                                 std::vector<int> &cells, std::mt19937 *rng_in,
                                 const int attempt_max) {

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

int uniform_density_within_dmplex_cells(
    DMPlexInterfaceSharedPtr mesh, const REAL number_density,
    std::vector<std::vector<double>> &positions, std::vector<int> &cells,
    std::mt19937 *rng_in, const int attempt_max) {

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

std::shared_ptr<ExternalCommon::QuadraturePointMapper>
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

std::shared_ptr<ExternalCommon::QuadraturePointMapper>
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

void ensure_dm_label_exists(DM dm, std::string label_name) {
  PetscBool has_label;
  PETSCCHK(DMHasLabel(dm, label_name.c_str(), &has_label));
  if (!has_label) {
    PETSCCHK(DMCreateLabel(dm, label_name.c_str()));
  }
}

void label_all_dmplex_boundaries(DM dm, std::string label_name,
                                 PetscInt value) {
  ensure_dm_label_exists(dm, label_name);
  DMLabel label;
  PETSCCHK(DMGetLabel(dm, label_name.c_str(), &label));
  PETSCCHK(DMPlexMarkBoundaryFaces(dm, value, label));
  PETSCCHK(DMPlexLabelComplete(dm, label));
}

PetscInt signed_global_id_to_global_id(const PetscInt c) {
  return (c > -1) ? c : ((c * (-1)) - 1);
}

void label_dmplex_edges(DM dm, std::string label_name,
                        std::vector<PetscInt> &vertex_starts,
                        std::vector<PetscInt> &vertex_ends,
                        std::vector<PetscInt> &edge_labels) {
  MPI_Comm comm;
  PETSCCHK(PetscObjectGetComm((PetscObject)(dm), &comm));

  NESOASSERT(vertex_starts.size() == vertex_ends.size(),
             "vertex_starts size != vertex_ends size");
  NESOASSERT(vertex_starts.size() == edge_labels.size(),
             "vertex_starts size != edge_labels size");
  ensure_dm_label_exists(dm, label_name);

  // Gather all the edges on all ranks as we don't know which rank owns the
  // edge.
  const int nedge_local = vertex_starts.size();
  int nedge_ex = 0;
  MPICHK(MPI_Exscan(&nedge_local, &nedge_ex, 1, MPI_INT, MPI_SUM, comm));

  int nedge_total;
  MPICHK(MPI_Allreduce(&nedge_local, &nedge_total, 1, MPI_INT, MPI_SUM, comm));

  int size, rank;
  MPICHK(MPI_Comm_size(comm, &size));
  MPICHK(MPI_Comm_rank(comm, &rank));

  std::vector<int> recvcounts_gather(size);
  std::vector<int> displs_gather(size);

  std::fill(recvcounts_gather.begin(), recvcounts_gather.end(), 1);
  std::iota(displs_gather.begin(), displs_gather.end(), 0);

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  MPICHK(MPI_Allgatherv(&nedge_local, 1, MPI_INT, recvcounts.data(),
                        recvcounts_gather.data(), displs_gather.data(), MPI_INT,
                        comm));
  MPICHK(MPI_Allgatherv(&nedge_ex, 1, MPI_INT, displs.data(),
                        recvcounts_gather.data(), displs_gather.data(), MPI_INT,
                        comm));

  std::vector<PetscInt> all_vertex_starts(nedge_total);
  std::vector<PetscInt> all_vertex_ends(nedge_total);
  std::vector<PetscInt> all_edge_labels(nedge_total);

  MPICHK(MPI_Allgatherv(vertex_starts.data(), nedge_local, MPIU_INT,
                        all_vertex_starts.data(), recvcounts.data(),
                        displs.data(), MPIU_INT, comm));
  MPICHK(MPI_Allgatherv(vertex_ends.data(), nedge_local, MPIU_INT,
                        all_vertex_ends.data(), recvcounts.data(),
                        displs.data(), MPIU_INT, comm));
  MPICHK(MPI_Allgatherv(edge_labels.data(), nedge_local, MPIU_INT,
                        all_edge_labels.data(), recvcounts.data(),
                        displs.data(), MPIU_INT, comm));

  // create a map from global vertex indices to local indices
  std::map<PetscInt, PetscInt> map_vertex_index_global_to_local;
  PetscInt vertex_start, vertex_end;
  PETSCCHK(DMPlexGetDepthStratum(dm, 0, &vertex_start, &vertex_end));
  IS global_vertex_numbers;
  PETSCCHK(DMPlexGetVertexNumbering(dm, &global_vertex_numbers));
  const PetscInt *ptr;
  PETSCCHK(ISGetIndices(global_vertex_numbers, &ptr));
  for (PetscInt point = vertex_start; point < vertex_end; point++) {
    PetscInt global_point =
        signed_global_id_to_global_id(ptr[point - vertex_start]);
    map_vertex_index_global_to_local[global_point] = point;
  }
  PETSCCHK(ISRestoreIndices(global_vertex_numbers, &ptr));
  // PETSCCHK(ISDestroy(&global_vertex_numbers));

  std::vector<int> edge_labelled(nedge_total);
  std::fill(edge_labelled.begin(), edge_labelled.end(), 0);

  std::vector<PetscInt> support_s;
  std::vector<PetscInt> support_e;
  std::vector<PetscInt> support_intersection;

  for (int ix = 0; ix < nedge_total; ix++) {
    const PetscInt vs = all_vertex_starts[ix];
    const PetscInt ve = all_vertex_ends[ix];
    const PetscInt ll = all_edge_labels[ix];

    // Does this rank have both vertices
    const bool has_local_s = map_vertex_index_global_to_local.count(vs);
    const bool has_local_e = map_vertex_index_global_to_local.count(ve);
    if (has_local_s && has_local_e) {
      PetscInt local_vs = map_vertex_index_global_to_local.at(vs);
      PetscInt local_ve = map_vertex_index_global_to_local.at(ve);

      support_s.clear();
      support_e.clear();
      support_intersection.clear();

      auto lambda_get_suppport = [&](auto point_, auto &set_) {
        PetscInt size;
        PETSCCHK(DMPlexGetSupportSize(dm, point_, &size));
        const PetscInt *support;
        PETSCCHK(DMPlexGetSupport(dm, point_, &support));
        for (PetscInt jx = 0; jx < size; jx++) {
          set_.push_back(support[jx]);
        }
      };

      lambda_get_suppport(local_vs, support_s);
      lambda_get_suppport(local_ve, support_e);
      std::set_intersection(support_s.begin(), support_s.end(),
                            support_e.begin(), support_e.end(),
                            std::back_inserter(support_intersection));

      // support_intersection should contain edges, hopefully just one, between
      // the two vertices.
      for (auto edge_local_point : support_intersection) {
        PETSCCHK(DMSetLabelValue(dm, label_name.c_str(), edge_local_point, ll));
      }
      if (support_intersection.size()) {
        edge_labelled[ix] = 1;
      }
    }
  }

  std::vector<PetscInt> edge_labelled_reduced(nedge_total);
  std::fill(edge_labelled_reduced.begin(), edge_labelled_reduced.end(), 0);
  MPICHK(MPI_Allreduce(edge_labelled.data(), edge_labelled_reduced.data(),
                       nedge_total, MPIU_INT, MPI_SUM, comm));

  for (int ix = 0; ix < nedge_total; ix++) {
    NESOASSERT(edge_labelled_reduced.at(ix) >= 1, "An edge was not labelled");
  }
}

} // namespace NESO::Particles::PetscInterface

#endif
