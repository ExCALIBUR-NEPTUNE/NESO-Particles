#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/external_interfaces/petsc/dmplex_mesh_coupler_dg0_numbering_map.hpp>

namespace NESO::Particles::PetscInterface {

DMPlexMeshCouplerDG0NumberingMap::DMPlexMeshCouplerDG0NumberingMap(
    MPI_Comm comm)
    : comm(comm) {}

void DMPlexMeshCouplerDG0NumberingMap::initalise_pre_distribute(DM dm) {

  NESOASSERT(!this->init_post, "initalise_post_distribute has been called "
                               "before initalise_pre_distribute.");
  this->init_pre = true;

  int rank;
  MPICHK(MPI_Comm_rank(this->comm, &rank));

  PetscInt point_start = 0;
  PetscInt point_end = 0;
  PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));

  if (rank) {
    NESOASSERT(point_start == point_end, "Expected zero points on ranks != 0.");
  }

  {
    PetscInt cell_start = -1;
    PetscInt cell_end = -1;
    PETSCCHK(DMPlexGetHeightStratum(dm, 0, &cell_start, &cell_end));
    MPICHK(MPI_Bcast(&cell_start, 1, MPIU_INT, 0, this->comm));
    this->cell_start_pre = cell_start;
    this->cell_end_pre = cell_end;
  }

  PetscInt num_points_local = point_end - point_start;
  MPICHK(MPI_Bcast(&num_points_local, 1, MPIU_INT, 0, this->comm));
  MPICHK(MPI_Bcast(&point_start, 1, MPIU_INT, 0, this->comm));

  this->h_init_map_local_to_global.resize(num_points_local);

  auto map_local_to_global = get_global_indices_map(dm);

  if (!rank) {
    for (auto &m : map_local_to_global) {
      const PetscInt l = m.first;
      const PetscInt g = m.second;
      NESOASSERT((point_start <= l) && (l < point_end), "Bad local point.");
      NESOASSERT((point_start <= g) && (g < point_end),
                 "Bad global point. get_global_distributed_points_map "
                 "currently assumes this range. Please raise an issue.");
      this->h_init_map_local_to_global.at(l - point_start) = g;
    }
  }

  MPICHK(MPI_Bcast(this->h_init_map_local_to_global.data(), num_points_local,
                   MPIU_INT, 0, this->comm));
}

void DMPlexMeshCouplerDG0NumberingMap::initalise_post_distribute(
    DM dm_distributed, PetscSF sf) {

  NESOASSERT(this->init_pre, "initalise_pre_distribute has not been called "
                             "before initalise_post_distribute.");
  this->init_post = true;

  auto post_distribution_map =
      get_global_distributed_points_map(dm_distributed, sf);

  for (auto &m : this->h_init_map_local_to_global) {
    const PetscInt old_global_point = m;
    const PetscInt new_global_point =
        post_distribution_map.at(old_global_point);
    m = new_global_point;
  }

  {
    PetscInt global_point_min = std::numeric_limits<PetscInt>::max();
    for (PetscInt px = this->cell_start_pre; px < this->cell_end_pre; px++) {
      const PetscInt global_point = this->h_init_map_local_to_global.at(px);
      global_point_min = std::min(global_point_min, global_point);
    }
    this->global_cell_start_post = global_point_min;
    MPICHK(MPI_Bcast(&global_point_min, 1, MPIU_INT, 0, this->comm));
    NESOASSERT(this->global_cell_start_post == global_point_min,
               "Missmatch in global cell start.");
  }
}

void DMPlexMeshCouplerDG0NumberingMap::get_global_point_indices(
    std::vector<PetscInt> &input_points, std::vector<PetscInt> &global_points) {

  NESOASSERT(this->init_pre, "initalise_pre_distribute has not been called "
                             "before get_global_point_indices.");
  NESOASSERT(this->init_post, "initalise_post_distribute has not been called "
                              "before get_global_point_indices.");

  global_points.clear();
  global_points.reserve(input_points.size());

  for (PetscInt &px : input_points) {
    NESOASSERT((0 <= px) && (px < static_cast<PetscInt>(
                                      this->h_init_map_local_to_global.size())),
               "Bad input point.");
    global_points.push_back(this->h_init_map_local_to_global.at(px));
  }
}

PetscInt DMPlexMeshCouplerDG0NumberingMap::get_global_cell_index(
    const PetscInt input_cell) {

  const PetscInt input_global_point = input_cell + this->cell_start_pre;

  NESOASSERT((this->cell_start_pre <= input_global_point) &&
                 (input_global_point < this->cell_end_pre),
             "Bad input cell.");

  const PetscInt output_global_point =
      this->h_init_map_local_to_global.at(input_global_point);

  const PetscInt output_global_cell_index =
      output_global_point - this->global_cell_start_post;

  return output_global_cell_index;
}
} // namespace NESO::Particles::PetscInterface

#endif
