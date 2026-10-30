#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/external_interfaces/petsc/dmplex_mesh_coupler_dg0_numbering_map.hpp>

namespace NESO::Particles::PetscInterface {

DMPlexMeshCouplerDG0NumberingMap::~DMPlexMeshCouplerDG0NumberingMap() {
  if (!this->is_freed) {
    nprint("Warning: DMPlexMeshCouplerDG0NumberingMap::free not called.");
  }
}

DMPlexMeshCouplerDG0NumberingMap::DMPlexMeshCouplerDG0NumberingMap(
    SYCLTargetSharedPtr sycl_target)
    : sycl_target(sycl_target) {}

void DMPlexMeshCouplerDG0NumberingMap::initalise_pre_distribute(DM dm) {

  NESOASSERT(!this->init_post, "initalise_post_distribute has been called "
                               "before initalise_pre_distribute.");
  this->init_pre = true;
  const int rank = this->sycl_target->comm_pair.rank_parent;
  MPI_Comm comm = this->sycl_target->comm_pair.comm_parent;

  PETSCCHK(DMPlexGetChart(dm, &this->point_start_pre, &this->point_end_pre));

  if (rank) {
    NESOASSERT(this->point_start_pre == this->point_end_pre,
               "Expected zero points on ranks != 0.");
  }
  MPICHK(MPI_Bcast(&this->point_start_pre, 1, MPIU_INT, 0, comm));
  {
    PetscInt cell_start = -1;
    PetscInt cell_end = -1;
    PETSCCHK(DMPlexGetHeightStratum(dm, 0, &cell_start, &cell_end));
    MPICHK(MPI_Bcast(&cell_start, 1, MPIU_INT, 0, comm));
    MPICHK(MPI_Bcast(&cell_end, 1, MPIU_INT, 0, comm));
    this->cell_start_pre = cell_start;
    this->cell_end_pre = cell_end;
  }
}

void DMPlexMeshCouplerDG0NumberingMap::initalise_post_distribute(
    DM dm_distributed, PetscSF sf) {

  NESOASSERT(this->init_pre, "initalise_pre_distribute has not been called "
                             "before initalise_post_distribute.");
  this->init_post = true;
  std::map<PetscInt, MeshHierarchyData::GenericSerialContainer<PetscInt>> m;
  const int size = this->sycl_target->comm_pair.size_parent;

  if (size == 1) {
    for (PetscInt px = this->point_start_pre; px < this->point_end_pre; px++) {
      m[px] = MeshHierarchyData::GenericSerialContainer<PetscInt>(px);
    }
  } else {

    PetscInt point_start = 0;
    PetscInt point_end = 0;
    PETSCCHK(DMPlexGetChart(dm_distributed, &point_start, &point_end));

    PetscInt nroots = 0;
    PetscInt nleaves = 0;
    PetscInt const *ilocal = nullptr;
    PetscSFNode const *iremote = nullptr;
    PETSCCHK(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote));

    IS global_point_numbers;
    PETSCCHK(DMPlexCreatePointNumbering(dm_distributed, &global_point_numbers));
    const PetscInt *ptr;
    PETSCCHK(ISGetIndices(global_point_numbers, &ptr));

    for (int ix = 0; ix < nleaves; ix++) {

      const PetscInt local_point_previous = iremote[ix].index;
      NESOASSERT(iremote[ix].rank == 0,
                 "Expected all points to originate on rank 0.");
      const PetscInt global_point_current = ptr[ix - point_start];

      if (global_point_current > -1) {
        m[local_point_previous] =
            MeshHierarchyData::GenericSerialContainer<PetscInt>(
                global_point_current);
      }
    }

    PETSCCHK(ISRestoreIndices(global_point_numbers, &ptr));
    PETSCCHK(ISDestroy(&global_point_numbers));
  }

  this->lut = std::make_shared<GlobalKeyValueLUTNodewise<
      PetscInt, MeshHierarchyData::GenericSerialContainer<PetscInt>>>(
      sycl_target, m);

  {
    std::set<PetscInt> to_test;
    for (auto &[k, v] : m) {
      to_test.insert(k);
    }
    auto r = this->lut->get(to_test);
    for (auto &[k, v] : m) {
      NESOASSERT(r.at(k).obj == v.obj, "Self test failed.");
    }
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

  std::set<PetscInt> to_query;
  for (PetscInt &px : input_points) {
    to_query.insert(px);
  }
  auto m = this->lut->get(to_query);
  to_query.clear();
  for (PetscInt &px : input_points) {
    global_points.push_back(m.at(px).obj);
  }
}

std::map<PetscInt, PetscInt>
DMPlexMeshCouplerDG0NumberingMap::get_global_cell_indices(
    std::set<PetscInt> input_cells) {

  std::map<PetscInt, PetscInt> m;
  std::set<PetscInt> to_query;
  for (auto input_cell : input_cells) {
    const PetscInt input_global_point = input_cell + this->cell_start_pre;
    NESOASSERT((this->cell_start_pre <= input_global_point) &&
                   (input_global_point < this->cell_end_pre),
               "Bad input cell.");
    to_query.insert(input_global_point);
  }

  auto mg = this->lut->get(to_query);

  for (auto input_cell : input_cells) {
    const PetscInt input_global_point = input_cell + this->cell_start_pre;
    m[input_cell] = mg.at(input_global_point).obj;
  }

  return m;
}

void DMPlexMeshCouplerDG0NumberingMap::free() {
  this->lut->free();
  this->is_freed = true;
}

} // namespace NESO::Particles::PetscInterface

#endif
