#ifndef _NESO_PARTICLES_DMPLEX_MESH_COUPLER_DG0_NUMBERING_MAP_HPP_
#define _NESO_PARTICLES_DMPLEX_MESH_COUPLER_DG0_NUMBERING_MAP_HPP_

#include "dmplex_helper.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Helper class for mapping between point and cell indices for
 * DMPlexMeshCouplerDG0.
 *
 * Use along the lines of:
 * ```
 * DM dm;
 * // Create DM, e.g. box mesh/gmsh.
 *
 * DMPlexMeshCouplerDG0NumberingMap nm(comm);
 * nm.initalise_pre_distribute(dm);
 *
 * PetscSF sf;
 * generic_distribute(&dm, comm, 0, &sf);
 *
 * nm.initalise_post_distribute(dm, sf);
 *
 * // Collect cells to map into a std::set<PetscInt> to_query
 *
 * auto global_cell_indices = nm.get_global_cell_indices(to_query);
 * ```
 *
 * @ingroup external_interfaces_petsc_dmplex_mesh_coupling
 */
class DMPlexMeshCouplerDG0NumberingMap {
protected:
  MPI_Comm comm;
  std::vector<PetscInt> h_init_map_local_to_global;

  bool init_pre{false};
  bool init_post{false};

  PetscInt cell_start_pre{-1};
  PetscInt cell_end_pre{-1};
  PetscInt global_cell_start_post{-1};

public:
  /**
   * Initialise instance. Must be called collectively on the communicator.
   *
   * @param comm MPI Communicator for decomposed mesh.
   */
  DMPlexMeshCouplerDG0NumberingMap(MPI_Comm comm);

  /**
   * Indicate the DMPlex that exists before decomposition. We assume that this
   * mesh exists only on rank zero and that the DMPlex on all other ranks has
   * zero points. The input DM will not be stored.
   *
   * @param dm Initial DMPlex before decomposition.
   */
  void initalise_pre_distribute(DM dm);

  /**
   * Indicate the DMPlex that exists after decomposition. Along with the
   * distribution map. The input DM will not be stored. Must be called
   * collectively on the communicator.
   *
   * @param dm_distributed DMPlex after decomposition.
   * @param sf Star forest returned from generic_distribute or DMPlexDistribute.
   */
  void initalise_post_distribute(DM dm_distributed, PetscSF sf);

  /**
   * Get the distributed global indices, for use with DMPlexMeshCouplerDG0, that
   * correspond to points on the original non-distributed DMPlex. Must be called
   * collectively on the communicator.
   *
   * @param input_points Input point indices, not cell indices, on the original
   * mesh.
   * @param global_points Global point indices on the distributed mesh.
   */
  void get_global_point_indices(std::vector<PetscInt> &input_points,
                                std::vector<PetscInt> &global_points);

  /**
   * Get the distributed global cell index, for use with DMPlexMeshCouplerDG0,
   * that correspond to cells on the original non-distributed DMPlex. The input
   * values should be local cell indices, not PETSc point indices, in [0,
   * num_cells_init). The output is global cell indices, not PETSc point
   * indices, in [0, num_cells_init).
   *
   * Must be called collectively on the communicator.
   *
   * @param input_cells Input cell indices, not point indices, on the original
   * mesh.
   * @returns Global point indices on the distributed mesh. As a map from input
   * indices to global indices.
   */
  std::map<PetscInt, PetscInt>
  get_global_cell_indices(std::set<PetscInt> input_cells);
};

} // namespace NESO::Particles::PetscInterface

#endif
