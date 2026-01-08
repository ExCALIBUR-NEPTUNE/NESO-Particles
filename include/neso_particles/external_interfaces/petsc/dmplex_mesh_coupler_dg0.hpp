#ifndef _NESO_PARTICLES_DMPLEX_MESH_COUPLER_DG0_HPP_
#define _NESO_PARTICLES_DMPLEX_MESH_COUPLER_DG0_HPP_

#include "../../compute_target.hpp"
#include "petsc_common.hpp"
#include <vector>

namespace NESO::Particles::PetscInterface {

/**
 * Type that holds a cell index and a pair of weights for combining data when
 * sent/received.
 */
struct DMPlexMeshCouplerDG0MapEntry {
  // The index of a cell.
  int cell_index{0};
  // Weight to use for the forward direction.
  REAL weight_forward{0.0};
  // Weight to use for the backward direction.
  REAL weight_backward{0.0};
};

/**
 * Class for transporting DG0/Finite Volume fields/quantites between overlapping
 * meshes.
 *
 * Consider two meshes A and B. Assume that N local cells a in A are indexed
 * locally by [0, N). Mesh B is a distributed DMPlex mesh with global indices b
 * in [0, M). For each a in [0, N) the user provides a map from a to a vector of
 * DMPlexMeshCouplerDG0MapEntry instances where the cell index of each entry is
 * a global cell index b in [0, M).
 *
 * The forward direction is defined as the direction which transfers data from A
 * to B. The backward direction is the direction which transfers data from B to
 * A.
 *
 * The forward direction computes
 *
 *  \f$x_b = \sum_a w_{a,b} y_a\f$
 *
 *  where \f$x_b\f$ is the DOF in cell b in B, \f$w_{a,b}\f$ is the forward
 * direction weight for source cell a and destination cell b and \f$y_a\f$ is
 * the source DOF. The backward direction is identical with the direction of
 * data transport reversed.
 */
class DMPlexMeshCouplerDG0 {
protected:
public:
  /// Disable (implicit) copies.
  DMPlexMeshCouplerDG0(const DMPlexMeshCouplerDG0 &st) = delete;
  /// Disable (implicit) copies.
  DMPlexMeshCouplerDG0 &operator=(DMPlexMeshCouplerDG0 const &a) = delete;

  ~DMPlexMeshCouplerDG0() = default;

  // The DMPlex representing B.
  DM dmplex_B;

  // The number of local cells in A.
  int cell_count_A{0};

  // Dist graph comm for forward transfer.
  MPI_Comm comm_forward{MPI_COMM_NULL};
  std::vector<int> sources_forward;
  std::vector<int> destinations_forward;
  std::vector<MPI_Aint> send_disps_forward;
  std::vector<MPI_Aint> recv_disps_forward;
  std::vector<MPI_Aint> send_disps_forward_real;
  std::vector<MPI_Aint> recv_disps_forward_real;
  std::vector<int> cells_forward_A;
  std::vector<int> cells_forward_B;
  std::vector<REAL> weights_forward_A;

  // Dist graph comm for backward transfer.
  MPI_Comm comm_backward{MPI_COMM_NULL};
  std::vector<int> sources_backward;
  std::vector<int> destinations_backward;

  /**
   * Create a coupler with the provided forward and backward maps. This
   * constructor is collective on the communicator.
   *
   * @param dmplex_B DMPlex representation of mesh B.
   * @param coupling_map Vector of entries which describe the non-zero adjacency
   * matrix weights for the forward and backward directions. The cell indices in
   * this map are global indices of PETSc DMPlex cells. This vector should have
   * length equal the number of locally owned cells in mesh B, i.e. N. This
   * vector can be freed after construction of the instance.
   */
  DMPlexMeshCouplerDG0(
      DM dmplex_B,
      std::vector<std::vector<DMPlexMeshCouplerDG0MapEntry>> &coupling_map);

  /**
   * Perform the forward transfer. This method is collective on the
   * communicator.
   *
   * @param dofs_A Source DOFs to send and combine.
   * @param dofs_B Destination DOFS.
   */
  void forward_transfer(std::vector<REAL> &dofs_A, std::vector<REAL> &dofs_B);

  /**
   * Perform the backward transfer. This method is collective on the
   * communicator.
   *
   * @param dofs_B Source DOFs to send and combine.
   * @param dofs_A Destination DOFS.
   */
  void backward_transfer(std::vector<REAL> &dofs_B, std::vector<REAL> &dofs_A);

  /*
   * Free the coupler. This method is collective on the communicator.
   */
  void free();
};
} // namespace NESO::Particles::PetscInterface

#endif
