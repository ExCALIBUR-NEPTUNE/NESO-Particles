#ifndef NESO_PARTICLES_LOCAL_CLAIM_HPP_
#define NESO_PARTICLES_LOCAL_CLAIM_HPP_

#include "../../mesh_hierarchy.hpp"
#include "../../typedefs.hpp"
#include <deque>
#include <map>
#include <set>
#include <vector>

#include "bounding_box.hpp"

namespace NESO::Particles::ExternalCommon {

/**
 *  Simple wrapper around an int and float to use for assembling cell claim
 *  weights.
 */
class ClaimWeight {
private:
public:
  /// The integer weight that will be used to make the claim.
  int weight;
  /// A floating point weight for reference/testing.
  double weightf;
  ~ClaimWeight(){};
  ClaimWeight() : weight(0), weightf(0.0){};
};

/**
 *  Container to collect claim weights local to this rank before passing them
 *  to the mesh hierarchy. Local collection prevents excessive MPI RMA comms.
 */
class LocalClaim {
private:
public:
  /// Map from global cell indices of a MeshHierarchy to ClaimWeights.
  std::map<std::int64_t, ClaimWeight> claim_weights;
  /// Set of cells which claims were made for.
  std::set<std::int64_t> claim_cells;
  ~LocalClaim(){};
  LocalClaim(){};
  /**
   *  Claim a cell with passed weights.
   *
   *  @param index Global linear index of cell in MeshHierarchy.
   *  @param weight Integer claim weight, this will be passed to the
   * MeshHierarchy to claim the cell.
   *  @param weightf Floating point weight for reference/testing.
   */
  void claim(const int64_t index, const int weight, const double weightf);
};

/**
 *  Use the bounds of an element in 1D to compute the overlap area with a given
 *  cell. Passed bounds should be shifted relative to an origin of 0.
 *
 *  @param lhs Lower bound of element.
 *  @param rhs Upepr bound of element.
 *  @param cell Cell index (base 0).
 *  @param cell_width_fine Width of each cell.
 */
double overlap_1d(const double lhs, const double rhs, const int cell,
                  const double cell_width_fine);

/**
 * Convert a mesh index (index_x, index_y, ...) for this cartesian mesh to
 * the format for a MeshHierarchy: (coarse_x, coarse_y,.., fine_x,
 * fine_y,...).
 *
 * @param ndim Number of dimensions.
 * @param index_mesh Tuple index into cartesian grid of cells.
 * @param mesh_hierarchy MeshHierarchy instance.
 * @param index_mh Output index in the MeshHierarchy.
 */
void mesh_tuple_to_mh_tuple(const int ndim, const int64_t *index_mesh,
                            std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                            INT *index_mh);

/**
 * Compute all claims to cells, and associated weights, for the passed element
 * using the element bounding box.
 *
 * @param[in] bounding_box Bounding box to use for intersection.
 * @param[in] mesh_hierarchy MeshHierarchy instance which cell claims will be
 * made into.
 * @param[in,out] cells Mesh heirarchy cells covered by bounding box.
 */
void bounding_box_map(BoundingBoxSharedPtr element_bounding_box,
                      std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                      std::deque<std::pair<INT, double>> &cells);

/**
 * Helper type mapping MeshHierarchy cells to mesh elements that intersect the
 * key.
 */
typedef std::map<INT, std::deque<int>> MHGeomMap;

/**
 * Compute all claims to cells, and associated weights, for the passed element
 * using the element bounding box.
 *
 * @param element_id Integer element id for map from MeshHierarchy cells to
 * element ids.
 * @param bounding_box Bounding box for element.
 * @param mesh_hierarchy MeshHierarchy instance which cell claims will be made
 * into.
 * @param local_claim LocalClaim instance in which cell claims are being
 * collected into.
 * @param mh_geom_map MHGeomMap from MeshHierarchy global cells ids to element
 * ids.
 */
void bounding_box_claim(int element_id, BoundingBoxSharedPtr bounding_box,
                        std::shared_ptr<MeshHierarchy> mesh_hierarchy,
                        LocalClaim &local_claim, MHGeomMap &mh_geom_map);

} // namespace NESO::Particles::ExternalCommon

#endif
