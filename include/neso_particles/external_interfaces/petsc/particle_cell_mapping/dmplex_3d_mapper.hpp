#ifndef _NESO_PARTICLES_DMPLEX_3D_MAPPER_H_
#define _NESO_PARTICLES_DMPLEX_3D_MAPPER_H_

#include "../../../compute_target.hpp"
#include "../../../containers/lookup_table.hpp"
#include "../../../error_propagate.hpp"
#include "../../../local_mapping.hpp"
#include "../../../loop/particle_loop_functions.hpp"
#include "../../../particle_group_impl.hpp"
#include "../../common/overlay_cartesian_mesh.hpp"
#include "../dmplex_interface.hpp"
#include <list>
#include <memory>
#include <stack>

namespace NESO::Particles::PetscInterface {

namespace Implementation3DLinear {

struct Linear3DData {
  int owning_rank;
  int local_id;
  int num_faces;
  REAL normal_origin[(3 + 3) * 6];
};

} // namespace Implementation3DLinear

/**
 * Class to implement binning particles into cells in linear 3D DMPlex meshes.
 */
class DMPlex3DMapper {
protected:
  std::unique_ptr<LookupTable<int, Implementation3DLinear::Linear3DData>>
      cell_data;
  std::shared_ptr<ExternalCommon::OverlayCartesianMesh> overlay_mesh;

  // size of candidate maps for each overlay cell
  std::unique_ptr<LookupTable<int, int>> map_sizes;
  // candidate maps for each overlay cell
  std::unique_ptr<LookupTable<int, int *>> map_candidates;
  // stack for device map of candidate cells
  std::stack<std::unique_ptr<BufferDevice<int>>> map_stack;
  std::unique_ptr<ErrorPropagate> ep;

public:
  SYCLTargetSharedPtr sycl_target;
  DMPlexInterfaceSharedPtr dmplex_interface;
  REAL tol;

  /**
   * Create mapper for a compute target and 3D DMPlex.
   *
   * @param sycl_target Compute target to create mapper on.
   * @param dmplex_interface DMPlexInterface containing 3D DMPlex to create
   * mapper for.
   * @param tol Tolerance for mapping.
   */
  DMPlex3DMapper(SYCLTargetSharedPtr sycl_target,
                 DMPlexInterfaceSharedPtr dmplex_interface,
                 const REAL tol = 0.0);

  /**
   * Map particles into cells.
   *
   * @param particle_group Particles to map into cells.
   * @param map_cell Cell to explicitly map. Values less than zero imply map
   * all cells.
   */
  void map(ParticleGroup &particle_group, const int map_cell);
};

} // namespace NESO::Particles::PetscInterface

#endif
