#ifndef _NESO_PARTICLES_DMPLEX_2D_MAPPER_H_
#define _NESO_PARTICLES_DMPLEX_2D_MAPPER_H_

#include "../../../compute_target.hpp"
#include "../../../containers/lookup_table.hpp"
#include "../../../error_propagate.hpp"
#include "../../../local_mapping.hpp"
#include "../../../loop/particle_loop.hpp"
#include "../../../particle_group_impl.hpp"
#include "../../common/overlay_cartesian_mesh.hpp"
#include "../dmplex_interface.hpp"
#include <list>
#include <memory>
#include <stack>

namespace NESO::Particles::PetscInterface {

namespace Implementation2DLinear {

struct Linear2DData {
  int owning_rank;
  int local_id;
  int num_vertices;
  REAL vertices[8];
  int faces[8];
};

} // namespace Implementation2DLinear

/**
 * Class to implement binning particles into cells in linear 2D DMPlex meshes.
 */
class DMPlex2DMapper {
protected:
  std::unique_ptr<LookupTable<int, Implementation2DLinear::Linear2DData>>
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

  /**
   * Create mapper for a compute target and 2D DMPlex.
   *
   * @param sycl_target Compute target to create mapper on.
   * @param dmplex_interface DMPlexInterface containing 2D DMPlex to create
   * mapper for.
   */
  DMPlex2DMapper(SYCLTargetSharedPtr sycl_target,
                 DMPlexInterfaceSharedPtr dmplex_interface);

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
