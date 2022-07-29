#ifndef _NESO_PARTICLES_DOMAIN
#define _NESO_PARTICLES_DOMAIN
#include "compute_target.hpp"
#include "local_mapping.hpp"
#include "mesh_hierarchy.hpp"
#include "mesh_interface.hpp"
#include "particle_dat.hpp"
#include "typedefs.hpp"
#include <cstdint>
#include <cstdlib>
#include <mpi.h>
#include <set>
#include <vector>

namespace NESO::Particles {

/**
 *  A domain wraps a mesh with useful methods such as how to bin particles into
 *  local cells on that mesh type.
 */
class Domain {
private:
public:
  /// HMesh derived mesh instance.
  HMesh &mesh;
  /// LocalMapper derived class instance to bin particles into mesh cells.
  LocalMapperShPtr local_mapper;
  /**
   * Construct a new Domain.
   *
   * @param mesh HMesh derived mesh object.
   * @param local_mapper Object to map particle positions into mesh cells.
   */
  Domain(HMesh &mesh, LocalMapperShPtr local_mapper = DummyLocalMapper())
      : mesh(mesh), local_mapper(local_mapper) {}
  ~Domain() {}
};

} // namespace NESO::Particles

#endif
