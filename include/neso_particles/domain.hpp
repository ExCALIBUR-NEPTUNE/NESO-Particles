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
 *  A domain wraps a mesh with a method to map particles into cells on that
 * mesh. This allows there to be multiple different methods for binning
 * particles into cells for each mesh type. Typically shared pointers to Domains
 * are used to construct ParticleGroups.
 *
 *  @ingroup neso_particles_core_domain
 */
class Domain {
private:
public:
  Domain(const Domain &st) = delete;
  Domain &operator=(Domain const &a) = delete;

  /// HMesh derived mesh instance.
  HMeshSharedPtr mesh;
  /// LocalMapper derived class instance to bin particles into mesh cells.
  LocalMapperSharedPtr local_mapper;
  /**
   * Construct a new Domain.
   *
   * @param mesh HMesh derived mesh object.
   * @param local_mapper Object to map particle positions into mesh cells.
   */
  Domain(HMeshSharedPtr mesh,
         LocalMapperSharedPtr local_mapper = DummyLocalMapper())
      : mesh(mesh), local_mapper(local_mapper) {}
  ~Domain() {}
};

/// Downstream interfaces will expect this type for Domains.
typedef std::shared_ptr<Domain> DomainSharedPtr;

} // namespace NESO::Particles

#endif
