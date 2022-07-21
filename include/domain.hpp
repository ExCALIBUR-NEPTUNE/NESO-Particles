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

class Domain {
private:
public:
  HMesh &mesh;
  LocalMapperShPtr local_mapper;
  Domain(HMesh &mesh, LocalMapperShPtr local_mapper = DummyLocalMapper())
      : mesh(mesh), local_mapper(local_mapper) {}
  ~Domain() {}
};

} // namespace NESO::Particles

#endif
