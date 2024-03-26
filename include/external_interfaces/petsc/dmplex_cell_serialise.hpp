#ifndef _NESO_PARTICLES_DMPLEX_CELL_SERIALISE_HPP_
#define _NESO_PARTICLES_DMPLEX_CELL_SERIALISE_HPP_

#include "../../mesh_hierarchy_data/serial_interface.hpp"
#include "petsc_common.hpp"
#include <vector>

namespace NESO::Particles::PetscInterface {

/**
 * TODO
 */
class DMPlexCellSerialise : public MeshHierarchyData::SerialInterface {
public:

  PetscInt cell_local_id;
  PetscInt cell_global_id;
  PetscInt owning_rank;
  DMPolytopeType cell_type;
  PetscInt coord_ndim;
  std::vector<PetscInt> global_vertex_ids;
  std::vector<PetscScalar> vertex_coords;

  virtual inline std::size_t get_num_bytes() const override {

  }

  virtual inline void serialise(std::byte *buffer,
                                const std::size_t num_bytes) const override {

  }

  virtual inline void deserialise(const std::byte *buffer,
                                  const std::size_t num_bytes) override {

  }


};

}

#endif
