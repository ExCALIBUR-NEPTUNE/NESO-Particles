#ifndef _NESO_PARTICLES_DOMAIN
#define _NESO_PARTICLES_DOMAIN

#include "typedefs.hpp"

namespace NESO::Particles {

class Mesh {
private:
  int cell_count;

public:
  Mesh(int cell_count) : cell_count(cell_count){};
  inline int get_cell_count() { return this->cell_count; };
};

class Domain {
private:
public:
  Mesh &mesh;
  Domain(Mesh &mesh) : mesh(mesh) {}
  ~Domain() {}
};

} // namespace NESO::Particles

#endif
