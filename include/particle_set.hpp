#ifndef _NESO_PARTICLES_PARTICLE_SET
#define _NESO_PARTICLES_PARTICLE_SET

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "access.hpp"
#include "compute_target.hpp"
#include "particle_spec.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

class ParticleSet {

private:
  std::map<Sym<REAL>, std::vector<REAL>> values_real;
  std::map<Sym<INT>, std::vector<INT>> values_int;

public:
  const int npart;

  ParticleSet(const int npart, ParticleSpec particle_spec) : npart(npart) {

    for (auto const &spec : particle_spec.properties_real) {
      values_real[spec.sym] = std::vector<REAL>(npart * spec.ncomp);
      std::fill(values_real[spec.sym].begin(), values_real[spec.sym].end(),
                0.0);
    }
    for (auto const &spec : particle_spec.properties_int) {
      values_int[spec.sym] = std::vector<INT>(npart * spec.ncomp);
      std::fill(values_int[spec.sym].begin(), values_int[spec.sym].end(), 0);
    }
  };

  inline ColumnMajorRowAccessor<std::vector, REAL> operator[](Sym<REAL> sym) {
    return ColumnMajorRowAccessor<std::vector, REAL>{values_real[sym],
                                                     this->npart};
  };
  inline ColumnMajorRowAccessor<std::vector, INT> operator[](Sym<INT> sym) {
    return ColumnMajorRowAccessor<std::vector, INT>{values_int[sym],
                                                    this->npart};
  };

  inline std::vector<REAL> &get(Sym<REAL> const &sym) {
    return values_real[sym];
  };
  inline std::vector<INT> &get(Sym<INT> const &sym) { return values_int[sym]; };
  inline bool contains(Sym<REAL> const &sym) {
    return (this->values_real.count(sym) > 0);
  }
  inline bool contains(Sym<INT> const &sym) {
    return (this->values_int.count(sym) > 0);
  }
};

} // namespace NESO::Particles

#endif
