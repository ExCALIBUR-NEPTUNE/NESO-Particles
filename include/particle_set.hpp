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

class ParticleGroup;

/**
 *  Container to hold particle data for a set of particles.
 */
class ParticleSet {
  friend class ParticleGroup;

private:
  std::map<Sym<REAL>, std::vector<REAL>> values_real;
  std::map<Sym<INT>, std::vector<INT>> values_int;

  std::vector<REAL> dummy_real;
  std::vector<INT> dummy_int;

protected:
  inline REAL *get_ptr(Sym<REAL> sym, const int particle_index,
                       const int component_index) {
    return values_real.at(sym).data() + component_index * this->npart +
           particle_index;
  }
  inline INT *get_ptr(Sym<INT> sym, const int particle_index,
                      const int component_index) {
    return values_int.at(sym).data() + component_index * this->npart +
           particle_index;
  }

public:
  /// Number of particles stored in the container.
  const int npart;

  /**
   *  Constructor for a set of particles.
   *
   *  @param npart Number of particles required.
   *  @param particle_spec ParticleSpec instance that describes the particle
   *  properties.
   */
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

  /**
   *  Access the std::vectors that correspond to a Sym<REAL>.
   */
  inline ColumnMajorRowAccessor<std::vector, REAL> operator[](Sym<REAL> sym) {
    return ColumnMajorRowAccessor<std::vector, REAL>{values_real[sym],
                                                     this->npart};
  };

  /**
   *  Access the std::vectors that correspond to a Sym<INT>.
   */
  inline ColumnMajorRowAccessor<std::vector, INT> operator[](Sym<INT> sym) {
    return ColumnMajorRowAccessor<std::vector, INT>{values_int[sym],
                                                    this->npart};
  };

  /**
   * Access REAL elements for a particle.
   *
   * @param sym Sym of particle property to access.
   * @param particle_index Index of particle to access.
   * @param component_index Index of component to access.
   * @returns modifiable reference to element.
   */
  inline REAL &at(Sym<REAL> sym, const int particle_index,
                  const int component_index) {
    NESOASSERT(this->contains(sym), "Property does not exist in ParticleSet");
    return this->values_real.at(sym).at(component_index * this->npart +
                                        particle_index);
  }

  /**
   * Access REAL elements for a particle.
   *
   * @param sym Sym of particle property to access.
   * @param particle_index Index of particle to access.
   * @param component_index Index of component to access.
   * @returns modifiable reference to element.
   */
  inline INT &at(Sym<INT> sym, const int particle_index,
                 const int component_index) {
    NESOASSERT(this->contains(sym), "Property does not exist in ParticleSet");
    return this->values_int.at(sym).at(component_index * this->npart +
                                       particle_index);
  }

  /**
   *  Get the vector of values describing the particle data for a given
   *  Sym<REAL>. Will return an empty std::vector if the passed Sym is not a
   *  stored property.
   *
   *  @param sym Sym<REAL> to access.
   *  @returns std::vector of data or empty std::vector.
   */
  inline std::vector<REAL> &get(Sym<REAL> const &sym) {
    if (contains(sym)) {
      return values_real[sym];
    } else {
      return dummy_real;
    }
  };
  /**
   *  Get the vector of values describing the particle data for a given
   *  Sym<INT>. Will return an empty std::vector if the passed Sym is not a
   *  stored property.
   *
   *  @param sym Sym<INT> to access.
   *  @returns std::vector of data or empty std::vector.
   */
  inline std::vector<INT> &get(Sym<INT> const &sym) {
    if (contains(sym)) {
      return values_int[sym];
    } else {
      return dummy_int;
    }
  };
  /**
   *  Test to see if this ParticleSet contains data for a given Sym<REAL>
   *
   *  @param sym Sym<REAL> to test for.
   *  @returns Bool indicating if data exists.
   */
  inline bool contains(Sym<REAL> const &sym) {
    return (this->values_real.count(sym) > 0);
  }
  /**
   *  Test to see if this ParticleSet contains data for a given Sym<INT>
   *
   *  @param sym Sym<INT> to test for.
   *  @returns Bool indicating if data exists.
   */
  inline bool contains(Sym<INT> const &sym) {
    return (this->values_int.count(sym) > 0);
  }
};

typedef std::shared_ptr<ParticleSet> ParticleSetSharedPtr;

} // namespace NESO::Particles

#endif
