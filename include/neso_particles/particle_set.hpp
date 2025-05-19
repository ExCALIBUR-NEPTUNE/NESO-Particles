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
class H5Part;
class ParticleSetDevice;

/**
 *  Container to hold particle data for a set of particles.
 */
class ParticleSet {
  friend class ParticleGroup;
  friend class H5Part;
  friend class ParticleSetDevice;

protected:
  std::map<Sym<REAL>, std::vector<REAL>> values_real;
  std::map<Sym<INT>, std::vector<INT>> values_int;
  std::map<Sym<REAL>, int> ncomp_real;
  std::map<Sym<INT>, int> ncomp_int;

  std::vector<REAL> dummy_real;
  std::vector<INT> dummy_int;

  REAL *get_ptr(Sym<REAL> sym, const int particle_index,
                const int component_index);
  INT *get_ptr(Sym<INT> sym, const int particle_index,
               const int component_index);

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
  ParticleSet(const int npart, ParticleSpec particle_spec);

  /**
   *  Access the std::vectors that correspond to a Sym<REAL>.
   */
  ColumnMajorRowAccessor<std::vector, REAL> operator[](Sym<REAL> sym);

  /**
   *  Access the std::vectors that correspond to a Sym<INT>.
   */
  ColumnMajorRowAccessor<std::vector, INT> operator[](Sym<INT> sym);

  /**
   * Access REAL elements for a particle.
   *
   * @param sym Sym of particle property to access.
   * @param particle_index Index of particle to access.
   * @param component_index Index of component to access.
   * @returns modifiable reference to element.
   */
  REAL &at(Sym<REAL> sym, const int particle_index, const int component_index);

  /**
   * Access REAL elements for a particle.
   *
   * @param sym Sym of particle property to access.
   * @param particle_index Index of particle to access.
   * @param component_index Index of component to access.
   * @returns modifiable reference to element.
   */
  INT &at(Sym<INT> sym, const int particle_index, const int component_index);

  /**
   *  Get the vector of values describing the particle data for a given
   *  Sym<REAL>. Will return an empty std::vector if the passed Sym is not a
   *  stored property.
   *
   *  @param sym Sym<REAL> to access.
   *  @returns std::vector of data or empty std::vector.
   */
  std::vector<REAL> &get(Sym<REAL> const &sym);

  /**
   *  Get the vector of values describing the particle data for a given
   *  Sym<INT>. Will return an empty std::vector if the passed Sym is not a
   *  stored property.
   *
   *  @param sym Sym<INT> to access.
   *  @returns std::vector of data or empty std::vector.
   */
  std::vector<INT> &get(Sym<INT> const &sym);

  /**
   *  Test to see if this ParticleSet contains data for a given Sym<REAL>
   *
   *  @param sym Sym<REAL> to test for.
   *  @returns Bool indicating if data exists.
   */
  bool contains(Sym<REAL> const &sym);

  /**
   *  Test to see if this ParticleSet contains data for a given Sym<INT>
   *
   *  @param sym Sym<INT> to test for.
   *  @returns Bool indicating if data exists.
   */
  bool contains(Sym<INT> const &sym);

  /**
   * Set all values of a Sym from a std::vector.
   *
   * @param sym Sym to set values for.
   * @param component Component to set values for.
   * @param values Vector of values to set.
   */
  void set(Sym<INT> sym, const int component, std::vector<INT> &values);

  /**
   * Set all values of a Sym from a std::vector.
   *
   * @param sym Sym to set values for.
   * @param component Component to set values for.
   * @param values Vector of values to set.
   */
  void set(Sym<REAL> sym, const int component, std::vector<REAL> &values);

  /**
   * Set all values in this ParticleSet from a another ParticleSet. This will
   * copy the values from the intersection of the properties defined in this
   * ParticleSet and the properties defined in the provided ParticleSet.
   * Component counts must agree between the ParticleSets. The number of
   * particles in the provided ParticleSet must be the same as this ParticleSet.
   *
   * @param particle_set ParticleSet to copy values from.
   */
  void set(ParticleSet &particle_set);

  /**
   * Set all values in this ParticleSet from a another ParticleSet. This will
   * copy the values from the intersection of the properties defined in this
   * ParticleSet and the properties defined in the provided ParticleSet.
   * Component counts must agree between the ParticleSets. The number of
   * particles in the provided ParticleSet must be the same as this ParticleSet.
   *
   * @param particle_set ParticleSet to copy values from.
   */
  void set(std::shared_ptr<ParticleSet> particle_set);
};

typedef std::shared_ptr<ParticleSet> ParticleSetSharedPtr;

} // namespace NESO::Particles

#endif
