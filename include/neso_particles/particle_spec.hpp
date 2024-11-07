#ifndef _NESO_PARTICLES_PARTICLE_SPEC
#define _NESO_PARTICLES_PARTICLE_SPEC

#include <cstdint>
#include <map>
#include <mpi.h>
#include <string>
#include <vector>

#include "typedefs.hpp"

namespace NESO::Particles {

/**
 *  Symbol describing a type of data and a name.
 */
template <typename U> class Sym {
private:
public:
  /// Name of the symbol.
  std::string name;

  Sym() = default;
  Sym<U> &operator=(const Sym<U> &) = default;
  Sym<U>(const Sym<U> &_) = default;

  /**
   * Construct a new Sym object.
   *
   * @param name Name of new symbol.
   */
  Sym(const std::string name) : name(name) {}

  /** Comparison of Sym types.
   *
   * std::map uses std::less as default comparison operator
   *
   * @param sym Other Sym to compare against.
   */
  bool operator<(const Sym &sym) const { return this->name < sym.name; }
  bool operator==(const Sym &sym) const { return this->name == sym.name; }
};

/**
 *  Class to describe a property of a particle.
 */
template <typename T> class ParticleProp {
private:
public:
  /// Symbol giving a label to the property.
  Sym<T> sym;
  /// Name of the property.
  std::string name;
  /// Number of components required by this property.
  int ncomp;
  /// Bool to indicate if this property is a particle position or cell id.
  bool positions;
  /**
   *  Constructor for particle properties.
   *
   *  @param sym Sym<T> instance for the property.
   *  @param ncomp Number of components for the property.
   *  @param positions Bool to indicate if the property hold positions or cells
   *  ids.
   */
  ParticleProp(const Sym<T> sym, int ncomp, bool positions = false)
      : sym(sym), name(sym.name), ncomp(ncomp), positions(positions) {}

  bool operator==(const ParticleProp<T> &prop) const {
    const bool same = (this->sym == prop.sym) && (this->name == prop.name) &&
                      (this->ncomp == prop.ncomp);
    return same;
  }
};

/**
 * A ParticleSpec is a particle specification described by a collection of
 * particle properties.
 */
class ParticleSpec {
private:
  template <typename... T> void push(T... args) { this->push(args...); }
  template <typename... T> void push(ParticleProp<REAL> pp, T... args) {
    this->properties_real.push_back(pp);
    this->push(args...);
  }
  template <typename... T> void push(ParticleProp<INT> pp, T... args) {
    this->properties_int.push_back(pp);
    this->push(args...);
  }

  inline auto find(ParticleProp<REAL> &prop) {
    return std::find(this->properties_real.begin(), this->properties_real.end(),
                     prop);
  }
  inline auto find(ParticleProp<INT> &prop) {
    return std::find(this->properties_int.begin(), this->properties_int.end(),
                     prop);
  }
  inline auto end(ParticleProp<REAL> &) { return this->properties_real.end(); }
  inline auto end(ParticleProp<INT> &) { return this->properties_int.end(); }
  template <typename T> inline void erase(ParticleProp<REAL>, T &location) {
    this->properties_real.erase(location);
  }
  template <typename T> inline void erase(ParticleProp<INT>, T &location) {
    this->properties_int.erase(location);
  }

public:
  /// Collection of REAL ParticleProp
  std::vector<ParticleProp<REAL>> properties_real;
  /// Collection of INT ParticleProp
  std::vector<ParticleProp<INT>> properties_int;

  /**
   *  Constructor to create a particle specification.
   *
   *  @param args ParticleSpec is called with a set of ParticleProp arguments.
   */
  template <typename... T> ParticleSpec(T... args) { this->push(args...); }

  /**
   * Create a particle specification from a vector of REAL properties and a
   * vector of INT properties.
   *
   *  @param properties_real REAL particle properties.
   *  @param properties_int INT particle properties.
   */
  ParticleSpec(std::vector<ParticleProp<REAL>> &properties_real,
               std::vector<ParticleProp<INT>> &properties_int)
      : properties_real(properties_real), properties_int(properties_int) {}

  /**
   *  Push a ParticleProp<REAL> property onto the specification.
   *
   *  @param pp ParticleProp<REAL> to add to the specification.
   */
  inline void push(ParticleProp<REAL> pp) {
    this->properties_real.push_back(pp);
  }
  /**
   *  Push a ParticleProp<INT> property onto the specification.
   *
   *  @param pp ParticleProp<INT> to add to the specification.
   */
  inline void push(ParticleProp<INT> pp) { this->properties_int.push_back(pp); }

  /**
   * Determine if a ParticleProp is contained in this specification.
   *
   * @param pp ParticleProp to search for.
   * @returns True if passed ParticleProp is in this ParticleSpec.
   */
  template <typename T> inline bool contains(ParticleProp<T> pp) {
    return this->find(pp) != this->end(pp);
  }

  /**
   * Determine if a passed ParticleSpec is a subset of this ParticleSpec.
   *
   * @param ps ParticleSpec to compare with.
   * @returns True if the passed particle spec is a subset of this particle
   * spec.
   */
  inline bool contains(ParticleSpec &ps) {
    auto dispatch = [&](auto container) -> bool {
      for (auto pp : container) {
        if (!this->contains(pp)) {
          return false;
        }
      }
      return true;
    };
    return dispatch(ps.properties_real) && dispatch(ps.properties_int);
  }

  /**
   * Remove a particle property from the specification.
   *
   * @param pp ParticleProp to remove from the specification.
   */
  template <typename T> inline void remove(ParticleProp<T> pp) {
    auto location = this->find(pp);
    NESOASSERT(location != this->end(pp),
               "Property to remove does not exist in specification.");
    this->erase(pp, location);
  }

  ParticleSpec() {};
  ~ParticleSpec() {};
};

/**
 *  Helper class to hold a collection of Sym instances for ParticleGroup::print.
 */
class SymStore {
private:
  template <typename... T> void push(T... args) { this->push(args...); }
  void push(Sym<REAL> pp) { this->syms_real.push_back(pp); }
  void push(Sym<INT> pp) { this->syms_int.push_back(pp); }
  template <typename... T> void push(Sym<REAL> pp, T... args) {
    this->syms_real.push_back(pp);
    this->push(args...);
  }
  template <typename... T> void push(Sym<INT> pp, T... args) {
    this->syms_int.push_back(pp);
    this->push(args...);
  }

public:
  /// Container of Sym<REAL> symbols.
  std::vector<Sym<REAL>> syms_real;
  /// Container of Sym<INT> symbols.
  std::vector<Sym<INT>> syms_int;
  /**
   *  Constructor for SymStore should be called with a list of arguments which
   *  are Sym instances.
   *
   *  @param args Passed arguments should be Sym<REAL> or Sym<INT>.
   */
  template <typename... T> SymStore(T... args) { this->push(args...); }

  SymStore() {};
  ~SymStore() {};
};

} // namespace NESO::Particles

#endif
