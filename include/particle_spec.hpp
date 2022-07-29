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
  const std::string name;

  /**
   * Construct a new Sym object.
   *
   * @param name Name of new symbol.
   */
  Sym(const std::string name) : name(name) {}

  // std::map uses std::less as default comparison operator
  bool operator<(const Sym &sym) const { return this->name < sym.name; }
};

/**
 *  Class to describe a property of a particle.
 */
template <typename T> class ParticleProp {
private:
public:
  /// Symbol giving a label to the property.
  const Sym<T> sym;
  /// Name of the property.
  const std::string name;
  /// Number of components required by this property.
  const int ncomp;
  /// Bool to indicate if this property is a particle position or cell id.
  const bool positions;
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
  template <typename... T> ParticleSpec(T... args) { this->push(args...); };
  /**
   *  Push a ParticleProp<REAL> property onto the specification.
   *
   *  @param pp ParticleProp<REAL> to add to the specification.
   */
  void push(ParticleProp<REAL> pp) { this->properties_real.push_back(pp); }
  /**
   *  Push a ParticleProp<INT> property onto the specification.
   *
   *  @param pp ParticleProp<INT> to add to the specification.
   */
  void push(ParticleProp<INT> pp) { this->properties_int.push_back(pp); }

  ParticleSpec(){};
  ~ParticleSpec(){};
};

/**
 *  Helper class to hold a collection of Sym instances for ParticleGroup::print.
 */
class PrintSpec {
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
   *  Constructor for PrintSpec should be called with a list of arguments which
   *  are Sym instances.
   *
   *  @param args Passed arguments should be Sym<REAL> or Sym<INT>.
   */
  template <typename... T> PrintSpec(T... args) { this->push(args...); };

  PrintSpec(){};
  ~PrintSpec(){};
};

} // namespace NESO::Particles

#endif
