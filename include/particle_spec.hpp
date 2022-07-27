#ifndef _NESO_PARTICLES_PARTICLE_SPEC
#define _NESO_PARTICLES_PARTICLE_SPEC

#include <cstdint>
#include <map>
#include <mpi.h>
#include <string>
#include <vector>

#include "typedefs.hpp"

namespace NESO::Particles {

template <typename U> class Sym {
private:
public:
  const std::string name;
  Sym(const std::string name) : name(name) {}

  // std::map uses std::less as default comparison operator
  bool operator<(const Sym &sym) const { return this->name < sym.name; }
};

template <typename T> class ParticleProp {
private:
public:
  const Sym<T> sym;
  const std::string name;
  const int ncomp;
  const bool positions;
  ParticleProp(const Sym<T> sym, int ncomp, bool positions = false)
      : sym(sym), name(sym.name), ncomp(ncomp), positions(positions) {}
};

class ParticleSpec {
private:
public:
  template <typename... T> void push(T... args) { this->push(args...); }
  void push(ParticleProp<REAL> pp) { this->properties_real.push_back(pp); }
  void push(ParticleProp<INT> pp) { this->properties_int.push_back(pp); }
  template <typename... T> void push(ParticleProp<REAL> pp, T... args) {
    this->properties_real.push_back(pp);
    this->push(args...);
  }
  template <typename... T> void push(ParticleProp<INT> pp, T... args) {
    this->properties_int.push_back(pp);
    this->push(args...);
  }
  std::vector<ParticleProp<REAL>> properties_real;
  std::vector<ParticleProp<INT>> properties_int;

  template <typename... T> ParticleSpec(T... args) { this->push(args...); };

  ParticleSpec(){};
  ~ParticleSpec(){};
};

class PrintSpec {
private:
public:
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
  std::vector<Sym<REAL>> syms_real;
  std::vector<Sym<INT>> syms_int;

  template <typename... T> PrintSpec(T... args) { this->push(args...); };

  PrintSpec(){};
  ~PrintSpec(){};
};

} // namespace NESO::Particles

#endif
