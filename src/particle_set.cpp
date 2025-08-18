#include <neso_particles/particle_set.hpp>

namespace NESO::Particles {

REAL *ParticleSet::get_ptr(Sym<REAL> sym, const int particle_index,
                           const int component_index) {
  return values_real.at(sym).data() + component_index * this->npart +
         particle_index;
}
INT *ParticleSet::get_ptr(Sym<INT> sym, const int particle_index,
                          const int component_index) {
  return values_int.at(sym).data() + component_index * this->npart +
         particle_index;
}

ParticleSet::ParticleSet(const int npart, ParticleSpec particle_spec)
    : npart(npart) {

  for (auto const &spec : particle_spec.properties_real) {
    values_real[spec.sym] = std::vector<REAL>(npart * spec.ncomp);
    ncomp_real[spec.sym] = spec.ncomp;
    std::fill(values_real[spec.sym].begin(), values_real[spec.sym].end(), 0.0);
  }
  for (auto const &spec : particle_spec.properties_int) {
    values_int[spec.sym] = std::vector<INT>(npart * spec.ncomp);
    ncomp_int[spec.sym] = spec.ncomp;
    std::fill(values_int[spec.sym].begin(), values_int[spec.sym].end(), 0);
  }
}

ColumnMajorRowAccessor<std::vector, REAL>
ParticleSet::operator[](Sym<REAL> sym) {
  return ColumnMajorRowAccessor<std::vector, REAL>{values_real[sym],
                                                   this->npart};
};

ColumnMajorRowAccessor<std::vector, INT> ParticleSet::operator[](Sym<INT> sym) {
  return ColumnMajorRowAccessor<std::vector, INT>{values_int[sym], this->npart};
};

REAL &ParticleSet::at(Sym<REAL> sym, const int particle_index,
                      const int component_index) {
  NESOASSERT(this->contains(sym), "Property does not exist in ParticleSet");
  return this->values_real.at(sym).at(component_index * this->npart +
                                      particle_index);
}

INT &ParticleSet::at(Sym<INT> sym, const int particle_index,
                     const int component_index) {
  NESOASSERT(this->contains(sym), "Property does not exist in ParticleSet");
  return this->values_int.at(sym).at(component_index * this->npart +
                                     particle_index);
}

std::vector<REAL> &ParticleSet::get(Sym<REAL> const &sym) {
  if (contains(sym)) {
    return values_real[sym];
  } else {
    return dummy_real;
  }
}

std::vector<INT> &ParticleSet::get(Sym<INT> const &sym) {
  if (contains(sym)) {
    return values_int[sym];
  } else {
    return dummy_int;
  }
}
bool ParticleSet::contains(Sym<REAL> const &sym) {
  return (this->values_real.count(sym) > 0);
}

bool ParticleSet::contains(Sym<INT> const &sym) {
  return (this->values_int.count(sym) > 0);
}

void ParticleSet::set(Sym<INT> sym, const int component,
                      std::vector<INT> &values) {
  NESOASSERT(this->contains(sym),
             "ParticleSet does not contain passed sym: " + sym.name);
  NESOASSERT(values.size() == static_cast<std::size_t>(this->npart),
             "Passed vector does not have the same length as the number of "
             "particles.");
  NESOASSERT(0 <= component && component < this->ncomp_int.at(sym),
             "Bad component passed.");
  std::memcpy(this->get_ptr(sym, 0, component), values.data(),
              this->npart * sizeof(INT));
}

void ParticleSet::set(Sym<REAL> sym, const int component,
                      std::vector<REAL> &values) {
  NESOASSERT(this->contains(sym),
             "ParticleSet does not contain passed sym: " + sym.name);
  NESOASSERT(values.size() == static_cast<std::size_t>(this->npart),
             "Passed vector does not have the same length as the number of "
             "particles.");
  NESOASSERT(0 <= component && component < this->ncomp_real.at(sym),
             "Bad component passed.");
  std::memcpy(this->get_ptr(sym, 0, component), values.data(),
              this->npart * sizeof(REAL));
}

void ParticleSet::set(ParticleSet &particle_set) {
  NESOASSERT(particle_set.npart == this->npart,
             "Missmatch in particle counts between ParticleSets.");
  for (auto &sym_vector : this->values_int) {
    auto sym = sym_vector.first;
    if (particle_set.contains(sym)) {
      const auto ncomp = particle_set.ncomp_int.at(sym);
      NESOASSERT(ncomp == this->ncomp_int.at(sym),
                 "Component count does not match for sym " + sym.name);
      std::memcpy(this->get_ptr(sym, 0, 0), particle_set.get_ptr(sym, 0, 0),
                  this->npart * ncomp * sizeof(INT));
    }
  }
  for (auto &sym_vector : this->values_real) {
    auto sym = sym_vector.first;
    if (particle_set.contains(sym)) {
      const auto ncomp = particle_set.ncomp_real.at(sym);
      NESOASSERT(ncomp == this->ncomp_real.at(sym),
                 "Component count does not match for sym " + sym.name);
      std::memcpy(this->get_ptr(sym, 0, 0), particle_set.get_ptr(sym, 0, 0),
                  this->npart * ncomp * sizeof(REAL));
    }
  }
}

void ParticleSet::set(std::shared_ptr<ParticleSet> particle_set) {
  this->set(*particle_set);
}

} // namespace NESO::Particles
