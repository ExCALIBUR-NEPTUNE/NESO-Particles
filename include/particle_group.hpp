#ifndef _NESO_PARTICLES_PARTICLE_GROUP
#define _NESO_PARTICLES_PARTICLE_GROUP

#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include "access.hpp"
#include "compute_target.hpp"
#include "domain.hpp"
#include "particle_dat.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

class ParticleGroup {
private:
  int ncell;
  int npart_local;
  std::vector<INT> npart_cell;
  std::vector<INT> npart_cell_tmp;

public:
  Domain domain;
  SYCLTarget &sycl_target;

  std::map<Sym<REAL>, ParticleDatShPtr<REAL>> particle_dats_real{};
  std::map<Sym<INT>, ParticleDatShPtr<INT>> particle_dats_int{};

  std::shared_ptr<Sym<REAL>> position_sym;
  ParticleDatShPtr<REAL> position_dat;
  std::shared_ptr<Sym<INT>> cell_id_sym;
  ParticleDatShPtr<INT> cell_id_dat;

  ParticleGroup(Domain domain, ParticleSpec &particle_spec,
                SYCLTarget &sycl_target)
      : domain(domain), sycl_target(sycl_target),
        ncell(domain.mesh.get_cell_count()) {

    for (auto &property : particle_spec.properties_real) {
      add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
    }
    for (auto &property : particle_spec.properties_int) {
      add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
    }
    this->npart_local = 0;
    this->npart_cell = std::vector<INT>(this->ncell);
    this->npart_cell_tmp = std::vector<INT>(this->ncell);
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->npart_cell[cellx] = 0;
      this->npart_cell_tmp[cellx] = 0;
    }
  }
  ~ParticleGroup() {}

  inline void add_particle_dat(ParticleDatShPtr<REAL> particle_dat);
  inline void add_particle_dat(ParticleDatShPtr<INT> particle_dat);

  inline void add_particles();
  template <typename U> inline void add_particles(U particle_data);
  inline void add_particles_local(ParticleSet &particle_data);

  inline int get_npart_local() { return this->npart_local; }

  inline ParticleDatShPtr<REAL> &operator[](Sym<REAL> sym) {
    return this->particle_dats_real.at(sym);
  };
  inline ParticleDatShPtr<INT> &operator[](Sym<INT> sym) {
    return this->particle_dats_int.at(sym);
  };
};

inline void
ParticleGroup::add_particle_dat(ParticleDatShPtr<REAL> particle_dat) {
  this->particle_dats_real[particle_dat->sym] = particle_dat;
  // Does this dat hold particle positions?
  if (particle_dat->positions) {
    this->position_dat = particle_dat;
    this->position_sym = std::make_shared<Sym<REAL>>(particle_dat->sym.name);
  }
}
inline void
ParticleGroup::add_particle_dat(ParticleDatShPtr<INT> particle_dat) {
  this->particle_dats_int[particle_dat->sym] = particle_dat;
  // Does this dat hold particle cell ids?
  if (particle_dat->positions) {
    this->cell_id_dat = particle_dat;
    this->cell_id_sym = std::make_shared<Sym<INT>>(particle_dat->sym.name);
  }
}

inline void ParticleGroup::add_particles(){};
template <typename U>
inline void ParticleGroup::add_particles(U particle_data){

};

inline void ParticleGroup::add_particles_local(ParticleSet &particle_data) {
  // loop over the cells of the new particles and allocate more space in the
  // dats
  for (int cellx = 0; cellx < this->ncell; cellx++) {
    this->npart_cell_tmp[cellx] = this->npart_cell[cellx];
  }
  const int npart = particle_data.npart;
  const int npart_new = this->npart_local + npart;
  auto cellids = particle_data.get(*this->cell_id_sym);
  for (int px = 0; px < npart_new; px++) {
    auto cellindex = cellids[px];
    NESOASSERT((cellindex >= 0) && (cellindex < this->ncell),
               "Bad particle cellid)");
    this->npart_cell_tmp[cellindex]++;
  }

  for (auto &dat : this->particle_dats_real) {
    dat.second->realloc(this->npart_cell_tmp);

    dat.second->append_particle_data(npart, particle_data.contains(dat.first),
                                     cellids, particle_data.get(dat.first));
  }

  for (auto &dat : this->particle_dats_int) {
    dat.second->realloc(this->npart_cell_tmp);

    dat.second->append_particle_data(npart, particle_data.contains(dat.first),
                                     cellids, particle_data.get(dat.first));
  }

  this->npart_local = npart_new;

  // The append is async
  this->sycl_target.queue.wait();
}

} // namespace NESO::Particles

#endif
