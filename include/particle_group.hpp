#ifndef _NESO_PARTICLES_PARTICLE_GROUP
#define _NESO_PARTICLES_PARTICLE_GROUP

#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include "access.hpp"
#include "cell_dat.hpp"
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

  BufferDevice<INT> compress_cells_old;
  BufferDevice<INT> compress_cells_new;
  BufferDevice<INT> compress_layers_old;
  BufferDevice<INT> compress_layers_new;
  BufferDevice<INT> device_npart_cell;
  BufferDevice<INT> device_move_counters;

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
        ncell(domain.mesh.get_cell_count()),
        compress_cells_old(sycl_target, 1),
        compress_cells_new(sycl_target, 1),
        compress_layers_old(sycl_target, 1),
        compress_layers_new(sycl_target, 1),
        device_npart_cell(sycl_target, domain.mesh.get_cell_count()),
        device_move_counters(sycl_target, domain.mesh.get_cell_count())
  {

    for (auto &property : particle_spec.properties_real) {
      add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
    }
    for (auto &property : particle_spec.properties_int) {
      add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
    }
    this->npart_local = 0;
    this->npart_cell = std::vector<INT>(this->ncell);
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->npart_cell[cellx] = 0;
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

  inline CellData<REAL> get_cell(Sym<REAL> sym, const int cell) {
    return particle_dats_real[sym]->cell_dat.get_cell(cell);
  }

  inline CellData<INT> get_cell(Sym<INT> sym, const int cell) {
    return particle_dats_int[sym]->cell_dat.get_cell(cell);
  }

  inline void remove_particles(
    const int npart,
    const std::vector<INT> &cells,
    const std::vector<INT> &layers
  ){
    
    compress_cells_old.realloc_no_copy(npart);
    compress_layers_old.realloc_no_copy(npart);
    compress_layers_new.realloc_no_copy(npart);
    const auto cell_count = domain.mesh.get_cell_count();
    NESOASSERT(cell_count <= this->npart_cell.size(), "bad buffer lengths on cell count");
    device_npart_cell.realloc_no_copy(cell_count);

    sycl::buffer<INT> b_npart_cell(this->npart_cell.data(), sycl::range<1>{this->npart_cell.size()});
    sycl::buffer<INT> b_cells(cells.data(), sycl::range<1>{cells.size()});
    sycl::buffer<INT> b_layers(layers.data(), sycl::range<1>{layers.size()});

    auto device_npart_cell_ptr = device_npart_cell.ptr;
    auto device_move_counters_ptr = device_move_counters.ptr;

    auto compress_cells_old_ptr = compress_cells_old.ptr;
    auto compress_layers_old_ptr = compress_layers_old.ptr;
    auto compress_layers_new_ptr = compress_layers_new.ptr;


    INT *** cell_ids_ptr = this->cell_id_dat->cell_dat.device_ptr();
    
    this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      

      auto a_npart_cell = b_npart_cell.get_access<sycl::access::mode::read_write>(cgh);
      auto a_cells = b_cells.get_access<sycl::access::mode::read>(cgh);
      auto a_layers = b_layers.get_access<sycl::access::mode::read>(cgh);

      cgh.parallel_for<>(sycl::range<1>((size_t) cell_count), [=](sycl::id<1> idx) {
        device_npart_cell_ptr[idx] = a_npart_cell[idx];
        device_move_counters_ptr[idx] = 0;
      });
      cgh.parallel_for<>(sycl::range<1>((size_t) npart), [=](sycl::id<1> idx) {

        const auto cell = a_cells[idx];
        const auto layer = a_layers[idx];

        // Atomically do device_npart_cell_ptr[cell]--
        sycl::atomic_ref<
            INT,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device
        > element_atomic(device_npart_cell_ptr[cell]);
        element_atomic.fetch_add(-1);

        // indicate this particle is removed by setting the cell index to -1
        cell_ids_ptr[cell][0][layer] = -1;
      });
      cgh.parallel_for<>(sycl::range<1>((size_t) npart), [=](sycl::id<1> idx) {

        const auto cell = a_cells[idx];
        const auto layer = a_layers[idx];
        
        // Is this layer less than the new cell count?
        // If so then there is a particle in a row greater than the cell count
        // to be copied into this layer.
        if (layer < device_npart_cell_ptr[cell]){

          // If there are n rows to be filled in row indices less than the new
          // cell count then there are n rows greater than the new cell count
          // which are to be copied down. Atomically compute which one of those
          // rows this index copies.
          sycl::atomic_ref<
              INT,
              sycl::memory_order::relaxed,
              sycl::memory_scope::device
          > element_atomic(device_move_counters_ptr[cell]);
          const INT source_row_offset = element_atomic.fetch_add(1);

          // find the source row counting up from new_cell_count to old_cell_count
          // potential source rows will have a cell index >= 0. This index
          // should pick the i^th source row where i was computed from the
          // atomic above.
          INT found_count = 0;
          INT source_row = -1;
          for(INT rowx=device_npart_cell_ptr[cell] ; rowx<a_npart_cell[idx] ; rowx++){
            // Is this a potential source row?
            if (cell_ids_ptr[cell][0][layer] != -1){
              if(source_row_offset == found_count++){
                source_row = rowx;
                break;
              }
            }
          }
          
          compress_cells_old_ptr[idx] = cell;
          compress_layers_new_ptr[idx] = layer;
          compress_layers_old_ptr[idx] = source_row;

        } else {

          compress_cells_old_ptr[idx] = -1;
          compress_layers_new_ptr[idx] = -1;
          compress_layers_old_ptr[idx] = -1;

        }

      });
    }).wait();
  }


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

  const int npart = particle_data.npart;
  const int npart_new = this->npart_local + npart;
  auto cellids = particle_data.get(*this->cell_id_sym);
  std::vector<INT> layers(npart_new);
  for (int px = 0; px < npart_new; px++) {
    auto cellindex = cellids[px];
    NESOASSERT((cellindex >= 0) && (cellindex < this->ncell),
               "Bad particle cellid)");
    layers[px] = this->npart_cell[cellindex]++;
  }

  for (auto &dat : this->particle_dats_real) {
    dat.second->realloc(this->npart_cell);
    dat.second->append_particle_data(npart, particle_data.contains(dat.first),
                                     cellids, layers,
                                     particle_data.get(dat.first));
  }

  for (auto &dat : this->particle_dats_int) {
    dat.second->realloc(this->npart_cell);
    dat.second->append_particle_data(npart, particle_data.contains(dat.first),
                                     cellids, layers,
                                     particle_data.get(dat.first));
  }

  this->npart_local = npart_new;

  // The append is async
  this->sycl_target.queue.wait();
}

} // namespace NESO::Particles

#endif
