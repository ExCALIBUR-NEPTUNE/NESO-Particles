#ifndef _NESO_PARTICLES_PARTICLE_GROUP
#define _NESO_PARTICLES_PARTICLE_GROUP

#include <CL/sycl.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include "access.hpp"
#include "cell_dat.hpp"
#include "cell_dat_compression.hpp"
#include "cell_dat_move.hpp"
#include "compute_target.hpp"
#include "domain.hpp"
#include "global_move.hpp"
#include "packing_unpacking.hpp"
#include "particle_dat.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

class ParticleGroup {
private:
  int ncell;
  BufferShared<INT> npart_cell;

  BufferDevice<INT> d_remove_cells;
  BufferDevice<INT> d_remove_layers;

  // these should be INT not int but hipsycl refused to do atomic refs on long
  // int
  BufferDevice<int> device_npart_cell;

  template <typename T>
  inline void compute_remove_compress_indicies(const int npart, T *usm_cells,
                                               T *usm_layers);
  template <typename T> inline void realloc_dat(ParticleDatShPtr<T> &dat) {
    dat->realloc(this->npart_cell);
    dat->wait_realloc();
  };
  template <typename T> inline void push_particle_spec(ParticleProp<T> prop) {
    this->particle_spec.push(prop);
  };

  // members for mpi communication
  // global communication context
  GlobalMove global_move_ctx;
  // method to map particle positions to global cells
  std::shared_ptr<MeshHierarchyGlobalMap> mesh_heirarchy_global_map;
  // neighbour communication context
  LocalMove local_move_ctx;
  // members for moving particles between local cells
  CellMove cell_move_ctx;

public:
  Domain domain;
  SYCLTarget &sycl_target;

  std::map<Sym<REAL>, ParticleDatShPtr<REAL>> particle_dats_real{};
  std::map<Sym<INT>, ParticleDatShPtr<INT>> particle_dats_int{};

  // ParticleDat storing Positions
  std::shared_ptr<Sym<REAL>> position_sym;
  ParticleDatShPtr<REAL> position_dat;
  // ParticleDat storing cell ids
  std::shared_ptr<Sym<INT>> cell_id_sym;
  ParticleDatShPtr<INT> cell_id_dat;
  // ParticleDat storing MPI rank
  std::shared_ptr<Sym<INT>> mpi_rank_sym;
  ParticleDatShPtr<INT> mpi_rank_dat;

  // ParticleSpec of all the ParticleDats of this ParticleGroup
  ParticleSpec particle_spec;

  // compression for dats when particles are removed
  LayerCompressor layer_compressor;

  ParticleGroup(Domain domain, ParticleSpec &particle_spec,
                SYCLTarget &sycl_target)
      : domain(domain), sycl_target(sycl_target),
        ncell(domain.mesh.get_cell_count()), d_remove_cells(sycl_target, 1),
        device_npart_cell(sycl_target, 1), d_remove_layers(sycl_target, 1),
        npart_cell(sycl_target, 1),
        layer_compressor(sycl_target, ncell, particle_dats_real,
                         particle_dats_int),
        global_move_ctx(sycl_target, layer_compressor, particle_dats_real,
                        particle_dats_int),
        local_move_ctx(sycl_target, layer_compressor, particle_dats_real,
                       particle_dats_int,
                       domain.mesh.get_local_communication_neighbours().size(),
                       domain.mesh.get_local_communication_neighbours().data()),
        cell_move_ctx(this->ncell, sycl_target, layer_compressor,
                      particle_dats_real, particle_dats_int)

  {

    this->npart_cell.realloc_no_copy(this->ncell);
    this->device_npart_cell.realloc_no_copy(this->ncell);

    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->npart_cell.ptr[cellx] = 0;
    }
    this->sycl_target.queue
        .memcpy(this->device_npart_cell.ptr, this->npart_cell.ptr,
                this->ncell * sizeof(int))
        .wait();

    for (auto &property : particle_spec.properties_real) {
      add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
    }
    for (auto &property : particle_spec.properties_int) {
      add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
    }
    // Create a ParticleDat to store the MPI rank of the particles in.
    mpi_rank_sym = std::make_shared<Sym<INT>>("NESO_MPI_RANK");
    mpi_rank_dat =
        ParticleDat(sycl_target, ParticleProp(*mpi_rank_sym, 2), ncell);
    add_particle_dat(mpi_rank_dat);
    this->global_move_ctx.set_mpi_rank_dat(mpi_rank_dat);
    this->local_move_ctx.set_mpi_rank_dat(mpi_rank_dat);

    this->layer_compressor.set_cell_id_dat(this->cell_id_dat);

    this->mesh_heirarchy_global_map = std::make_shared<MeshHierarchyGlobalMap>(
        this->sycl_target, this->domain.mesh, this->position_dat,
        this->cell_id_dat, this->mpi_rank_dat);
  }
  ~ParticleGroup() {}

  inline void add_particle_dat(ParticleDatShPtr<REAL> particle_dat);
  inline void add_particle_dat(ParticleDatShPtr<INT> particle_dat);

  inline void add_particles();
  template <typename U> inline void add_particles(U particle_data);
  inline void add_particles_local(ParticleSet &particle_data);

  inline int get_npart_local() { return this->position_dat->get_npart_local(); }

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

  inline void remove_particles(const int npart, const std::vector<INT> &cells,
                               const std::vector<INT> &layers);
  template <typename T>
  inline void remove_particles(const int npart, T *usm_cells, T *usm_layers);

  inline INT get_npart_cell(const int cell) {
    return this->npart_cell.ptr[cell];
  }
  inline ParticleSpec &get_particle_spec() { return this->particle_spec; }
  inline void global_move();
  inline void local_move();
  inline void hybrid_move();
  /*
   * Number of bytes required to store the data for one particle.
   */
  inline size_t particle_size();

  inline void cell_move() {
    this->cell_move_ctx.move();
    this->set_npart_cell_from_dat();
  };

  // set the cell particle counts on the ParticleGroup from the position dat
  inline void set_npart_cell_from_dat() {

    const auto k_ncell = this->ncell;
    auto k_dat_npart_cell = this->position_dat->d_npart_cell;
    auto k_npart_cell = this->npart_cell.ptr;
    auto k_device_npart_cell = this->device_npart_cell.ptr;

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(k_ncell), [=](sycl::id<1> idx) {
            k_npart_cell[idx] = k_dat_npart_cell[idx];
            k_device_npart_cell[idx] = k_dat_npart_cell[idx];
          });
        })
        .wait_and_throw();
  }

  template <typename... T> inline void print(T... args);
};

inline void
ParticleGroup::add_particle_dat(ParticleDatShPtr<REAL> particle_dat) {
  this->particle_dats_real[particle_dat->sym] = particle_dat;
  // Does this dat hold particle positions?
  if (particle_dat->positions) {
    this->position_dat = particle_dat;
    this->position_sym = std::make_shared<Sym<REAL>>(particle_dat->sym.name);
  }
  realloc_dat(particle_dat);
  // TODO clean up this ParticleProp handling
  push_particle_spec(ParticleProp(particle_dat->sym, particle_dat->ncomp,
                                  particle_dat->positions));
  particle_dat->set_npart_cells_device(this->device_npart_cell.ptr).wait();
  particle_dat->npart_device_to_host();
}
inline void
ParticleGroup::add_particle_dat(ParticleDatShPtr<INT> particle_dat) {
  this->particle_dats_int[particle_dat->sym] = particle_dat;
  // Does this dat hold particle cell ids?
  if (particle_dat->positions) {
    this->cell_id_dat = particle_dat;
    this->cell_id_sym = std::make_shared<Sym<INT>>(particle_dat->sym.name);
  }
  realloc_dat(particle_dat);
  // TODO clean up this ParticleProp handling
  push_particle_spec(ParticleProp(particle_dat->sym, particle_dat->ncomp,
                                  particle_dat->positions));
  particle_dat->set_npart_cells_device(this->device_npart_cell.ptr).wait();
  particle_dat->npart_device_to_host();
}

inline void ParticleGroup::add_particles(){};
template <typename U>
inline void ParticleGroup::add_particles(U particle_data){

};

/*
 * Number of bytes required to store the data for one particle.
 */
inline size_t ParticleGroup::particle_size() {
  size_t s = 0;
  for (auto &dat : this->particle_dats_real) {
    s += dat.second->cell_dat.row_size();
  }
  for (auto &dat : this->particle_dats_int) {
    s += dat.second->cell_dat.row_size();
  }
  return s;
};

inline void ParticleGroup::add_particles_local(ParticleSet &particle_data) {
  // loop over the cells of the new particles and allocate more space in the
  // dats

  const int npart_new = particle_data.npart;
  auto cellids = particle_data.get(*this->cell_id_sym);
  std::vector<INT> layers(npart_new);
  for (int px = 0; px < npart_new; px++) {
    auto cellindex = cellids[px];
    NESOASSERT((cellindex >= 0) && (cellindex < this->ncell),
               "Bad particle cellid)");

    layers[px] = this->npart_cell.ptr[cellindex]++;
  }

  for (auto &dat : this->particle_dats_real) {
    realloc_dat(dat.second);
    dat.second->append_particle_data(npart_new,
                                     particle_data.contains(dat.first), cellids,
                                     layers, particle_data.get(dat.first));
  }

  for (auto &dat : this->particle_dats_int) {
    realloc_dat(dat.second);
    dat.second->append_particle_data(npart_new,
                                     particle_data.contains(dat.first), cellids,
                                     layers, particle_data.get(dat.first));
  }

  // The append is async
  this->sycl_target.queue.wait();
  for (auto &dat : particle_dats_real) {
    dat.second->npart_host_to_device();
  }
  for (auto &dat : particle_dats_int) {
    dat.second->npart_host_to_device();
  }
  this->set_npart_cell_from_dat();
}

template <typename T>
inline void ParticleGroup::remove_particles(const int npart, T *usm_cells,
                                            T *usm_layers) {
  this->layer_compressor.remove_particles(npart, usm_cells, usm_layers);
  this->set_npart_cell_from_dat();
}

inline void ParticleGroup::remove_particles(const int npart,
                                            const std::vector<INT> &cells,
                                            const std::vector<INT> &layers) {

  this->d_remove_cells.realloc_no_copy(npart);
  this->d_remove_layers.realloc_no_copy(npart);

  auto k_cells = this->d_remove_cells.ptr;
  auto k_layers = this->d_remove_layers.ptr;

  NESOASSERT(cells.size() >= npart, "Bad cells length compared to npart");
  NESOASSERT(layers.size() >= npart, "Bad layers length compared to npart");

  auto b_cells = sycl::buffer<INT>(cells.data(), sycl::range<1>(npart));
  auto b_layers = sycl::buffer<INT>(layers.data(), sycl::range<1>(npart));

  this->sycl_target.queue
      .submit([&](sycl::handler &cgh) {
        auto a_cells = b_cells.get_access<sycl::access::mode::read>(cgh);
        auto a_layers = b_layers.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for<>(sycl::range<1>(static_cast<size_t>(npart)),
                           [=](sycl::id<1> idx) {
                             k_cells[idx] = static_cast<INT>(a_cells[idx]);
                             k_layers[idx] = static_cast<INT>(a_layers[idx]);
                           });
      })
      .wait_and_throw();

  this->remove_particles(npart, this->d_remove_cells.ptr,
                         this->d_remove_layers.ptr);
}

/*
 * Perform global move operation to send particles to the MPI ranks stored in
 * the first component of the NESO_MPI_RANK ParticleDat. Must be called
 * collectively on the communicator.
 */
inline void ParticleGroup::global_move() {
  this->global_move_ctx.move();
  this->set_npart_cell_from_dat();
}

/*
 * Perform local move operation to send particles to the MPI ranks stored in
 * the second component of the NESO_MPI_RANK ParticleDat. Must be called
 * collectively on the communicator.
 */
inline void ParticleGroup::local_move() {
  this->local_move_ctx.move();
  this->set_npart_cell_from_dat();
}

inline std::string fixed_width_format(INT value) {
  char buffer[128];
  const int err = snprintf(buffer, 128, "%ld", value);
  return std::string(buffer);
}
inline std::string fixed_width_format(REAL value) {
  char buffer[128];
  const int err = snprintf(buffer, 128, "%f", value);
  return std::string(buffer);
}

template <typename... T> inline void ParticleGroup::print(T... args) {

  PrintSpec print_spec(args...);

  std::cout << "==============================================================="
               "================="
            << std::endl;
  for (int cellx = 0; cellx < this->domain.mesh.get_cell_count(); cellx++) {

    std::vector<CellData<REAL>> cell_data_real;
    std::vector<CellData<INT>> cell_data_int;

    int nrow = -1;
    for (auto &symx : print_spec.syms_real) {
      auto cell_data = this->particle_dats_real[symx]->cell_dat.get_cell(cellx);
      cell_data_real.push_back(cell_data);
      if (nrow >= 0) {
        NESOASSERT(nrow == cell_data->nrow, "nrow missmatch");
      }
      nrow = cell_data->nrow;
    }
    for (auto &symx : print_spec.syms_int) {
      auto cell_data = this->particle_dats_int[symx]->cell_dat.get_cell(cellx);
      cell_data_int.push_back(cell_data);
      if (nrow >= 0) {
        NESOASSERT(nrow == cell_data->nrow, "nrow missmatch");
      }
      nrow = cell_data->nrow;
    }

    std::cout << "------- " << cellx << " -------" << std::endl;
    for (auto &symx : print_spec.syms_real) {
      std::cout << "| " << symx.name << " ";
    }
    for (auto &symx : print_spec.syms_int) {
      std::cout << "| " << symx.name << " ";
    }
    std::cout << "|" << endl;

    for (int rowx = 0; rowx < nrow; rowx++) {
      for (auto &cx : cell_data_real) {
        std::cout << "| ";
        for (int colx = 0; colx < cx->ncol; colx++) {
          std::cout << fixed_width_format((*cx)[colx][rowx]) << " ";
        }
      }
      for (auto &cx : cell_data_int) {
        std::cout << "| ";
        for (int colx = 0; colx < cx->ncol; colx++) {
          std::cout << fixed_width_format((*cx)[colx][rowx]) << " ";
        }
      }

      std::cout << "|" << std::endl;
    }
  }

  std::cout << "==============================================================="
               "================="
            << std::endl;
}

/*
 * Perform global move operation. Uses particle positions to determine
 * global/neighbour MPI ranks and uses the global then local move methods (in
 * that order). Must be called collectively on the communicator.
 */
inline void ParticleGroup::hybrid_move() {

  reset_mpi_ranks(this->mpi_rank_dat);
  this->domain.local_mapper->map(this->position_dat, this->cell_id_dat,
                                 this->mpi_rank_dat);
  this->mesh_heirarchy_global_map->execute();

  this->global_move_ctx.move();
  this->set_npart_cell_from_dat();

  this->domain.local_mapper->map(this->position_dat, this->cell_id_dat,
                                 this->mpi_rank_dat);

  this->local_move_ctx.move();
  this->set_npart_cell_from_dat();
}

} // namespace NESO::Particles

#endif
