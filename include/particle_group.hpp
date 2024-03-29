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
#include "global_mapping.hpp"
#include "global_move.hpp"
#include "local_move.hpp"
#include "packing_unpacking.hpp"
#include "particle_dat.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

class ParticleSubGroup;

/**
 * Type to replace std::variant<Sym<INT>, Sym<REAL>> as the version tracking
 * key as this has been causing segfaults.
 */
struct ParticleDatVersionT {
  Sym<INT> si;
  Sym<REAL> sr;
  int index;

  ParticleDatVersionT(Sym<INT> s) {
    this->si = s;
    this->index = 0;
  }
  ParticleDatVersionT(Sym<REAL> s) {
    this->sr = s;
    this->index = 1;
  }
  ParticleDatVersionT() { this->index = -1; }
  ParticleDatVersionT &operator=(const ParticleDatVersionT &) = default;
  ParticleDatVersionT &operator=(const Sym<INT> &s) {
    this->si = s;
    this->index = 0;
    return *this;
  };
  ParticleDatVersionT &operator=(const Sym<REAL> &s) {
    this->sr = s;
    this->index = 1;
    return *this;
  };
  bool operator<(const ParticleDatVersionT &v) const {
    if (v.index == this->index) {
      if (this->index == -1) {
        return false;
      } else if (this->index == 0) {
        return this->si < v.si;
      } else {
        return this->sr < v.sr;
      }
    } else {
      return this->index < v.index;
    }
  }
};

/**
 *  Fundamentally a ParticleGroup is a collection of ParticleDats, domain and a
 *  compute device.
 */
class ParticleGroup {
  friend class ParticleSubGroup;

protected:
  // This type should be replaceable with typedef std::variant<Sym<INT>,
  // Sym<REAL>> ParticleDatVersion; But we see issues with nvc++.
  typedef ParticleDatVersionT ParticleDatVersion;
  typedef std::map<ParticleDatVersion, int64_t> ParticleDatVersionTracker;

private:
  int ncell;
  BufferHost<INT> h_npart_cell;

  BufferDevice<INT> d_remove_cells;
  BufferDevice<INT> d_remove_layers;

  template <typename T> inline void realloc_dat(ParticleDatSharedPtr<T> &dat) {
    dat->realloc(this->h_npart_cell);
    dat->wait_realloc();
  };
  template <typename T> inline void push_particle_spec(ParticleProp<T> prop) {
    this->particle_spec.push(prop);
  };

  // members for mpi communication
  // global communication context
  GlobalMove global_move_ctx;
  // method to map particle positions to global cells
  std::shared_ptr<MeshHierarchyGlobalMap> mesh_hierarchy_global_map;
  // neighbour communication context
  LocalMove local_move_ctx;
  // members for moving particles between local cells
  CellMove cell_move_ctx;

  std::map<ParticleDatVersion, std::tuple<int64_t, bool>> particle_dat_versions;

  inline void invalidate_callback_inner(ParticleDatVersion sym,
                                        const int mode) {
    std::get<0>(this->particle_dat_versions.at(sym))++;
    std::get<1>(this->particle_dat_versions.at(sym)) = (bool)mode;
  }

  inline void invalidate_callback_int(const Sym<INT> &sym, const int mode) {
    this->invalidate_callback_inner(sym, mode);
  }

  inline void invalidate_callback_real(const Sym<REAL> &sym, const int mode) {
    this->invalidate_callback_inner(sym, mode);
  }

  inline void add_invalidate_callback(ParticleDatSharedPtr<INT> particle_dat) {
    particle_dat->add_write_callback(
        std::bind(&ParticleGroup::invalidate_callback_int, this,
                  std::placeholders::_1, std::placeholders::_2));
  }

  inline void add_invalidate_callback(ParticleDatSharedPtr<REAL> particle_dat) {
    particle_dat->add_write_callback(
        std::bind(&ParticleGroup::invalidate_callback_real, this,
                  std::placeholders::_1, std::placeholders::_2));
  }

  template <typename T>
  inline void add_particle_dat_common(ParticleDatSharedPtr<T> particle_dat) {
    realloc_dat(particle_dat);
    push_particle_spec(ParticleProp(particle_dat->sym, particle_dat->ncomp,
                                    particle_dat->positions));
    particle_dat->set_npart_cells_host(this->h_npart_cell.ptr);
    particle_dat->npart_host_to_device();
    this->add_invalidate_callback(particle_dat);
    // This store is initialised with 1 and the to test keys should be
    // initialised with 0.
    ParticleDatVersion key{particle_dat->sym};
    std::tuple<int64_t, bool> value = {1, false};
    NESOASSERT(this->particle_dat_versions.count(key) == 0,
               "ParticleDat already in version tracker.");
    this->particle_dat_versions[key] = value;
  }

protected:
  /**
   * Returns true if the passed version is behind and was updated.
   */
  inline bool check_validation(ParticleDatVersionTracker &to_check) {
    bool updated = false;
    for (auto &item : to_check) {
      const auto &key = item.first;
      const int64_t to_check_value = item.second;
      const auto local_entry = this->particle_dat_versions.at(key);
      const int64_t local_value = std::get<0>(local_entry);
      const bool local_bool = std::get<1>(local_entry);
      // If a local count is different to the count on the to_check then an
      // updated is required on the object that holds to check.
      if (local_value != to_check_value) {
        updated = true;
        to_check.at(key) = local_value;
      }
      // If this bool has been set then an update is always required.
      if (local_bool) {
        updated = true;
      }
    }
    return updated;
  }

public:
  /// Disable (implicit) copies.
  ParticleGroup(const ParticleGroup &st) = delete;
  /// Disable (implicit) copies.
  ParticleGroup &operator=(ParticleGroup const &a) = delete;

  /// Domain this instance is defined over.
  DomainSharedPtr domain;
  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  /// Map from Sym instances to REAL valued ParticleDat instances.
  std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> particle_dats_real{};
  /// Map from Sym instances to INT valued ParticleDat instances.
  std::map<Sym<INT>, ParticleDatSharedPtr<INT>> particle_dats_int{};

  /// Sym of ParticleDat storing particle positions.
  std::shared_ptr<Sym<REAL>> position_sym;
  /// ParticleDat storing particle positions.
  ParticleDatSharedPtr<REAL> position_dat;
  /// Sym of ParticleDat storing particle cell ids.
  std::shared_ptr<Sym<INT>> cell_id_sym;
  /// ParticleDat storing particle cell ids.
  ParticleDatSharedPtr<INT> cell_id_dat;
  /// Sym of ParticleDat storing particle MPI ranks.
  std::shared_ptr<Sym<INT>> mpi_rank_sym;
  /// ParticleDat storing particle MPI ranks.
  ParticleDatSharedPtr<INT> mpi_rank_dat;

  /// ParticleSpec of all the ParticleDats of this ParticleGroup.
  ParticleSpec particle_spec;

  /// Layer compression instance for dats when particles are removed from cells.
  LayerCompressor layer_compressor;

  /// Explicitly free a ParticleGroup without relying on out-of-scope
  // destructor calls.
  inline void free() { this->global_move_ctx.free(); }

  /**
   * Construct a new ParticleGroup.
   *
   * @param domain Domain instance containing these particles.
   * @param particle_spec ParticleSpec that describes the ParticleDat instances
   * required.
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   */
  ParticleGroup(DomainSharedPtr domain, ParticleSpec &particle_spec,
                SYCLTargetSharedPtr sycl_target)
      : domain(domain), sycl_target(sycl_target),
        ncell(domain->mesh->get_cell_count()), d_remove_cells(sycl_target, 1),
        d_remove_layers(sycl_target, 1), h_npart_cell(sycl_target, 1),
        layer_compressor(sycl_target, ncell, particle_dats_real,
                         particle_dats_int),
        global_move_ctx(sycl_target, layer_compressor, particle_dats_real,
                        particle_dats_int),
        local_move_ctx(
            sycl_target, layer_compressor, particle_dats_real,
            particle_dats_int,
            domain->mesh->get_local_communication_neighbours().size(),
            domain->mesh->get_local_communication_neighbours().data()),
        cell_move_ctx(sycl_target, this->ncell, layer_compressor,
                      particle_dats_real, particle_dats_int)

  {

    this->h_npart_cell.realloc_no_copy(this->ncell);

    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->h_npart_cell.ptr[cellx] = 0;
    }

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
    this->cell_move_ctx.set_cell_id_dat(this->cell_id_dat);

    this->mesh_hierarchy_global_map = std::make_shared<MeshHierarchyGlobalMap>(
        this->sycl_target, this->domain->mesh, this->position_dat,
        this->cell_id_dat, this->mpi_rank_dat);

    // call the callback on the local mapper to complete the setup of that
    // object
    this->domain->local_mapper->particle_group_callback(*this);
  }
  ~ParticleGroup() {}

  /**
   *  Add a ParticleDat to the ParticleGroup after construction.
   *
   *  @param particle_dat New ParticleDat to add.
   */
  inline void add_particle_dat(ParticleDatSharedPtr<REAL> particle_dat);

  /**
   *  Add a ParticleDat to the ParticleGroup after construction.
   *
   *  @param particle_dat New ParticleDat to add.
   */
  inline void add_particle_dat(ParticleDatSharedPtr<INT> particle_dat);

  /**
   *  Add particles to the ParticleGroup. Any rank may add particles that exist
   *  anywhere in the domain. This call is collective across the ParticleGroup
   *  and ranks that do not add particles should not pass any new particle
   *  data.
   */
  inline void add_particles();

  /**
   *  Add particles to the ParticleGroup. Any rank may add particles that exist
   *  anywhere in the domain. This call is collective across the ParticleGroup
   *  and ranks that do not add particles should not pass any new particle
   *  data.
   *
   *  @param particle_data New particle data to add to the ParticleGroup.
   */
  template <typename U> inline void add_particles(U particle_data);

  /**
   *  Add particles only to this MPI rank. It is assumed that the added
   *  particles are in the domain region owned by this MPI rank. If not, see
   *  `ParticleGroup::add_particles`.
   *
   *  @param particle_data New particles to add.
   */
  inline void add_particles_local(ParticleSet &particle_data);

  /**
   *  Get the total number of particles on this MPI rank.
   *
   *  @returns Local particle count.
   */
  inline int get_npart_local() { return this->position_dat->get_npart_local(); }

  /**
   *  Determine if the ParticleGroup contains a ParticleDat of a given name.
   *
   *  @param sym Symbol of ParticleDat.
   *  @returns True if ParticleDat exists on this ParticleGroup.
   */
  inline bool contains_dat(Sym<REAL> sym) {
    return (bool)this->particle_dats_real.count(sym);
  }

  /**
   *  Determine if the ParticleGroup contains a ParticleDat of a given name.
   *
   *  @param sym Symbol of ParticleDat.
   *  @returns True if ParticleDat exists on this ParticleGroup.
   */
  inline bool contains_dat(Sym<INT> sym) {
    return (bool)this->particle_dats_int.count(sym);
  }

  /**
   * Template for get_dat method called like:
   * ParticleGroup::get_dat(Sym<REAL>("POS")) for a real valued ParticleDat.
   *
   * @param name Name of ParticleDat.
   * @param check_exists Check if the dat exists, default true.
   * @returns ParticleDatSharedPtr<T> particle dat.
   */
  template <typename T>
  inline ParticleDatSharedPtr<T> get_dat(Sym<T> sym,
                                         const bool check_exists = true) {
    if (check_exists) {
      const bool dat_exists = this->contains_dat(sym);
      NESOASSERT(dat_exists,
                 "This ParticleGroup does not contain the requested dat.");
    }
    return (*this)[sym];
  }

  /**
   *  Users are recomended to use "get_dat" instead.
   *  Enables access to the ParticleDat instances using the subscript operators.
   *
   *  @param sym Sym<REAL> of ParticleDat to access.
   */
  inline ParticleDatSharedPtr<REAL> &operator[](Sym<REAL> sym) {
    return this->particle_dats_real.at(sym);
  };
  /**
   *  Users are recomended to use "get_dat" instead.
   *  Enables access to the ParticleDat instances using the subscript operators.
   *
   *  @param sym Sym<INT> of ParticleDat to access.
   */
  inline ParticleDatSharedPtr<INT> &operator[](Sym<INT> sym) {
    return this->particle_dats_int.at(sym);
  };

  /**
   *  Get a CellData instance that holds all the particle data for a
   *  ParticleDat for a cell.
   *
   *  @param sym Sym<REAL> indicating which ParticleDat to access.
   *  @param cell Cell index to access.
   *  @returns CellData for requested cell.
   */
  inline CellData<REAL> get_cell(Sym<REAL> sym, const int cell) {
    return particle_dats_real[sym]->cell_dat.get_cell(cell);
  }

  /**
   *  Get a CellData instance that holds all the particle data for a
   *  ParticleDat for a cell.
   *
   *  @param sym Sym<INT> indicating which ParticleDat to access.
   *  @param cell Cell index to access.
   *  @returns CellData for requested cell.
   */
  inline CellData<INT> get_cell(Sym<INT> sym, const int cell) {
    return particle_dats_int[sym]->cell_dat.get_cell(cell);
  }

  /**
   *  Remove particles from the ParticleGroup.
   *
   *  @param npart Number of particles to remove.
   *  @param cells Vector of particle cells.
   *  @param layers Vector of particle layers(rows).
   */
  inline void remove_particles(const int npart, const std::vector<INT> &cells,
                               const std::vector<INT> &layers);
  /**
   *  Remove particles from the ParticleGroup.
   *
   *  @param npart Number of particles to remove.
   *  @param usm_cells Device accessible array of particle cells.
   *  @param usm_layers Device accessible array of particle layers(rows).
   */
  template <typename T>
  inline void remove_particles(const int npart, T *usm_cells, T *usm_layers);

  /**
   * Get the number of particles in a cell.
   *
   * @param cell Cell to query.
   * @returns Number of particles in queried cell.
   */
  inline INT get_npart_cell(const int cell) {
    return this->h_npart_cell.ptr[cell];
  }

  /**
   *  Get the ParticleSpec of the ParticleDats stored in this ParticleGroup.
   *
   *  @returns ParticleSpec of stored particles.
   */
  inline ParticleSpec &get_particle_spec() { return this->particle_spec; }

  /**
   * Use the MPI ranks in the first component of the MPI rank dat to move
   * particles globally using the MeshHierarchy of the mesh in the domain.
   *
   * Must be called collectively on the ParticleGroup.
   */
  inline void global_move();
  /**
   * Use the MPI ranks in the second component of the MPI rank dat to move
   * particles to neighbouring ranks using the neighbouring ranks defined on
   * the mesh.
   *
   * Must be called collectively on the ParticleGroup.
   */
  inline void local_move();
  /**
   * Perform a global move using non-negative MPI ranks in the first component
   * of the MPI rank dat. Then bin moved particles into local cells to obtain
   * any required ranks for the local move. Second perform a local move to move
   * particles based on the MPI rank stored in the second component of the MPI
   * rank dat.
   *
   * Must be called collectively on the ParticleGroup.
   */
  inline void hybrid_move();
  /**
   * Number of bytes required to store the data for one particle.
   *
   * @returns Number of bytes required to store one particle.
   */
  inline size_t particle_size();

  /**
   *  Move particles between cells using the cell ids stored in the cell id dat.
   */
  inline void cell_move() {
    this->cell_move_ctx.move();
    this->set_npart_cell_from_dat();
  };

  /**
   *  Copy the particle counts per cell from the position ParticleDat to the
   *  npart cell array of the ParticleGroup.
   */
  inline void set_npart_cell_from_dat() {
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->h_npart_cell.ptr[cellx] = this->position_dat->h_npart_cell[cellx];
    }
  }

  /**
   *  Print particle data for all particles for the specified ParticleDats.
   *  Empty cells are not printed.
   *
   *  @param args Sym<REAL> or Sym<INT> instances that indicate which particle
   *  data to print.
   */
  template <typename... T> inline void print(T... args);

  /**
   *  Remove a ParticleDat from the ParticleGroup
   *
   *  @param sym Sym object that refers to a ParticleDat
   */
  inline void remove_particle_dat(Sym<REAL> sym) {
    NESOASSERT(this->particle_dats_real.count(sym) == 1,
               "ParticleDat not found.");
    this->particle_dats_real.erase(sym);
    this->particle_dat_versions.erase(sym);
  }
  /**
   *  Remove a ParticleDat from the ParticleGroup
   *
   *  @param sym Sym object that refers to a ParticleDat
   */
  inline void remove_particle_dat(Sym<INT> sym) {
    NESOASSERT(this->particle_dats_int.count(sym) == 1,
               "ParticleDat not found.");
    this->particle_dats_int.erase(sym);
    this->particle_dat_versions.erase(sym);
  }
};

inline void
ParticleGroup::add_particle_dat(ParticleDatSharedPtr<REAL> particle_dat) {
  NESOASSERT(this->particle_dats_real.count(particle_dat->sym) == 0,
             "ParticleDat Sym already exists in ParticleGroup.");
  this->particle_dats_real[particle_dat->sym] = particle_dat;
  // Does this dat hold particle positions?
  if (particle_dat->positions) {
    this->position_dat = particle_dat;
    this->position_sym = std::make_shared<Sym<REAL>>(particle_dat->sym.name);
  }
  add_particle_dat_common(particle_dat);
}
inline void
ParticleGroup::add_particle_dat(ParticleDatSharedPtr<INT> particle_dat) {
  NESOASSERT(this->particle_dats_int.count(particle_dat->sym) == 0,
             "ParticleDat Sym already exists in ParticleGroup.");
  this->particle_dats_int[particle_dat->sym] = particle_dat;
  // Does this dat hold particle cell ids?
  if (particle_dat->positions) {
    this->cell_id_dat = particle_dat;
    this->cell_id_sym = std::make_shared<Sym<INT>>(particle_dat->sym.name);
  }
  add_particle_dat_common(particle_dat);
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

    layers[px] = this->h_npart_cell.ptr[cellindex]++;
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
  this->sycl_target->queue.wait();
  for (auto &dat : particle_dats_real) {
    dat.second->npart_host_to_device();
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      NESOASSERT(dat.second->h_npart_cell[cellx] ==
                     this->h_npart_cell.ptr[cellx],
                 "Bad cell count");
    }
  }
  for (auto &dat : particle_dats_int) {
    dat.second->npart_host_to_device();
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      NESOASSERT(dat.second->h_npart_cell[cellx] ==
                     this->h_npart_cell.ptr[cellx],
                 "Bad cell count");
    }
  }
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

  this->sycl_target->queue
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

  SymStore print_spec(args...);

  std::cout << "==============================================================="
               "================="
            << std::endl;
  for (int cellx = 0; cellx < this->domain->mesh->get_cell_count(); cellx++) {
    if (this->h_npart_cell.ptr[cellx] > 0) {

      std::vector<CellData<REAL>> cell_data_real;
      std::vector<CellData<INT>> cell_data_int;

      int nrow = -1;
      for (auto &symx : print_spec.syms_real) {
        auto cell_data =
            this->particle_dats_real[symx]->cell_dat.get_cell(cellx);
        cell_data_real.push_back(cell_data);
        if (nrow >= 0) {
          NESOASSERT(nrow == cell_data->nrow, "nrow missmatch");
        }
        nrow = cell_data->nrow;
      }
      for (auto &symx : print_spec.syms_int) {
        auto cell_data =
            this->particle_dats_int[symx]->cell_dat.get_cell(cellx);
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
      std::cout << "|" << std::endl;

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
  this->domain->local_mapper->map(*this);
  this->mesh_hierarchy_global_map->execute();

  this->global_move_ctx.move();
  this->set_npart_cell_from_dat();

  this->domain->local_mapper->map(*this, 0);

  this->local_move_ctx.move();
  this->set_npart_cell_from_dat();
}

typedef std::shared_ptr<ParticleGroup> ParticleGroupSharedPtr;

} // namespace NESO::Particles

#endif
