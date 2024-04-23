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
#include "containers/product_matrix.hpp"
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
class DescendantProducts;
class ParticleSubGroup;
namespace ParticleSubGroupImplementation {
class SubGroupSelector;
}

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
  friend class ParticleSubGroupImplementation::SubGroupSelector;
  friend class ParticleSubGroup;

protected:
  // This type should be replaceable with typedef std::variant<Sym<INT>,
  // Sym<REAL>> ParticleDatVersion; But we see issues with nvc++.
  typedef ParticleDatVersionT ParticleDatVersion;
  typedef std::map<ParticleDatVersion, int64_t> ParticleDatVersionTracker;

  int ncell;
  int npart_local;
  BufferHost<INT> h_npart_cell;
  BufferDevice<INT> d_npart_cell;

  BufferDevice<INT> d_remove_cells;
  BufferDevice<INT> d_remove_layers;

  template <typename T>
  inline void realloc_dat_start(ParticleDatSharedPtr<T> &dat) {
    dat->realloc(this->h_npart_cell);
  };
  template <typename T>
  inline void realloc_dat_wait(ParticleDatSharedPtr<T> &dat) {
    dat->wait_realloc();
  };
  template <typename T> inline void realloc_dat(ParticleDatSharedPtr<T> &dat) {
    this->realloc_dat_start(dat);
    this->realloc_dat_wait(dat);
  };

  inline void realloc_all_dats() {
    for (auto &dat : this->particle_dats_real) {
      realloc_dat_start(dat.second);
    }
    for (auto &dat : this->particle_dats_int) {
      realloc_dat_start(dat.second);
    }
    for (auto &dat : this->particle_dats_real) {
      realloc_dat_wait(dat.second);
    }
    for (auto &dat : this->particle_dats_int) {
      realloc_dat_wait(dat.second);
    }
  }

  inline void check_dats_and_group_agree() {
    for (auto &dat : particle_dats_real) {
      for (int cellx = 0; cellx < this->ncell; cellx++) {
        NESOASSERT(dat.second->h_npart_cell[cellx] ==
                       this->h_npart_cell.ptr[cellx],
                   "Bad cell count");
      }
    }
    for (auto &dat : particle_dats_int) {
      for (int cellx = 0; cellx < this->ncell; cellx++) {
        NESOASSERT(dat.second->h_npart_cell[cellx] ==
                       this->h_npart_cell.ptr[cellx],
                   "Bad cell count");
      }
    }
  }

  template <typename T>
  inline bool existing_compatible_dat(ParticleDatSharedPtr<T> particle_dat) {
    auto sym = particle_dat->sym;
    if (this->contains_dat(sym)) {
      auto existing_dat = this->get_dat(sym);
      NESOASSERT(
          existing_dat->ncomp == particle_dat->ncomp,
          "Existing ParticleDat and ParticleDat to add are incompaptible");
      return true;
    } else {
      return false;
    }
  }

  inline void get_new_layers(const int npart, const INT *RESTRICT cells_ptr,
                             INT *RESTRICT layers_ptr) {
    buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait_and_throw();
    INT *k_npart_cell = this->d_npart_cell.ptr;
    const INT k_ncell = this->ncell;
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(static_cast<size_t>(npart)), [=](sycl::id<1> idx) {
                const INT cell = cells_ptr[idx];
                if ((-1 < cell) && (cell < k_ncell)) {
                  sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      element_atomic(k_npart_cell[cell]);
                  const INT layer = element_atomic.fetch_add((INT)1);
                  layers_ptr[idx] = layer;
                } else {
                  layers_ptr[idx] = -1;
                }
              });
        })
        .wait_and_throw();
    buffer_memcpy(this->h_npart_cell, this->d_npart_cell).wait_and_throw();
  }

  template <typename T>
  inline void zero_dat_properties(std::shared_ptr<T> dat, const int npart,
                                  const INT *RESTRICT cells_ptr,
                                  const INT *RESTRICT layers_ptr,
                                  EventStack &es) {
    auto dat_ptr = dat->impl_get();
    const int k_ncomp = dat->ncomp;
    es.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(sycl::range<1>(static_cast<size_t>(npart)),
                         [=](sycl::id<1> idx) {
                           const INT cell = cells_ptr[idx];
                           const INT layer = layers_ptr[idx];
                           if (layer > -1) {
                             for (int nx = 0; nx < k_ncomp; nx++) {
                               dat_ptr[cell][nx][layer] = 0;
                             }
                           }
                         });
    }));
  }

  template <typename T> inline void push_particle_spec(ParticleProp<T> prop) {
    this->particle_spec.push(prop);
  };

  // members for mpi communication
  // global communication context
  GlobalMove global_move_ctx;
  // method to map particle positions to global cells
  std::shared_ptr<MeshHierarchyGlobalMap> mesh_hierarchy_global_map;
  // neighbour communication context
  std::unique_ptr<LocalMove> local_move_ctx;
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
  inline void remove_particle_dat_common(ParticleDatSharedPtr<T> particle_dat) {
    this->particle_dat_versions.erase(particle_dat->sym);
    this->particle_spec.remove(ParticleProp(
        particle_dat->sym, particle_dat->ncomp, particle_dat->positions));
  }

  template <typename T>
  inline void add_particle_dat_common(ParticleDatSharedPtr<T> particle_dat) {
    realloc_dat(particle_dat);
    push_particle_spec(ParticleProp(particle_dat->sym, particle_dat->ncomp,
                                    particle_dat->positions));
    particle_dat->set_npart_cells_host(this->h_npart_cell.ptr);
    particle_dat->set_d_npart_cell_es(this->dh_npart_cell_es->d_buffer.ptr);
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

  /// BufferDeviceHost holding the exclusive sum of the number of particles in
  /// each cell.
  std::shared_ptr<BufferDeviceHost<INT>> dh_npart_cell_es;

  /// The sub group creation requires get_npart_local temporary space and we
  /// create this space here on the parent subgroup such that it is shared
  /// across all subgroups to avoid exessive memory usage/realloc
  std::shared_ptr<BufferDevice<int>> d_sub_group_layers;

  /**
   * Returns true if the passed version is behind and can be updated. By
   * default updates the passed version.
   */
  inline bool check_validation(ParticleDatVersionTracker &to_check,
                               const bool update_to_check = true) {
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
        if (update_to_check) {
          to_check.at(key) = local_value;
        }
      }
      // If this bool has been set then an update is always required.
      if (local_bool) {
        updated = true;
      }
    }
    return updated;
  }

  inline void recompute_npart_cell_es() {
    auto h_ptr_s = this->h_npart_cell.ptr;
    auto h_ptr = this->dh_npart_cell_es->h_buffer.ptr;
    INT total = 0;
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      h_ptr[cellx] = total;
      total += h_ptr_s[cellx];
    }
    this->npart_local = static_cast<int>(total);
    this->dh_npart_cell_es->host_to_device();
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
        d_npart_cell(sycl_target, 1),
        layer_compressor(sycl_target, ncell, particle_dats_real,
                         particle_dats_int),
        global_move_ctx(sycl_target, layer_compressor, particle_dats_real,
                        particle_dats_int),
        cell_move_ctx(sycl_target, this->ncell, layer_compressor,
                      particle_dats_real, particle_dats_int),
        npart_local(0) {

    this->h_npart_cell.realloc_no_copy(this->ncell);
    this->d_npart_cell.realloc_no_copy(this->ncell);

    this->dh_npart_cell_es =
        std::make_shared<BufferDeviceHost<INT>>(sycl_target, this->ncell);

    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->h_npart_cell.ptr[cellx] = 0;
      this->dh_npart_cell_es->h_buffer.ptr[cellx] = 0;
    }
    buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait();
    this->dh_npart_cell_es->host_to_device();

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

    this->local_move_ctx = std::make_unique<LocalMove>(
        sycl_target, layer_compressor, particle_dats_real, particle_dats_int,
        domain->mesh->get_local_communication_neighbours().size(),
        domain->mesh->get_local_communication_neighbours().data());
    this->local_move_ctx->set_mpi_rank_dat(mpi_rank_dat);

    this->layer_compressor.set_cell_id_dat(this->cell_id_dat);
    this->cell_move_ctx.set_cell_id_dat(this->cell_id_dat);

    this->mesh_hierarchy_global_map = std::make_shared<MeshHierarchyGlobalMap>(
        this->sycl_target, this->domain->mesh, this->position_dat,
        this->cell_id_dat, this->mpi_rank_dat);

    // call the callback on the local mapper to complete the setup of that
    // object
    this->domain->local_mapper->particle_group_callback(*this);
    this->d_sub_group_layers =
        std::make_shared<BufferDevice<int>>(sycl_target, 1);
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
   *  Add particles only to this MPI rank. It is assumed that the added
   *  particles are in the domain region owned by this MPI rank. If not, see
   *  `ParticleGroup::add_particles`.
   *
   *  @param particle_data New particles to add.
   */
  inline void add_particles_local(ParticleSetSharedPtr particle_data);

protected:
  inline void add_particles_local(std::shared_ptr<ProductMatrix> product_matrix,
                                  const INT *d_cells, const INT *d_layers,
                                  ParticleGroup *source_particle_group);

public:
  /**
   *  Add particles only to this MPI rank. It is assumed that the added
   *  particles are in the domain region owned by this MPI rank. If not, see
   *  `ParticleGroup::add_particles`. Particle properties which exist on the
   *  ParticleGroup and not in the ProductMatrix will be zero initialised.
   *  Particle properties which exist on the ProductMatrix and not in the
   *  ParticleGroup will be ignored.
   *
   *  @param product_matrix New particles to add.
   */
  inline void
  add_particles_local(std::shared_ptr<ProductMatrix> product_matrix);

  /**
   *  Add new particles to this ParticleGroup via a DescendantProducts
   *  instance. If no alternative source ParticleGroup is provided then
   *  properties are either defined in the DescendantProducts, from which they
   *  are copied, or if the property is not defined in the DescendantProducts
   *  then the property values are copied from the parent particle.
   *
   *  When a source ParticleGroup is provided: Properties which are defined in
   *  the DescendantProducts are copied from the DescendantProducts. If a
   *  property is defined in the source ParticleGroup and not in this
   *  ParticleGroup then it is ignored. If a property is defined in this
   *  ParticleGroup and not in the source ParticleGroup then the values are
   *  zero initialised. Properties not defined in the DescendantProducts which
   *  exist as particle properties in both the source ParticleGroup and this
   *  ParticleGroup are copied from the parent particles. When the source
   *  ParticleGroup is specified the parent indices in DescendantProducts
   *  always refer to particles in the parent ParticleGroup.
   *
   *  @param descendant_products New particles to add.
   *  @param source_particleGroup Alternative ParticleGroup to use as the set
   *  of parent particles. The descendant_products parent indices must have
   *  been created with this ParticleGroup. By default no alternative
   *  ParticleGroup is specified and the parents are assumed to be in the
   *  ParticleGroup on which add_particles_local was called.
   */
  inline void add_particles_local(
      std::shared_ptr<DescendantProducts> descendant_products,
      std::shared_ptr<ParticleGroup> source_particle_group = nullptr);

  /**
   * Add particles to this ParticleGroup from another ParticleGroup. Properties
   * which exist in the destination ParticleGroup and not the source
   * ParticleGroup are zero initialised. Properties which exist in both
   * ParticleGroups are copied. Properties which only exist in the source
   * ParticleGroup are ignored.
   *
   * Particle properties will be copied cell-wise from the source to
   * destination ParticleGroups. This cell-wise copy requires that the source
   * and destination domains share the same number of cells on each MPI rank.
   *
   *  @param particle_group New particles to add.
   */
  inline void
  add_particles_local(std::shared_ptr<ParticleGroup> particle_group);

  /**
   * Add particles to this ParticleGroup from another ParticleGroup. Properties
   * which exist in the destination ParticleGroup and not the source
   * ParticleGroup are zero initialised. Properties which exist in both
   * ParticleGroups are copied. Properties which only exist in the source
   * ParticleGroup are ignored.
   *
   * Particle properties will be copied cell-wise from the source to
   * destination ParticleGroups. This cell-wise copy requires that the source
   * and destination domains share the same number of cells on each MPI rank.
   *
   *  @param particle_sub_group New particles to add.
   */
  inline void
  add_particles_local(std::shared_ptr<ParticleSubGroup> particle_sub_group);

  /**
   *  Get the total number of particles on this MPI rank.
   *
   *  @returns Local particle count.
   */
  inline int get_npart_local() { return this->npart_local; }

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
                 "This ParticleGroup does not contain the requested dat: " +
                     sym.name);
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
   * Clear all particles from the ParticleGroup on the calling MPI rank.
   */
  inline void clear();

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
   *  Remove all the particles particles from the ParticleGroup which are
   *  members of a ParticleSubGroup.
   *
   *  @param particle_sub_group.
   */
  inline void
  remove_particles(std::shared_ptr<ParticleSubGroup> particle_sub_group);

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
    buffer_memcpy(this->d_npart_cell, this->h_npart_cell).wait_and_throw();
    this->recompute_npart_cell_es();
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
    this->remove_particle_dat_common(this->particle_dats_real.at(sym));
    this->particle_dats_real.erase(sym);
  }
  /**
   *  Remove a ParticleDat from the ParticleGroup
   *
   *  @param sym Sym object that refers to a ParticleDat
   */
  inline void remove_particle_dat(Sym<INT> sym) {
    NESOASSERT(this->particle_dats_int.count(sym) == 1,
               "ParticleDat not found.");
    this->remove_particle_dat_common(this->particle_dats_int.at(sym));
    this->particle_dats_int.erase(sym);
  }

  /**
   * Create a ParticleSet containing the data from particles held in the
   * ParticleGroup. e.g. to Extract the first two particles from the second
   * cell:
   *
   * cells  = [1, 1]
   * layers = [0, 1]
   *
   * @param cells Vector of cell indices of particles to extract.
   * @param cells Vector of layer indices of particles to extract.
   * @returns ParticleSet of particle data.
   */
  inline ParticleSetSharedPtr get_particles(std::vector<INT> &cells,
                                            std::vector<INT> &layers) {
    NESOASSERT(cells.size() == layers.size(),
               "Cells and layers vectors have different sizes.");
    const int num_particles = cells.size();
    auto ps = std::make_shared<ParticleSet>(num_particles, this->particle_spec);
    const INT num_cells = this->domain->mesh->get_cell_count();

    EventStack es;

    auto lambda_copy = [&](const std::size_t num_bytes_per_element, auto dat,
                           const INT particle_index, const INT cell,
                           const INT layer) {
      const int ncomp = dat->ncomp;
      for (int nx = 0; nx < ncomp; nx++) {
        auto dest_ptr = ps->get_ptr(dat->sym, particle_index, nx);
        const auto source_ptr = dat->col_device_const_ptr(cell, nx) + layer;
        es.push(this->sycl_target->queue.memcpy(dest_ptr, source_ptr,
                                                num_bytes_per_element));
      }
    };

    for (int px = 0; px < num_particles; px++) {
      const INT cell = cells.at(px);
      const INT layer = layers.at(px);
      NESOASSERT((cell > -1) && (cell < num_cells), "Cell index not in range.");
      NESOASSERT((layer > -1) && (layer < this->get_npart_cell(cell)),
                 "Layer index not in range.");
      for (auto &dat : this->particle_dats_real) {
        lambda_copy(sizeof(REAL), dat.second, px, cell, layer);
      }
      for (auto &dat : this->particle_dats_int) {
        lambda_copy(sizeof(INT), dat.second, px, cell, layer);
      }
    }

    es.wait();
    return ps;
  }
};

typedef std::shared_ptr<ParticleGroup> ParticleGroupSharedPtr;

} // namespace NESO::Particles

#endif
