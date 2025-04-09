#ifndef _NESO_PARTICLES_PARTICLE_IO
#define _NESO_PARTICLES_PARTICLE_IO

#include "loop/particle_loop.hpp"
#include "particle_group.hpp"
#include "particle_spec.hpp"
#include "particle_sub_group/particle_sub_group.hpp"
#include "typedefs.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mpi.h>
#include <string>

namespace NESO::Particles {
#ifdef NESO_PARTICLES_HDF5

class H5Part {

private:
  std::string filename;
  CommPair &comm_pair;
  SymStore sym_store;
  bool is_closed = true;
  hid_t plist_id, file_id;
  INT step;
  ParticleGroupSharedPtr particle_group;
  ParticleSubGroupSharedPtr particle_sub_group;
  static constexpr char position_labels[][3] = {"x", "y", "z"};

  bool multi_dim_mode = false;

  inline size_t get_max_particle_size() {
    int max_ncomp = 0;
    for (auto &sym : this->sym_store.syms_real) {
      max_ncomp = MAX(max_ncomp, (*this->particle_group)[sym]->ncomp);
    }
    for (auto &sym : this->sym_store.syms_int) {
      max_ncomp = MAX(max_ncomp, (*this->particle_group)[sym]->ncomp);
    }
    max_ncomp = MAX(max_ncomp, this->particle_group->position_dat->ncomp);
    size_t size_el = MAX(sizeof(double), sizeof(long long));
    return static_cast<size_t>(max_ncomp) * size_el;
  };

  // Copy particle data from source to destination whilst casting to a suitable
  // HDF5 type.
  inline void memcpy_dat(char *dst, REAL *src, const INT npart) {
    double *dst_real = (REAL *)dst;
    for (INT px = 0; px < npart; px++) {
      dst_real[px] = src[px];
    }
  }
  inline void memcpy_dat(char *dst, INT *src, const INT npart) {
    long long *dst_ll = (long long *)dst;
    for (INT px = 0; px < npart; px++) {
      dst_ll[px] = src[px];
    }
  };

  // Get the HDF5 type that matches the datatype the particle data was cast to
  // when linearised on the host
  static inline hid_t memtypeid(ParticleDatSharedPtr<REAL>) {
    return H5T_NATIVE_DOUBLE;
  }
  static inline hid_t memtypeid(ParticleDatSharedPtr<INT>) {
    return H5T_NATIVE_LLONG;
  }

  /**
   *  Write a ParticleDat to the HDF5 file where each component has its own
   *  entry (preferred by Paraview).
   */
  template <typename GROUP_TYPE, typename T>
  static inline void
  write_dat_column_wise(std::shared_ptr<GROUP_TYPE> parent,
                        const int64_t npart_local, ParticleDatSharedPtr<T> dat,
                        hid_t dxpl, hid_t group_step, hid_t memspace,
                        hid_t filespace, bool is_position) {

    auto sycl_target = get_particle_group(parent)->sycl_target;
    const int ncomp = dat->ncomp;
    NESOASSERT(npart_local == parent->get_npart_local(),
               "Missmatch in npart local.");

    auto dh_buffer = get_resource<BufferDeviceHost<T>,
                                  ResourceStackInterfaceBufferDeviceHost<T>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDeviceHost<T>{},
        sycl_target);
    dh_buffer->realloc_no_copy(ncomp * npart_local);

    auto k_ptr = dh_buffer->d_buffer.ptr;

    particle_loop(
        "H5Part::write_dat_column_wise", parent,
        [=](auto INDEX, auto DAT) {
          const auto dst_index = INDEX.get_loop_linear_index();
          for (int cx = 0; cx < ncomp; cx++) {
            k_ptr[cx * npart_local + dst_index] = DAT.at(cx);
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(dat->sym))
        ->execute();

    dh_buffer->device_to_host();

    // write each component as a separate array in the HDF5 file.
    for (int cx = 0; cx < ncomp; cx++) {

      std::string name;
      if (is_position) {
        name = position_labels[cx];
      } else {
        name = dat->name + "_" + std::to_string(cx);
      }

      // hid_t dset = H5Dcreate1(group_step, name.c_str(), memtypeid(dat),
      //                         filespace, H5P_DEFAULT);

      hid_t dset = H5Dcreate2(group_step, name.c_str(), memtypeid(dat),
                              filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5CHK(H5Dwrite(dset, memtypeid(dat), memspace, filespace, dxpl,
                     dh_buffer->h_buffer.ptr + npart_local * cx));

      H5CHK(H5Dclose(dset));
    }

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDeviceHost<T>{}, dh_buffer);
  }

  /**
   *  Write a ParticleDat to the HDF5 file where each dat is stored as a 2D
   *  array.
   */
  template <typename GROUP_TYPE, typename T>
  inline void write_dat_2d(std::shared_ptr<GROUP_TYPE> parent,
                           const int64_t npart_total, const int64_t npart_local,
                           const int64_t offset, ParticleDatSharedPtr<T> dat,
                           hid_t group_step, hid_t dxpl) {

    auto sycl_target = get_particle_group(parent)->sycl_target;
    const int ncomp = dat->ncomp;
    NESOASSERT(npart_local == parent->get_npart_local(),
               "Missmatch in npart local.");

    auto dh_buffer = get_resource<BufferDeviceHost<T>,
                                  ResourceStackInterfaceBufferDeviceHost<T>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDeviceHost<T>{},
        sycl_target);
    dh_buffer->realloc_no_copy(ncomp * npart_local);

    auto k_ptr = dh_buffer->d_buffer.ptr;

    particle_loop(
        "H5Part::write_dat_column_wise", parent,
        [=](auto INDEX, auto DAT) {
          const auto dst_index = INDEX.get_loop_linear_index();
          for (int cx = 0; cx < ncomp; cx++) {
            k_ptr[cx * npart_local + dst_index] = DAT.at(cx);
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(dat->sym))
        ->execute();

    dh_buffer->device_to_host();

    // create the memspace and filespace for the 2D array
    hsize_t dims_memspace[2] = {static_cast<hsize_t>(npart_local),
                                static_cast<hsize_t>(ncomp)};
    hid_t memspace = H5Screate_simple(2, dims_memspace, NULL);
    hsize_t dims_filespace[2] = {static_cast<hsize_t>(npart_total),
                                 static_cast<hsize_t>(ncomp)};
    hid_t filespace = H5Screate_simple(2, dims_filespace, NULL);
    hsize_t slab_offsets[2] = {static_cast<hsize_t>(offset),
                               static_cast<hsize_t>(0)};
    hsize_t slab_counts[2] = {static_cast<hsize_t>(npart_local),
                              static_cast<hsize_t>(ncomp)};

    // select the relevant part of the hyperslab for this rank
    std::string name = dat->name;
    H5CHK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, slab_offsets, NULL,
                              slab_counts, NULL));

    // hid_t dset = H5Dcreate1(group_step, name.c_str(), memtypeid(dat),
    // filespace,
    //                         H5P_DEFAULT);
    hid_t dset = H5Dcreate2(group_step, name.c_str(), memtypeid(dat), filespace,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write the 2D array
    H5CHK(H5Dwrite(dset, memtypeid(dat), memspace, filespace, dxpl,
                   dh_buffer->h_buffer.ptr));

    H5CHK(H5Dclose(dset));

    H5CHK(H5Sclose(filespace));
    H5CHK(H5Sclose(memspace));

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDeviceHost<T>{}, dh_buffer);
  }

  /**
   *  Check the ParticleDat the user passed actually exists at the time of
   *  writing.
   */
  template <typename T>
  inline void check_dat_exists_write(ParticleGroupSharedPtr particle_group,
                                     T sym) {
    if (!particle_group->contains_dat(sym)) {
      std::string msg = "H5Part::write called with Sym '" + sym.name +
                        "' which does not exist in the ParticleGroup (check "
                        "datatype/spelling).";
      NESOASSERT(false, msg.c_str());
    }
  }

  /**
   * Open the file if it is closed.
   */
  inline void open_read_write() {
    if (this->is_closed) {
      this->plist_id = H5Pcreate(H5P_FILE_ACCESS);
      H5CHK(H5Pset_fapl_mpio(this->plist_id, this->comm_pair.comm_parent,
                             MPI_INFO_NULL));
      H5CHK(this->file_id =
                H5Fopen(this->filename.c_str(), H5F_ACC_RDWR, this->plist_id));
      this->is_closed = false;
    }
  }

  /**
   *  Write ParticleDats as 2D arrays in the HDF5 file.
   *  TODO This needs better testing if anyone wants it and for it to be made
   * public.
   */
  inline void enable_multi_dim_mode() { this->multi_dim_mode = true; }

  template <typename GROUP_TYPE>
  inline void write_inner(std::shared_ptr<GROUP_TYPE> group) {
    auto particle_group = get_particle_group(group);
    auto sycl_target = particle_group->sycl_target;

    // check the ParticleDats actually exist
    for (auto &sym : this->sym_store.syms_real) {
      this->check_dat_exists_write(particle_group, sym);
    }
    for (auto &sym : this->sym_store.syms_int) {
      this->check_dat_exists_write(particle_group, sym);
    }

    // Create the group for this time step.
    std::string step_name = "Step#";
    step_name += std::to_string(this->step);
    hid_t group_step = H5Gcreate(file_id, step_name.c_str(), H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);

    // Perform the bookkeeping logic once for all ParticleDats
    const int64_t npart_local = group->get_npart_local();
    int64_t offset;
    MPICHK(MPI_Scan(&npart_local, &offset, 1, MPI_INT64_T, MPI_SUM,
                    this->comm_pair.comm_parent));

    int64_t npart_total = offset;
    MPICHK(MPI_Bcast(&npart_total, 1, MPI_INT64_T,
                     this->comm_pair.size_parent - 1,
                     this->comm_pair.comm_parent));
    offset -= npart_local;

    if (npart_total > 0) {

      // Create a memspace and filespace for the columnwise writes.
      hsize_t dims_memspace[1] = {static_cast<hsize_t>(npart_local)};
      hid_t memspace = H5Screate_simple(1, dims_memspace, NULL);

      hsize_t dims_filespace[1] = {static_cast<hsize_t>(npart_total)};
      hid_t filespace = H5Screate_simple(1, dims_filespace, NULL);

      hsize_t slab_offsets[1] = {static_cast<hsize_t>(offset)};
      hsize_t slab_counts[1] = {static_cast<hsize_t>(npart_local)};

      // select the hyperslab for the columnwite writes.
      H5CHK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, slab_offsets, NULL,
                                slab_counts, NULL));
      hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
      H5CHK(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));

      // Write the positions explicitly in a way Paraview interprets as a H5Part
      // format.
      write_dat_column_wise(group, npart_local, particle_group->position_dat,
                            dxpl, group_step, memspace, filespace, true);

      // Write the particle data for each ParticleDat
      if (multi_dim_mode) {
        for (auto &sym : this->sym_store.syms_real) {
          const auto dat = particle_group->get_dat(sym, false);
          this->write_dat_2d(group, npart_total, npart_local, offset, dat,
                             group_step, dxpl);
        }
        for (auto &sym : this->sym_store.syms_int) {
          const auto dat = particle_group->get_dat(sym, false);
          this->write_dat_2d(group, npart_total, npart_local, offset, dat,
                             group_step, dxpl);
        }
      } else {

        for (auto &sym : this->sym_store.syms_real) {
          const auto dat = particle_group->get_dat(sym, false);
          write_dat_column_wise(group, npart_local, dat, dxpl, group_step,
                                memspace, filespace, false);
        }
        for (auto &sym : this->sym_store.syms_int) {
          const auto dat = particle_group->get_dat(sym, false);
          write_dat_column_wise(group, npart_local, dat, dxpl, group_step,
                                memspace, filespace, false);
        }
      }

      H5CHK(H5Pclose(dxpl));
      H5CHK(H5Sclose(filespace));
      H5CHK(H5Sclose(memspace));
    }
    H5CHK(H5Gclose(group_step));
  }

public:
  /// Disable (implicit) copies.
  H5Part(const H5Part &st) = delete;
  /// Disable (implicit) copies.
  H5Part &operator=(H5Part const &a) = delete;

  ~H5Part() {
    NESOASSERT(this->is_closed, "H5Part file was not closed correctly.");
  };

  /**
   *  Construct a H5Part writer for a given set of ParticleDats described by
   *  Sym<type>(name) instances. Must be called collectively on the
   * communicator.
   *
   *  @param filename Output filename, e.g. "foo.h5part".
   *  @param particle_group ParticleGroupSharedPtr instance.
   *  @param args Remaining arguments (variable length) should be sym instances
   *  indicating which ParticleDats are to be written.
   */
  template <typename... T>
  H5Part(std::string filename, ParticleGroupSharedPtr particle_group,
         T &&...args)
      : filename(filename), comm_pair(particle_group->sycl_target->comm_pair),
        sym_store(std::forward<T>(args)...), particle_group(particle_group),
        particle_sub_group(nullptr), multi_dim_mode(false) {
    this->plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5CHK(H5Pset_fapl_mpio(this->plist_id, this->comm_pair.comm_parent,
                           MPI_INFO_NULL));
    this->file_id = H5Fcreate(this->filename.c_str(), H5F_ACC_TRUNC,
                              H5P_DEFAULT, this->plist_id);
    this->is_closed = false;
    this->step = 0;
  }

  /**
   *  Construct a H5Part writer for a given set of ParticleDats described by
   *  Sym<type>(name) instances. Must be called collectively on the
   * communicator.
   *
   *  @param filename Output filename, e.g. "foo.h5part".
   *  @param particle_sub_group ParticleSubGroupSharedPtr instance.
   *  @param args Remaining arguments (variable length) should be sym instances
   *  indicating which ParticleDats are to be written.
   */
  template <typename... T>
  H5Part(std::string filename, ParticleSubGroupSharedPtr particle_sub_group,
         T &&...args)
      : H5Part(filename, get_particle_group(particle_sub_group), args...) {
    this->particle_sub_group = particle_sub_group;
  }

  /**
   *  Close the H5Part writer. Must be called before execution completes. Must
   *  be called collectively on the communicator. Can optionally be called
   *  after calling write to close the file such that if the simulation errors
   *  the particle trajectory is readable.
   */
  inline void close() {
    if (!this->is_closed) {
      H5CHK(H5Fclose(this->file_id));
      H5CHK(H5Pclose(this->plist_id));
    }
    this->is_closed = true;
  };

  /**
   * Write the current particle data to the HDF5 file as a new time step. Must
   * be called collectively on the communicator. Will open the file if required.
   *
   * @param step_in Optionally set the step explicitly.
   */
  inline void write(INT step_in = -1) {
    // open the file for writing if required.
    this->open_read_write();

    if (step_in >= 0) {
      this->step = step_in;
    }

    if (this->particle_sub_group != nullptr) {
      this->write_inner(this->particle_sub_group);
    } else {
      this->write_inner(this->particle_group);
    }

    this->step++;
  };
};

#else

class H5Part {
private:
  /**
   *  Write ParticleDats as 2D arrays in the HDF5 file.
   */
  inline void enable_multi_dim_mode() {}

public:
  /// Disable (implicit) copies.
  H5Part(const H5Part &st) = delete;
  /// Disable (implicit) copies.
  H5Part &operator=(H5Part const &a) = delete;

  /**
   *  Construct a H5Part writer for a given set of ParticleDats described by
   *  Sym<type>(name) instances. Must be called collectively on the
   * communicator.
   *
   *  @param filename Output filename, e.g. "foo.h5part".
   *  @param particle_group ParticleGroupSharedPtr instance.
   *  @param args Remaining arguments (variable length) should be sym instances
   *  indicating which ParticleDats are to be written.
   */
  template <typename... T>
  H5Part(std::string filename, ParticleGroupSharedPtr particle_group,
         T... args){};

  /**
   *  Construct a H5Part writer for a given set of ParticleDats described by
   *  Sym<type>(name) instances. Must be called collectively on the
   * communicator.
   *
   *  @param filename Output filename, e.g. "foo.h5part".
   *  @param particle_sub_group ParticleSubGroupSharedPtr instance.
   *  @param args Remaining arguments (variable length) should be sym instances
   *  indicating which ParticleDats are to be written.
   */
  template <typename... T>
  H5Part(std::string filename, ParticleSubGroupSharedPtr particle_sub_group,
         T... args){};

  /**
   *  Close the H5Part writer. Must be called. Must be called collectively on
   *  the communicator.
   */
  inline void close() {};

  /**
   * Write the current particle data to the HDF5 file as a new time step. Must
   * be called collectively on the communicator.
   */
  inline void write(INT step_in = -1) {};
};

#endif
} // namespace NESO::Particles

#endif
