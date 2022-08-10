#ifndef _NESO_PARTICLES_PARTICLE_IO
#define _NESO_PARTICLES_PARTICLE_IO

#include "particle_group.hpp"
#include "particle_spec.hpp"
#include "typedefs.hpp"
#include <cstdint>
#include <cstring>
#include <mpi.h>
#include <string>

namespace NESO::Particles {
#ifdef NESO_PARTICLES_HDF5
#include <hdf5.h>

#define H5CHK(cmd) NESOASSERT((cmd) >= 0, "HDF5 ERROR");

class H5Part {

private:
  std::string filename;
  CommPair &comm_pair;
  SymStore sym_store;
  bool is_closed = true;
  hid_t plist_id, file_id;
  INT step;
  ParticleGroup &particle_group;
  std::string position_labels[3] = {"x", "y", "z"};
  bool multi_dim_mode = false;

  inline size_t get_max_particle_size() {
    int max_ncomp = 0;
    for (auto &sym : this->sym_store.syms_real) {
      max_ncomp = MAX(max_ncomp, this->particle_group[sym]->ncomp);
    }
    for (auto &sym : this->sym_store.syms_int) {
      max_ncomp = MAX(max_ncomp, this->particle_group[sym]->ncomp);
    }
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
  inline hid_t memtypeid(ParticleDatShPtr<REAL>) { return H5T_NATIVE_DOUBLE; }
  inline hid_t memtypeid(ParticleDatShPtr<INT>) { return H5T_NATIVE_LLONG; }

  /**
   *  Write a ParticleDat to the HDF5 file where each component has its own
   *  entry (preferred by Paraview).
   */
  template <typename T>
  inline void write_dat_column_wise(const int64_t npart_local,
                                    BufferHost<char> &pack_buffer,
                                    ParticleDatShPtr<T> dat, hid_t dxpl,
                                    hid_t group_step, hid_t memspace,
                                    hid_t filespace, bool is_position) {
    const int ncomp = dat->ncomp;
    const int cell_count = dat->ncell;
    const size_t component_stride =
        static_cast<size_t>(npart_local) * sizeof(T);

    // Get the data on the host
    size_t pack_offset = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto cell_dat = dat->cell_dat.get_cell(cellx);

      for (int cx = 0; cx < ncomp; cx++) {
        this->memcpy_dat(pack_buffer.ptr + cx * component_stride + pack_offset,
                         cell_dat->data[cx].data(), cell_dat->nrow);
      }
      pack_offset += sizeof(T) * cell_dat->nrow;
    }

    // write each component as a separate array in the HDF5 file.
    for (int cx = 0; cx < ncomp; cx++) {

      std::string name;
      if (is_position) {
        name = this->position_labels[cx];
      } else {
        name = dat->name + "_" + std::to_string(cx);
      }

      hid_t dset = H5Dcreate1(group_step, name.c_str(), memtypeid(dat),
                              filespace, H5P_DEFAULT);
      H5CHK(H5Dwrite(dset, memtypeid(dat), memspace, filespace, dxpl,
                     pack_buffer.ptr + component_stride * cx));

      H5CHK(H5Dclose(dset));
    }
  }

  /**
   *  Write a ParticleDat to the HDF5 file where each dat is stored as a 2D
   *  array.
   */
  template <typename T>
  inline void write_dat_2d(const int64_t npart_total, const int64_t npart_local,
                           const int64_t offset, BufferHost<char> &pack_buffer,
                           ParticleDatShPtr<T> dat, hid_t group_step,
                           hid_t dxpl) {
    const int ncomp = dat->ncomp;
    const int cell_count = dat->ncell;

    // Linearise the data on the host.
    INT pack_offset = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto cell_dat = dat->cell_dat.get_cell(cellx);

      for (int rowx = 0; rowx < cell_dat->nrow; rowx++) {
        for (int colx = 0; colx < cell_dat->ncol; colx++) {
          this->memcpy_dat(pack_buffer.ptr +
                               (pack_offset + rowx * ncomp + colx) * sizeof(T),
                           cell_dat->data[colx].data() + rowx, 1);
        }
      }
      pack_offset += cell_dat->nrow * cell_dat->ncol;
    }

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

    hid_t dset = H5Dcreate1(group_step, name.c_str(), memtypeid(dat), filespace,
                            H5P_DEFAULT);

    // write the 2D array
    H5CHK(H5Dwrite(dset, memtypeid(dat), memspace, filespace, dxpl,
                   pack_buffer.ptr));

    H5CHK(H5Dclose(dset));

    H5CHK(H5Sclose(filespace));
    H5CHK(H5Sclose(memspace));
  }

public:
  ~H5Part() {
    NESOASSERT(this->is_closed, "H5Part file was not closed correctly.");
  };

  /**
   *  Construct a H5Part writer for a given set of ParticleDats described by
   *  Sym<type>(name) instances. Must be called collectively on the
   * communicator.
   *
   *  @param filename Output filename, e.g. "foo.h5part".
   *  @param particle_group ParticleGroup instance.
   *  @param args Remaining arguments (variable length) should be sym instances
   *  indicating which ParticleDats are to be written.
   */
  template <typename... T>
  H5Part(std::string filename, ParticleGroup &particle_group, T... args)
      : filename(filename), particle_group(particle_group),
        comm_pair(particle_group.sycl_target.comm_pair), sym_store(args...),
        multi_dim_mode(false) {
    this->plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5CHK(H5Pset_fapl_mpio(this->plist_id, MPI_COMM_WORLD, MPI_INFO_NULL));
    this->file_id = H5Fcreate(this->filename.c_str(), H5F_ACC_TRUNC,
                              H5P_DEFAULT, this->plist_id);
    this->is_closed = false;
    this->step = 0;
  };

  /**
   *  Close the H5Part writer. Must be called. Must be called collectively on
   *  the communicator.
   */
  inline void close() {
    H5CHK(H5Fclose(this->file_id));
    H5CHK(H5Pclose(this->plist_id));
    this->is_closed = true;
  };

  /**
   *  Write ParticleDats as 2D arrays in the HDF5 file.
   */
  inline void enable_multi_dim_mode() { this->multi_dim_mode = true; }

  /**
   * Write the current particle data to the HDF5 file as a new time step. Must
   * be called collectively on the communicator.
   */
  inline void write(INT step_in = -1) {
    if (step_in >= 0) {
      this->step = step_in;
    }

    // Create the group for this time step.
    std::string step_name = "Step#";
    step_name += std::to_string(this->step);
    hid_t group_step = H5Gcreate(file_id, step_name.c_str(), H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);

    // Perform the bookkeeping logic once for all ParticleDats
    const int64_t npart_local =
        this->particle_group.position_dat->get_npart_local();
    int64_t offset;
    MPICHK(MPI_Scan(&npart_local, &offset, 1, MPI_INT64_T, MPI_SUM,
                    this->comm_pair.comm_parent));

    int64_t npart_total = offset;
    MPICHK(MPI_Bcast(&npart_total, 1, MPI_INT64_T,
                     this->comm_pair.size_parent - 1,
                     this->comm_pair.comm_parent));
    offset -= npart_local;

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

    // create a host buffer in which to serialise the particle data before
    // passing to HDF5
    BufferHost<char> pack_buffer(this->particle_group.sycl_target,
                                 npart_local * get_max_particle_size());

    // Write the positions explicitly in a way Paraview interprets as a H5Part
    // format.
    write_dat_column_wise(npart_local, pack_buffer,
                          this->particle_group.position_dat, dxpl, group_step,
                          memspace, filespace, true);

    // Write the particle data for each ParticleDat
    if (multi_dim_mode) {
      for (auto &sym : this->sym_store.syms_real) {
        const auto dat = this->particle_group[sym];
        this->write_dat_2d(npart_total, npart_local, offset, pack_buffer, dat,
                           group_step, dxpl);
      }
      for (auto &sym : this->sym_store.syms_int) {
        const auto dat = this->particle_group[sym];
        this->write_dat_2d(npart_total, npart_local, offset, pack_buffer, dat,
                           group_step, dxpl);
      }
    } else {

      for (auto &sym : this->sym_store.syms_real) {
        const auto dat = this->particle_group[sym];
        write_dat_column_wise(npart_local, pack_buffer, dat, dxpl, group_step,
                              memspace, filespace, false);
      }
      for (auto &sym : this->sym_store.syms_int) {
        const auto dat = this->particle_group[sym];
        write_dat_column_wise(npart_local, pack_buffer, dat, dxpl, group_step,
                              memspace, filespace, false);
      }
    }

    H5CHK(H5Pclose(dxpl));
    H5CHK(H5Sclose(filespace));
    H5CHK(H5Sclose(memspace));

    H5CHK(H5Gclose(group_step));
    this->step++;
  };
};

#else

class H5Part {
private:
public:
  /**
   *  Construct a H5Part writer for a given set of ParticleDats described by
   *  Sym<type>(name) instances. Must be called collectively on the
   * communicator.
   *
   *  @param filename Output filename, e.g. "foo.h5part".
   *  @param particle_group ParticleGroup instance.
   *  @param args Remaining arguments (variable length) should be sym instances
   *  indicating which ParticleDats are to be written.
   */
  template <typename... T>
  H5Part(std::string filename, ParticleGroup &particle_group, T... args){};

  /**
   *  Close the H5Part writer. Must be called. Must be called collectively on
   *  the communicator.
   */
  inline void close(){};

  /**
   *  Write ParticleDats as 2D arrays in the HDF5 file.
   */
  inline void enable_multi_dim_mode() {}

  /**
   * Write the current particle data to the HDF5 file as a new time step. Must
   * be called collectively on the communicator.
   */
  inline void write(INT step_in = -1){};
};

#endif

} // namespace NESO::Particles
#endif
