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

  inline hid_t memtypeid(ParticleDatShPtr<REAL>) { return H5T_NATIVE_DOUBLE; }

  inline hid_t memtypeid(ParticleDatShPtr<INT>) { return H5T_NATIVE_LLONG; }

public:
  ~H5Part() {
    NESOASSERT(this->is_closed, "H5Part file was not closed correctly.");
  };

  template <typename... T>
  H5Part(std::string filename, ParticleGroup &particle_group, T... args)
      : filename(filename), particle_group(particle_group),
        comm_pair(particle_group.sycl_target.comm_pair), sym_store(args...) {
    this->plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5CHK(H5Pset_fapl_mpio(this->plist_id, MPI_COMM_WORLD, MPI_INFO_NULL));
    this->file_id = H5Fcreate(this->filename.c_str(), H5F_ACC_TRUNC,
                              H5P_DEFAULT, this->plist_id);
    this->is_closed = false;
    this->step = 0;
  };

  inline void close() {
    H5CHK(H5Fclose(this->file_id));
    H5CHK(H5Pclose(this->plist_id));
    this->is_closed = true;
  };

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

  template <typename T>
  inline void write_dat(const int64_t npart_local,
                        BufferHost<char> &pack_buffer, ParticleDatShPtr<T> dat,
                        hid_t dxpl, hid_t group_step, hid_t memspace,
                        hid_t filespace, bool is_position) {
    const int ncomp = dat->ncomp;
    const int cell_count = dat->ncell;
    const size_t component_stride =
        static_cast<size_t>(npart_local) * sizeof(T);

    size_t pack_offset = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto cell_dat = dat->cell_dat.get_cell(cellx);

      for (int cx = 0; cx < ncomp; cx++) {
        this->memcpy_dat(pack_buffer.ptr + cx * component_stride + pack_offset,
                         cell_dat->data[cx].data(), cell_dat->nrow);
      }
      pack_offset += sizeof(T) * cell_dat->nrow;
    }

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

  inline void write(INT step_in = -1) {
    if (step_in >= 0) {
      this->step = step_in;
    }

    std::string step_name = "Step#";
    step_name += std::to_string(this->step);
    hid_t group_step = H5Gcreate(file_id, step_name.c_str(), H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);

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

    hsize_t dims_memspace[1] = {static_cast<hsize_t>(npart_local)};
    hid_t memspace = H5Screate_simple(1, dims_memspace, NULL);

    hsize_t dims_filespace[1] = {static_cast<hsize_t>(npart_total)};
    hid_t filespace = H5Screate_simple(1, dims_filespace, NULL);

    hsize_t slab_offsets[1] = {static_cast<hsize_t>(offset)};
    hsize_t slab_counts[1] = {static_cast<hsize_t>(npart_local)};

    H5CHK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, slab_offsets, NULL,
                              slab_counts, NULL));
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5CHK(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));

    BufferHost<char> pack_buffer(this->particle_group.sycl_target,
                                 npart_local * get_max_particle_size());

    write_dat(npart_local, pack_buffer, this->particle_group.position_dat, dxpl,
              group_step, memspace, filespace, true);

    for (auto &sym : this->sym_store.syms_real) {
      const auto dat = this->particle_group[sym];
      write_dat(npart_local, pack_buffer, dat, dxpl, group_step, memspace,
                filespace, false);
    }
    for (auto &sym : this->sym_store.syms_int) {
      const auto dat = this->particle_group[sym];
      write_dat(npart_local, pack_buffer, dat, dxpl, group_step, memspace,
                filespace, false);
    }

    H5CHK(H5Pclose(dxpl));
    H5CHK(H5Sclose(filespace));
    H5CHK(H5Sclose(memspace));
    H5CHK(H5Gclose(group_step));
    this->step++;
  };
};

#endif

} // namespace NESO::Particles
#endif
