#ifndef _NESO_PARTICLES_PACKING_UNPACKING
#define _NESO_PARTICLES_PACKING_UNPACKING

#include <CL/sycl.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include "cell_dat.hpp"
#include "compute_target.hpp"
#include "domain.hpp"
#include "particle_dat.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

class ParticlePacker {
private:
  CellDat<char> cell_dat;
  BufferShared<int> s_npart_packed;

  int num_dats_real = 0;
  int num_dats_int = 0;
  BufferShared<REAL ***> s_particle_dat_ptr_real;
  BufferShared<INT ***> s_particle_dat_ptr_int;
  BufferShared<int> s_particle_dat_ncomp_real;
  BufferShared<int> s_particle_dat_ncomp_int;

  inline size_t
  particle_size(std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
                std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int) {
    size_t s = 0;
    for (auto &dat : particle_dats_real) {
      s += dat.second->cell_dat.row_size();
    }
    for (auto &dat : particle_dats_int) {
      s += dat.second->cell_dat.row_size();
    }
    return s;
  };

  inline void get_particle_dat_info(
      std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
      std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int) {

    num_dats_real = particle_dats_real.size();
    s_particle_dat_ptr_real.realloc_no_copy(num_dats_real);
    s_particle_dat_ncomp_real.realloc_no_copy(num_dats_real);

    num_dats_int = particle_dats_real.size();
    s_particle_dat_ptr_int.realloc_no_copy(num_dats_int);
    s_particle_dat_ncomp_int.realloc_no_copy(num_dats_int);

    int index = 0;
    for (auto &dat : particle_dats_real) {
      s_particle_dat_ptr_real.ptr[index] = dat.second->cell_dat.device_ptr();
      s_particle_dat_ncomp_real.ptr[index] = dat.second->ncomp;
      index++;
    }
    index = 0;
    for (auto &dat : particle_dats_int) {
      s_particle_dat_ptr_int.ptr[index] = dat.second->cell_dat.device_ptr();
      s_particle_dat_ncomp_int.ptr[index] = dat.second->ncomp;
      index++;
    }
  }

public:
  SYCLTarget &sycl_target;
  ~ParticlePacker(){};
  ParticlePacker(SYCLTarget &sycl_target)
      : sycl_target(sycl_target),
        cell_dat(sycl_target, sycl_target.comm_pair.size_parent, 1),
        s_npart_packed(sycl_target, sycl_target.comm_pair.size_parent),
        s_particle_dat_ptr_real(sycl_target, 1),
        s_particle_dat_ptr_int(sycl_target, 1),
        s_particle_dat_ncomp_real(sycl_target, 1),
        s_particle_dat_ncomp_int(sycl_target, 1){};

  inline void reset() {
    const int size_parent = sycl_target.comm_pair.size_parent;
    auto s_npart_packed_ptr = s_npart_packed.ptr;
    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(static_cast<size_t>(size_parent)),
              [=](sycl::id<1> idx) { s_npart_packed_ptr[idx] = 0; });
        })
        .wait_and_throw();
  }

  inline void
  pack(const int num_remote_send_ranks, const int *s_send_ranks_ptr,
       const int *s_send_rank_npart_ptr, const int num_particles_leaving,
       const int *s_pack_cells_ptr, const int *s_pack_layers_src_ptr,
       const int *s_pack_layers_dst_ptr,
       std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
       std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int) {

    // Allocate enough space to store the particles to pack
    const int num_bytes_per_particle =
        particle_size(particle_dats_real, particle_dats_int);
    for (int rankx = 0; rankx < num_remote_send_ranks; rankx++) {
      const int npart = s_send_rank_npart_ptr[rankx];
      this->cell_dat.set_nrow(rankx, (npart + s_npart_packed.ptr[rankx]) *
                                         num_bytes_per_particle);
    }

    // get the pointers to the particle dat data and the number of components in
    // each dat
    get_particle_dat_info(particle_dats_real, particle_dats_int);

    // loop over the particles to pack and for each particle pack the data into
    // the CellDat where each cell is the data to send to a remote rank.

    const int k_num_dats_real = this->num_dats_real;
    const int k_num_dats_int = this->num_dats_int;

    const auto k_particle_dat_ptr_real = s_particle_dat_ptr_real.ptr;
    const auto k_particle_dat_ptr_int = s_particle_dat_ptr_int.ptr;
    const auto k_particle_dat_ncomp_real = s_particle_dat_ncomp_real.ptr;
    const auto k_particle_dat_ncomp_int = s_particle_dat_ncomp_int.ptr;
    const auto k_particle_dat_rank =
        particle_dats_int[Sym<INT>("NESO_MPI_RANK")]->cell_dat.device_ptr();

    auto k_pack_cell_dat = this->cell_dat.device_ptr();

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              // for each leaving particle
              sycl::range<1>(static_cast<size_t>(num_particles_leaving)),
              [=](sycl::id<1> idx) {
                const int cell = s_pack_cells_ptr[idx];
                const int layer_src = s_pack_layers_src_ptr[idx];
                const int layer_dst = s_pack_layers_src_ptr[idx];
                const int rank = k_particle_dat_rank[cell][0][layer_src];

                char *base_pack_ptr =
                    &k_pack_cell_dat[rank][0]
                                    [layer_dst * num_bytes_per_particle];
                REAL *pack_ptr_real = (REAL *)base_pack_ptr;
                // for each real dat
                int index = 0;
                for (int dx = 0; dx < k_num_dats_real; dx++) {
                  REAL ***dat_ptr = k_particle_dat_ptr_real[dx];
                  const int ncomp = k_particle_dat_ncomp_real[dx];
                  // for each component
                  for (int cx = 0; cx < ncomp; cx++) {
                    pack_ptr_real[index + cx] = dat_ptr[cell][cx][layer_src];
                  }
                  index += ncomp;
                }
                // for each int dat
                INT *pack_ptr_int = (INT *)(pack_ptr_real + index);
                index = 0;
                for (int dx = 0; dx < k_num_dats_int; dx++) {
                  INT ***dat_ptr = k_particle_dat_ptr_int[dx];
                  const int ncomp = k_particle_dat_ncomp_int[dx];
                  // for each component
                  for (int cx = 0; cx < ncomp; cx++) {
                    pack_ptr_int[index + cx] = dat_ptr[cell][cx][layer_src];
                  }
                  index += ncomp;
                }
              });
        })
        .wait_and_throw();
  };
};

} // namespace NESO::Particles

#endif
