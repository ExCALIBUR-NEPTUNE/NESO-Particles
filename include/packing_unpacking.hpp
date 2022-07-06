#ifndef _NESO_PARTICLES_PACKING_UNPACKING
#define _NESO_PARTICLES_PACKING_UNPACKING

#include <CL/sycl.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <stack>
#include <string>

#include "compute_target.hpp"
#include "particle_dat.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

class ParticlePacker {
private:
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
  int num_bytes_per_particle;
  CellDat<char> cell_dat;

  BufferHost<char> h_send_buffer;
  BufferHost<INT> h_send_offsets;
  INT required_send_buffer_length;

  SYCLTarget &sycl_target;
  ~ParticlePacker(){};
  ParticlePacker(SYCLTarget &sycl_target)
      : sycl_target(sycl_target),
        cell_dat(sycl_target, sycl_target.comm_pair.size_parent, 1),
        s_npart_packed(sycl_target, sycl_target.comm_pair.size_parent),
        h_send_buffer(sycl_target, 1), h_send_offsets(sycl_target, 1),
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

  inline char *get_packed_data_on_host(const int num_remote_send_ranks,
                                       const int *h_send_rank_npart_ptr) {
    this->h_send_buffer.realloc_no_copy(this->required_send_buffer_length);
    this->h_send_offsets.realloc_no_copy(num_remote_send_ranks);
    NESOASSERT((this->cell_dat.ncells) >=
                   (this->sycl_target.comm_pair.size_parent),
               "Insuffient cells");

    INT offset = 0;
    std::stack<sycl::event> copy_events{};
    for (int rankx = 0; rankx < num_remote_send_ranks; rankx++) {
      const int npart_tmp = h_send_rank_npart_ptr[rankx];
      const int nbytes_tmp = npart_tmp * this->num_bytes_per_particle;
      auto device_ptr = this->cell_dat.col_device_ptr(rankx, 0);
      copy_events.push(this->sycl_target.queue.memcpy(
          &this->h_send_buffer.ptr[offset], device_ptr, nbytes_tmp));
      this->h_send_offsets.ptr[rankx] = offset;
      offset += nbytes_tmp;
    }

    while (!copy_events.empty()) {
      auto event = copy_events.top();
      event.wait_and_throw();
      copy_events.pop();
    }

    return this->h_send_buffer.ptr;
  }

  inline sycl::event
  pack(const int num_remote_send_ranks, const int *s_send_ranks_ptr,
       const int *s_send_rank_npart_ptr, const int *s_send_rank_map_ptr,
       const int num_particles_leaving, const int *s_pack_cells_ptr,
       const int *s_pack_layers_src_ptr, const int *s_pack_layers_dst_ptr,
       std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
       std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int) {

    // Allocate enough space to store the particles to pack
    this->num_bytes_per_particle =
        particle_size(particle_dats_real, particle_dats_int);

    this->required_send_buffer_length = 0;
    for (int rankx = 0; rankx < num_remote_send_ranks; rankx++) {
      const int npart = s_send_rank_npart_ptr[rankx];
      const INT rankx_contrib =
          (npart + s_npart_packed.ptr[rankx]) * this->num_bytes_per_particle;
      this->cell_dat.set_nrow(rankx, rankx_contrib);
      this->required_send_buffer_length += rankx_contrib;
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

    const int k_num_bytes_per_particle = this->num_bytes_per_particle;

    auto k_pack_cell_dat = this->cell_dat.device_ptr();
    sycl::event event = this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(
          // for each leaving particle
          sycl::range<1>(static_cast<size_t>(num_particles_leaving)),
          [=](sycl::id<1> idx) {
            const int cell = s_pack_cells_ptr[idx];
            const int layer_src = s_pack_layers_src_ptr[idx];
            const int layer_dst = s_pack_layers_dst_ptr[idx];
            const int rank = k_particle_dat_rank[cell][0][layer_src];
            const int rank_packing_cell = s_send_rank_map_ptr[rank];

            char *base_pack_ptr =
                &k_pack_cell_dat[rank_packing_cell][0]
                                [layer_dst * k_num_bytes_per_particle];
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
    });

    return event;
  };
};

class ParticleUnpacker {
private:
  BufferShared<int> s_npart_packed;

  int num_dats_real = 0;
  int num_dats_int = 0;
  BufferShared<REAL ***> s_particle_dat_ptr_real;
  BufferShared<INT ***> s_particle_dat_ptr_int;
  BufferShared<int> s_particle_dat_ncomp_real;
  BufferShared<int> s_particle_dat_ncomp_int;

  BufferHost<char> d_recv_buffer;

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
    this->num_bytes_per_particle = s;
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
  BufferHost<char> h_recv_buffer;
  BufferHost<INT> h_recv_offsets;
  int npart_recv;
  int num_bytes_per_particle;

  SYCLTarget &sycl_target;
  ~ParticleUnpacker(){};
  ParticleUnpacker(SYCLTarget &sycl_target)
      : sycl_target(sycl_target), h_recv_buffer(sycl_target, 1),
        h_recv_offsets(sycl_target, 1), d_recv_buffer(sycl_target, 1),
        s_npart_packed(sycl_target, sycl_target.comm_pair.size_parent),
        s_particle_dat_ptr_real(sycl_target, 1),
        s_particle_dat_ptr_int(sycl_target, 1),
        s_particle_dat_ncomp_real(sycl_target, 1),
        s_particle_dat_ncomp_int(sycl_target, 1){};

  inline void
  reset(const int num_remote_recv_ranks, BufferHost<int> &h_recv_rank_npart,
        std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
        std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int) {

    // realloc the array that holds where in the recv buffer the data from each
    // remote rank should be placed
    this->h_recv_offsets.realloc_no_copy(num_remote_recv_ranks);
    this->num_bytes_per_particle =
        this->particle_size(particle_dats_real, particle_dats_int);

    // compute the offsets in the recv buffer
    this->npart_recv = 0;
    for (int rankx = 0; rankx < num_remote_recv_ranks; rankx++) {
      this->h_recv_offsets.ptr[rankx] =
          this->npart_recv * this->num_bytes_per_particle;
      this->npart_recv += h_recv_rank_npart.ptr[rankx];
    }

    // realloc the recv buffer
    this->h_recv_buffer.realloc_no_copy(this->npart_recv *
                                        this->num_bytes_per_particle);
    this->d_recv_buffer.realloc_no_copy(this->npart_recv *
                                        this->num_bytes_per_particle);
  }

  /*
   * Unpack the recv buffer into the particle group. Particles unpack into cell
   * 0.
   */
  inline void
  unpack(std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
         std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int) {

    // copy packed data to device
    auto event_memcpy = this->sycl_target.queue.memcpy(
        this->d_recv_buffer.ptr, this->h_recv_buffer.ptr,
        this->npart_recv * this->num_bytes_per_particle);

    // old cell occupancy
    auto mpi_rank_dat = particle_dats_int[Sym<INT>("NESO_MPI_RANK")];
    const int npart_cell_0_old = mpi_rank_dat->s_npart_cell[0];
    const int npart_cell_0_new = npart_cell_0_old + this->npart_recv;
    // realloc cell 0 on the dats
    for (auto &dat : particle_dats_real) {
      dat.second->realloc(0, npart_cell_0_new);
      dat.second->set_npart_cell(0, npart_cell_0_new);
    }
    for (auto &dat : particle_dats_int) {
      dat.second->realloc(0, npart_cell_0_new);
      dat.second->set_npart_cell(0, npart_cell_0_new);
    }

    const int k_npart_recv = this->npart_recv;

    // get the pointers to the particle dat data and the number of components in
    // each dat
    get_particle_dat_info(particle_dats_real, particle_dats_int);

    // unpack into cell 0
    const int k_num_bytes_per_particle = this->num_bytes_per_particle;
    const int k_num_dats_real = this->num_dats_real;
    const int k_num_dats_int = this->num_dats_int;
    const auto k_particle_dat_ptr_real = s_particle_dat_ptr_real.ptr;
    const auto k_particle_dat_ptr_int = s_particle_dat_ptr_int.ptr;
    const auto k_particle_dat_ncomp_real = s_particle_dat_ncomp_real.ptr;
    const auto k_particle_dat_ncomp_int = s_particle_dat_ncomp_int.ptr;
    char *k_recv_buffer = this->d_recv_buffer.ptr;

    event_memcpy.wait_and_throw();
    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              // for each new particle
              sycl::range<1>(static_cast<size_t>(k_npart_recv)),
              [=](sycl::id<1> idx) {
                const int cell = 0;
                // destination layer in the cell
                const int layer = npart_cell_0_old + idx;
                // source position in the packed buffer
                const int offset = k_num_bytes_per_particle * idx;
                char *unpack_base_ptr = k_recv_buffer + offset;
                REAL *unpack_ptr_real = (REAL *)unpack_base_ptr;
                // for each real dat
                int index = 0;
                for (int dx = 0; dx < k_num_dats_real; dx++) {
                  REAL ***dat_ptr = k_particle_dat_ptr_real[dx];
                  const int ncomp = k_particle_dat_ncomp_real[dx];
                  // for each component
                  for (int cx = 0; cx < ncomp; cx++) {
                    dat_ptr[cell][cx][layer] = unpack_ptr_real[index + cx];
                  }
                  index += ncomp;
                }
                // for each int dat
                INT *unpack_ptr_int = (INT *)(unpack_ptr_real + index);
                index = 0;
                for (int dx = 0; dx < k_num_dats_int; dx++) {
                  INT ***dat_ptr = k_particle_dat_ptr_int[dx];
                  const int ncomp = k_particle_dat_ncomp_int[dx];
                  // for each component
                  for (int cx = 0; cx < ncomp; cx++) {
                    dat_ptr[cell][cx][layer] = unpack_ptr_int[index + cx];
                  }
                  index += ncomp;
                }
              });
        })
        .wait_and_throw();
  }
};

} // namespace NESO::Particles

#endif
