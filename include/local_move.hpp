#ifndef _NESO_PARTICLES_LOCAL_MOVE
#define _NESO_PARTICLES_LOCAL_MOVE

#include <CL/sycl.hpp>
#include <mpi.h>
#include <set>
#include <vector>

#include "cell_dat_compression.hpp"
#include "communication.hpp"
#include "compute_target.hpp"
#include "packing_unpacking.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

class LocalMove {

private:
  std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real;
  std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int;
  ParticleDatShPtr<INT> mpi_rank_dat;

  ParticlePacker particle_packer;
  ParticleUnpacker particle_unpacker;

  // Reference to the layer compressor on the particle group such that this
  // global move can remove the sent particles
  LayerCompressor &layer_compressor;

  BufferHost<int> h_send_ranks;
  BufferHost<int> h_recv_ranks;
  BufferHost<MPI_Request> h_send_requests;
  BufferHost<MPI_Request> h_recv_requests;
  BufferHost<MPI_Status> h_status;

public:
  SYCLTarget &sycl_target;

  MPI_Comm comm;
  int num_remote_send_ranks;
  int num_remote_recv_ranks;

  ~LocalMove(){};
  LocalMove(SYCLTarget &sycl_target, LayerCompressor &layer_compressor,
            std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
            std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int,
            const int nranks = 0, const int *ranks = nullptr)
      : sycl_target(sycl_target), layer_compressor(layer_compressor),
        particle_dats_real(particle_dats_real),
        particle_dats_int(particle_dats_int), particle_packer(sycl_target),
        particle_unpacker(sycl_target), h_send_ranks(sycl_target, 1),
        h_recv_ranks(sycl_target, 1), h_send_requests(sycl_target, 1),
        h_recv_requests(sycl_target, 1), h_status(sycl_target, 1) {

    std::set<int> ranks_set{};
    const int rank = this->sycl_target.comm_pair.rank_parent;
    const int size = this->sycl_target.comm_pair.size_parent;
    this->comm = this->sycl_target.comm_pair.comm_parent;

    // Get the set of remote ranks this rank can send to using a local pattern
    for (int rankx = 0; rankx < nranks; rankx++) {
      const int remote_rank = ranks[rankx];
      NESOASSERT(((remote_rank >= 0) && (remote_rank < size)),
                 "Unrealistic rank passed.");
      if (remote_rank != rank) {
        ranks_set.insert(remote_rank);
      }
    }
    this->num_remote_send_ranks = ranks_set.size();
    this->h_send_ranks.realloc_no_copy(this->num_remote_send_ranks);
    this->h_send_requests.realloc_no_copy(this->num_remote_send_ranks);
    int ix_tmp = 0;
    for (auto &rx : ranks_set) {
      this->h_send_ranks.ptr[ix_tmp++] = rx;
    }

    // inform the remote ranks that this rank will send to them

    // allocate a buffer to accumulate the number of remote ranks that
    // will send to this rank
    MPI_Win rank_win;
    int *rank_win_data = nullptr;
    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &rank_win_data, &rank_win));
    NESOASSERT((rank_win_data != nullptr), "Failed to allocate window");

    // loop over the ranks to send to and increment the remote counter.
    rank_win_data[0] = 0;
    MPICHK(MPI_Barrier(this->comm));
    const int one[1] = {1};
    std::vector<int> tags(this->num_remote_send_ranks);
    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {
      const int rank_tmp = this->h_send_ranks.ptr[rankx];
      MPICHK(MPI_Win_lock(MPI_LOCK_SHARED, rank_tmp, 0, rank_win));
      MPICHK(MPI_Get_accumulate(one, 1, MPI_INT, tags.data() + rankx, 1,
                                MPI_INT, rank_tmp, 0, 1, MPI_INT, MPI_SUM,
                                rank_win));
      MPICHK(MPI_Win_unlock(rank_tmp, rank_win));
    }
    MPICHK(MPI_Barrier(this->comm));
    this->num_remote_recv_ranks = rank_win_data[0];
    MPICHK(MPI_Win_free(&rank_win));

    // allocate space to store the remote ranks
    this->h_recv_ranks.realloc_no_copy(this->num_remote_recv_ranks);
    this->h_recv_requests.realloc_no_copy(this->num_remote_recv_ranks);

    // sendrecv ranks
    // setup recv
    for (int rankx = 0; rankx < this->num_remote_recv_ranks; rankx++) {
      MPICHK(MPI_Irecv(&this->h_recv_ranks.ptr[rankx], 1, MPI_INT,
                       MPI_ANY_SOURCE, rankx, this->comm,
                       &this->h_recv_requests.ptr[rankx]));
    }
    // setup send
    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {
      MPICHK(MPI_Isend(&rank, 1, MPI_INT, this->h_send_ranks.ptr[rankx],
                       tags[rankx], this->comm,
                       &this->h_send_requests.ptr[rankx]));
    }

    h_status.realloc_no_copy(
        MAX(this->num_remote_send_ranks, this->num_remote_recv_ranks));

    // wait for sends
    MPICHK(MPI_Waitall(this->num_remote_send_ranks, this->h_send_requests.ptr,
                       this->h_status.ptr));
    // wait for recvs
    MPICHK(MPI_Waitall(this->num_remote_recv_ranks, this->h_recv_requests.ptr,
                       this->h_status.ptr));
    // h_recv_ranks should now contain remote ranks that want to send to this
    // rank
  };

  inline void set_mpi_rank_dat(ParticleDatShPtr<INT> mpi_rank_dat) {
    this->mpi_rank_dat = mpi_rank_dat;
  }

  inline void move() {
    auto t0 = profile_timestamp();

    sycl_target.profile_map.inc("LocalMove", "Move", 1,
                                profile_elapsed(t0, profile_timestamp()));
  };
};

} // namespace NESO::Particles

#endif
