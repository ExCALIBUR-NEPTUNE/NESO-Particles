#ifndef _NESO_PARTICLES_GLOBAL_MOVE_EXCHANGE
#define _NESO_PARTICLES_GLOBAL_MOVE_EXCHANGE

#include <CL/sycl.hpp>
#include <mpi.h>

#include "communication.hpp"
#include "compute_target.hpp"
#include "packing_unpacking.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

/**
 * This class handles the efficient and asynchronous exchange of particle
 * counts and data between MPI ranks when moving particles between cells of the
 * MeshHierarchy.
 */
class GlobalMoveExchange {

private:
  MPI_Comm comm;
  MPI_Win recv_win;
  int *recv_win_data;
  MPI_Request mpi_request;

  BufferHost<MPI_Request> h_send_requests;
  BufferHost<MPI_Request> h_recv_requests;
  BufferHost<MPI_Status> h_recv_status;

public:
  /// Number of remote ranks to send particles to.
  int num_remote_send_ranks;
  /// Number of remote ranks to recv particles from.
  int num_remote_recv_ranks;
  /// Host array of MPI ranks to send particles to.
  BufferHost<int> h_send_ranks;
  /// Host array of MPI ranks to recv particles from.
  BufferHost<int> h_recv_ranks;
  /// Host array of particle counts for each send rank.
  BufferHost<int> h_send_rank_npart;
  /// Host array of particle counts for each recv rank.
  BufferHost<int> h_recv_rank_npart;

  /// Compute device used by the instance.
  SYCLTarget &sycl_target;

  ~GlobalMoveExchange() { MPICHK(MPI_Win_free(&this->recv_win)); };

  /**
   * Construct a new instance to exchange particle counts and data.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   */
  GlobalMoveExchange(SYCLTarget &sycl_target)
      : sycl_target(sycl_target), h_send_ranks(sycl_target, 1),
        h_recv_ranks(sycl_target, 1), h_send_rank_npart(sycl_target, 1),
        h_recv_rank_npart(sycl_target, 1), h_send_requests(sycl_target, 1),
        h_recv_requests(sycl_target, 1), h_recv_status(sycl_target, 1),
        comm(sycl_target.comm_pair.comm_parent) {
    // Create a MPI_Win used to sum the number of remote ranks that will
    // send particles to this rank.
    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &this->recv_win_data, &this->recv_win));
  };

  /**
   * Initialise the start of a new epoch where global communication is
   * identified. Collective on the communicator.
   */
  inline void npart_exchange_init() {

    // copy the entries from the shared buffer to the host buffer to prevent
    // implicit copies back and forwards if this exchange is ran async with
    // the packing.
    recv_win_data[0] = 0;
    MPICHK(MPI_Ibarrier(this->comm, &this->mpi_request));
  }

  /**
   * Communicate how many ranks will send particles to each rank. Collective on
   * the communicator.
   *
   * @param num_remote_send_ranks Number of remote ranks particles will be sent
   * to.
   * @param dh_send_ranks Array of remote ranks particles will be sent to.
   * @param h_send_rank_npart Array of particle counts that will be sent to
   * each rank.
   */
  inline void npart_exchange_sendrecv(const int num_remote_send_ranks,
                                      BufferDeviceHost<int> &dh_send_ranks,
                                      BufferHost<int> &h_send_rank_npart) {

    auto t0 = profile_timestamp();
    this->h_send_ranks.realloc_no_copy(dh_send_ranks.size);
    this->h_send_rank_npart.realloc_no_copy(h_send_rank_npart.size);
    this->num_remote_send_ranks = num_remote_send_ranks;

    NESOASSERT(dh_send_ranks.size >= num_remote_send_ranks,
               "Buffer size does not match number of remote ranks.");
    NESOASSERT(h_send_rank_npart.size >= num_remote_send_ranks,
               "Buffer size does not match number of remote ranks.");
    for (int rx = 0; rx < num_remote_send_ranks; rx++) {
      const int rank_tmp = dh_send_ranks.h_buffer.ptr[rx];
      NESOASSERT(
          ((rank_tmp >= 0) && (rank_tmp < sycl_target.comm_pair.size_parent)),
          "Invalid rank");
      this->h_send_ranks.ptr[rx] = rank_tmp;
      const int npart_tmp = h_send_rank_npart.ptr[rx];
      NESOASSERT(npart_tmp > 0, "Invalid particle count");
      this->h_send_rank_npart.ptr[rx] = npart_tmp;
    }

    sycl_target.profile_map.inc("GlobalMoveExchange", "npart_exchange_sendrecv_pre_wait", 1,
                                profile_elapsed(t0, profile_timestamp()));
    MPICHK(MPI_Wait(&this->mpi_request, MPI_STATUS_IGNORE));
    sycl_target.profile_map.inc("GlobalMoveExchange", "npart_exchange_sendrecv_post_wait", 1,
                                profile_elapsed(t0, profile_timestamp()));

    const int one[1] = {1};
    int recv[1];

    auto t0_rma = profile_timestamp();
    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {
      const int rank = this->h_send_ranks.ptr[rankx];
      MPICHK(MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->recv_win));
      MPICHK(MPI_Get_accumulate(one, 1, MPI_INT, recv, 1, MPI_INT, rank, 0, 1,
                                MPI_INT, MPI_SUM, this->recv_win));
      MPICHK(MPI_Win_unlock(rank, this->recv_win));
    }
    sycl_target.profile_map.inc("GlobalMoveExchange", "RMA", 1,
                                profile_elapsed(t0_rma, profile_timestamp()));

    MPICHK(MPI_Ibarrier(this->comm, &this->mpi_request));

    sycl_target.profile_map.inc("GlobalMoveExchange", "npart_exchange_sendrecv", 1,
                                profile_elapsed(t0, profile_timestamp()));
  }

  /**
   * Finalise the communication which indicates to remote ranks the number of
   * ranks that will send them particles.
   */
  inline void npart_exchange_finalise() {

    auto t0_npart = profile_timestamp();
    // Wait for the accumulation that counts how many remote ranks will send to
    // this rank.
    MPICHK(MPI_Wait(&this->mpi_request, MPI_STATUS_IGNORE));
    this->num_remote_recv_ranks = this->recv_win_data[0];

    // send particle counts for ranks this ranks will send to and recv counts
    // from the num_remote_ranks this rank will recv from.
    this->h_recv_rank_npart.realloc_no_copy(this->num_remote_recv_ranks);
    this->h_recv_requests.realloc_no_copy(this->num_remote_recv_ranks);
    this->h_send_requests.realloc_no_copy(this->num_remote_send_ranks);

    // non-blocking recv of particle counts
    for (int rankx = 0; rankx < this->num_remote_recv_ranks; rankx++) {
      MPICHK(MPI_Irecv(&this->h_recv_rank_npart.ptr[rankx], 1, MPI_INT,
                       MPI_ANY_SOURCE, 42, this->comm,
                       &this->h_recv_requests.ptr[rankx]));
    }

    // non-blocking send of particle counts
    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {
      MPICHK(MPI_Isend(&this->h_send_rank_npart.ptr[rankx], 1, MPI_INT,
                       this->h_send_ranks.ptr[rankx], 42, this->comm,
                       &this->h_send_requests.ptr[rankx]));
    }

    // space to store the remote ranks
    this->h_recv_ranks.realloc_no_copy(this->num_remote_recv_ranks);
    this->h_recv_status.realloc_no_copy(
        MAX(this->num_remote_send_ranks, this->num_remote_recv_ranks));
    MPICHK(MPI_Waitall(this->num_remote_recv_ranks, this->h_recv_requests.ptr,
                       this->h_recv_status.ptr));
    // the order of the recvs gives an order to the ranks that will send to
    // this rank
    for (int rankx = 0; rankx < this->num_remote_recv_ranks; rankx++) {
      const int remote_rank = this->h_recv_status.ptr[rankx].MPI_SOURCE;
      this->h_recv_ranks.ptr[rankx] = remote_rank;
      NESOASSERT(((remote_rank >= 0) &&
                  (remote_rank < this->sycl_target.comm_pair.size_parent)),
                 "Recv rank is invalid.");
      NESOASSERT((this->h_recv_rank_npart.ptr[rankx] > 0),
                 "A remote rank is trying to send 0 (or fewer) particles to "
                 "this rank");
    }

    MPICHK(MPI_Waitall(this->num_remote_send_ranks, this->h_send_requests.ptr,
                       this->h_recv_status.ptr));
    sycl_target.profile_map.inc("GlobalMoveExchange", "npart_send_recv", 1,
                                profile_elapsed(t0_npart, profile_timestamp()));
  }

  /**
   *  Start the exchange the particle data. Collective on the communicator.
   *
   *  @param particle_packer ParticlePacker instance to pack particle data.
   *  @param particle_unpacker ParticleUnpacker instance to use to unpack
   *  particle data.
   */
  inline void exchange_init(ParticlePacker &particle_packer,
                            ParticleUnpacker &particle_unpacker) {

    auto t0 = profile_timestamp();

    // Get the packed particle data on the host
    auto h_send_buffer = particle_packer.get_packed_data_on_host(
        this->num_remote_send_ranks, this->h_send_rank_npart.ptr);

    auto h_send_offsets = particle_packer.h_send_offsets.ptr;
    auto h_recv_buffer = particle_unpacker.h_recv_buffer.ptr;
    auto h_recv_offsets = particle_unpacker.h_recv_offsets.ptr;
    const int num_bytes_per_particle = particle_packer.num_bytes_per_particle;

    NESOASSERT(particle_packer.num_bytes_per_particle ==
                   particle_unpacker.num_bytes_per_particle,
               "packer and unpacker disagree on size of a particle");

    // non-blocking recv of packed particle data
    for (int rankx = 0; rankx < this->num_remote_recv_ranks; rankx++) {

      MPICHK(
          MPI_Irecv(&h_recv_buffer[h_recv_offsets[rankx]],
                    this->h_recv_rank_npart.ptr[rankx] * num_bytes_per_particle,
                    MPI_CHAR, this->h_recv_ranks.ptr[rankx], 43, this->comm,
                    &this->h_recv_requests.ptr[rankx]));
    }

    // non-blocking send of particle data
    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {

      MPICHK(
          MPI_Isend(&h_send_buffer[h_send_offsets[rankx]],
                    this->h_send_rank_npart.ptr[rankx] * num_bytes_per_particle,
                    MPI_CHAR, this->h_send_ranks.ptr[rankx], 43, this->comm,
                    &this->h_send_requests.ptr[rankx]));
    }

    sycl_target.profile_map.inc("GlobalMoveExchange", "exchange_init", 1,
                                profile_elapsed(t0, profile_timestamp()));
  }

  /**
   *  Finalise the exchange the particle data. Collective on the communicator.
   *
   *  @param particle_unpacker ParticleUnpacker instance to use to unpack
   *  particle data.
   */
  inline void exchange_finalise(ParticleUnpacker &particle_unpacker) {

    const int num_bytes_per_particle = particle_unpacker.num_bytes_per_particle;
    MPICHK(MPI_Waitall(this->num_remote_recv_ranks, this->h_recv_requests.ptr,
                       this->h_recv_status.ptr));

    // Check this rank recv'd the correct number of bytes from each remote
    for (int rankx = 0; rankx < this->num_remote_recv_ranks; rankx++) {
      MPI_Status *status = &this->h_recv_status.ptr[rankx];
      int count = -1;
      MPICHK(MPI_Get_count(status, MPI_CHAR, &count));
      NESOASSERT(count == (this->h_recv_rank_npart.ptr[rankx] *
                           num_bytes_per_particle),
                 "recv'd incorrect number of bytes");
    }

    MPICHK(MPI_Waitall(this->num_remote_send_ranks, this->h_send_requests.ptr,
                       this->h_recv_status.ptr));
  }
};

} // namespace NESO::Particles

#endif
