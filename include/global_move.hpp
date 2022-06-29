#ifndef _NESO_PARTICLES_GLOBAL_MOVE
#define _NESO_PARTICLES_GLOBAL_MOVE

#include <CL/sycl.hpp>
#include <mpi.h>

#include "communication.hpp"
#include "compute_target.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

class GlobalMoveExchange {

private:
  BufferHost<int> h_send_ranks;
  BufferHost<int> h_send_rank_npart;
  MPI_Comm comm;
  MPI_Win recv_win;
  int *recv_win_data;
  MPI_Request mpi_request;
  int num_remote_send_ranks;

public:
  SYCLTarget &sycl_target;

  ~GlobalMoveExchange() { MPICHK(MPI_Win_free(&this->recv_win)); };
  GlobalMoveExchange(SYCLTarget &sycl_target)
      : sycl_target(sycl_target), h_send_ranks(sycl_target, 1),
        h_send_rank_npart(sycl_target, 1),
        comm(sycl_target.comm_pair.comm_parent) {
    // Create a MPI_Win used to sum the number of remote ranks that will
    // send particles to this rank.
    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &this->recv_win_data, &this->recv_win));
  };

  inline void npart_exchange_init() {

    // copy the entries from the shared buffer to the host buffer to prevent
    // implicit copies back and forwards if this exchange is ran async with
    // the packing.
    recv_win_data[0] = 0;
    MPICHK(MPI_Ibarrier(this->comm, &this->mpi_request));
  }

  inline void npart_exchange_sendrecv(const int num_remote_send_ranks,
                                      BufferShared<int> &s_send_ranks,
                                      BufferShared<int> &s_send_rank_npart) {
    h_send_ranks.realloc_no_copy(s_send_ranks.size);
    h_send_rank_npart.realloc_no_copy(s_send_rank_npart.size);
    this->num_remote_send_ranks = num_remote_send_ranks;

    // explicit copy to avoid a data migration if possible
    auto e1 = this->sycl_target.queue.memcpy(h_send_ranks.ptr, s_send_ranks.ptr,
                                             s_send_ranks.size);
    auto e2 = this->sycl_target.queue.memcpy(
        h_send_rank_npart.ptr, s_send_rank_npart.ptr, s_send_rank_npart.size);
    e1.wait();
    e2.wait();

    MPICHK(MPI_Wait(&this->mpi_request, MPI_STATUS_IGNORE));

    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {
    }

    MPICHK(MPI_Ibarrier(this->comm, &this->mpi_request));
  }

  inline void npart_exchange_finalise() {
    MPICHK(MPI_Wait(&this->mpi_request, MPI_STATUS_IGNORE));
  }
};

} // namespace NESO::Particles

#endif
