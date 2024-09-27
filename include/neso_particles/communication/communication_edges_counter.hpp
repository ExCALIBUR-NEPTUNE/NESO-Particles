#ifndef _NESO_PARTICLES_COMMUNICATION_EDGES_COUNTER_HPP_
#define _NESO_PARTICLES_COMMUNICATION_EDGES_COUNTER_HPP_

#include "communication_typedefs.hpp"
#include <cstdint>
#include <mpi.h>
#include <vector>

namespace NESO::Particles {

/**
 * Helper class for setting up communication between MPI ranks where at the
 * outset only one of the two MPI ranks knows that communication should occur.
 * See @ref get_remote_ranks and @ref exchange_send_recv_data as the primary
 * methods.
 */
class CommunicationEdgesCounter {
protected:
  bool allocated;
  MPI_Comm comm;
  MPI_Win recv_win;
  int *recv_win_data;
  MPI_Request mpi_request;

public:
  ~CommunicationEdgesCounter() {
    if (this->allocated == true) {
      nprint("CommunicationEdgesCounter::free() was not called.");
    }
  }

  /**
   * Create a new instance for subsequent communication. Collective for all MPI
   * ranks on the communicator.
   *
   * @param comm MPI communicator.
   */
  CommunicationEdgesCounter(MPI_Comm comm) : comm(comm) {
    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &this->recv_win_data, &this->recv_win));
    this->allocated = true;
  }

  /**
   * Reset the internal state. Collective on the communicator.
   */
  inline void reset() {
    recv_win_data[0] = 0;
    MPICHK(MPI_Ibarrier(this->comm, &this->mpi_request));
  }

  /**
   * Determine how many remote MPI ranks are to send/recv data with this MPI
   * rank. The number of ranks is returned by @ref get_count. Collective on the
   * communicator.
   *
   * @param ranks Vector of remote ranks which this rank will communicate with.
   */
  inline void init_count(std::vector<int> &ranks) {
    MPICHK(MPI_Wait(&this->mpi_request, MPI_STATUS_IGNORE));
    const int one[1] = {1};
    int recv[1];

    for (const int rank : ranks) {
      MPICHK(MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->recv_win));
      MPICHK(MPI_Get_accumulate(one, 1, MPI_INT, recv, 1, MPI_INT, rank, 0, 1,
                                MPI_INT, MPI_SUM, this->recv_win));
      MPICHK(MPI_Win_unlock(rank, this->recv_win));
    }

    MPICHK(MPI_Ibarrier(this->comm, &this->mpi_request));
  }

  /**
   * Get the number of remote ranks which will setup communication patterns
   * with this MPI rank. See @ref init_count which must be called before
   * calling this method.
   *
   * @returns Number of remote MPI ranks which had this MPI rank in the vector
   * they passed to @ref init_count.
   */
  inline int get_count() {
    MPICHK(MPI_Wait(&this->mpi_request, MPI_STATUS_IGNORE));
    return this->recv_win_data[0];
  }

  /**
   * Send to each remote rank an integer quantity, e.g. a number of bytes which
   * will be sent in a call to @ref exchange_send_recv_data. Collective on the
   * communicator.
   *
   * @param recv_ranks[in] Remote MPI ranks which this rank will send data.
   * @param recv_data[in] An integer to send to each remote rank.
   * @param send_ranks[in, out] On return contains remote MPI ranks which will
   * send this rank data.
   * @param send_data[in, out] On return contains the integers that the remote
   * ranks sent to this rank.
   */
  inline void exchange_send_recv_counts(std::vector<int> &recv_ranks,
                                        std::vector<std::int64_t> &recv_data,
                                        std::vector<int> &send_ranks,
                                        std::vector<std::int64_t> &send_data,
                                        const bool ordered = false) {
    const int num_recv_ranks = recv_ranks.size();
    const int num_send_ranks = send_ranks.size();

    std::vector<MPI_Request> send_requests(num_send_ranks);
    std::vector<MPI_Request> recv_requests(num_recv_ranks);

    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int rank = ordered ? send_ranks.at(rankx) : MPI_ANY_SOURCE;
      MPICHK(MPI_Irecv(&send_data[rankx], 1, MPI_INT64_T, rank, 127, this->comm,
                       &send_requests[rankx]));
    }

    for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
      const int rank = recv_ranks.at(rankx);
      MPICHK(MPI_Isend(&recv_data[rankx], 1, MPI_INT64_T, rank, 127, this->comm,
                       &recv_requests[rankx]));
    }
    std::vector<MPI_Status> send_status(num_send_ranks);
    MPICHK(
        MPI_Waitall(num_send_ranks, send_requests.data(), send_status.data()));

    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int rank = send_status.at(rankx).MPI_SOURCE;
      if (ordered) {
        NESOASSERT(send_ranks.at(rankx) == rank, "Unexpected rank missmatch.");
      } else {
        send_ranks.at(rankx) = rank;
      }
    }

    MPICHK(
        MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUSES_IGNORE));
  }

  /**
   *  Send to each remote rank variable length data. Collective on all ranks
   *  that participate. For exchanging the number of bytes to send with this
   *  call see @ref exchange_send_recv_counts.
   *
   *  @param[in] recv_ranks Remote MPI ranks this rank will send data to.
   *  @param[in] recv_counts Number of bytes to send to each remote rank.
   *  @param[in] recv_data Data to send to each remote rank.
   *  @param[in] send_ranks Remote MPI ranks this rank will receive from.
   *  @param[in] send_counts Number of bytes each remote rank will send to this
   * rank.
   *  @param[in, out] send_data Output locations for received data. Each pointer
   * should point to memory sufficiently sized for the send counts.
   */
  inline void exchange_send_recv_data(std::vector<int> &recv_ranks,
                                      std::vector<std::int64_t> &recv_counts,
                                      std::vector<void *> &recv_data,
                                      std::vector<int> &send_ranks,
                                      std::vector<std::int64_t> &send_counts,
                                      std::vector<void *> &send_data) {
    const int num_recv_ranks = recv_ranks.size();
    const int num_send_ranks = send_ranks.size();
    std::vector<MPI_Request> send_requests(num_send_ranks);
    std::vector<MPI_Request> recv_requests(num_recv_ranks);
    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      const int num_bytes = static_cast<int>(send_counts.at(rankx));
      const int rank = send_ranks.at(rankx);
      MPICHK(MPI_Irecv(send_data[rankx], num_bytes, MPI_BYTE, rank, 126,
                       this->comm, &send_requests[rankx]));
    }
    for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
      const int num_bytes = static_cast<int>(recv_counts.at(rankx));
      const int rank = recv_ranks.at(rankx);
      MPICHK(MPI_Isend(recv_data[rankx], num_bytes, MPI_BYTE, rank, 126,
                       this->comm, &recv_requests[rankx]));
    }
    MPICHK(
        MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUSES_IGNORE));
    MPICHK(
        MPI_Waitall(num_send_ranks, send_requests.data(), MPI_STATUSES_IGNORE));
  }

  /**
   * Helper method to call the relevant methods required to setup a "standard"
   * pattern. Collective on the communicator.
   *
   * @param[in] recv_ranks Remote MPI ranks this rank will send data to.
   * @param[in] recv_data Integer quantity to initially send to the remote rank,
   * e.g. byte count.
   * @param[in, out] send_ranks On return contains the remote ranks which will
   * send data to this rank.
   * @param[in, out] send_data On return contains the integer that the remote
   * ranks send to this rank.
   */
  inline void get_remote_ranks(std::vector<int> &recv_ranks,
                               std::vector<std::int64_t> &recv_data,
                               std::vector<int> &send_ranks,
                               std::vector<std::int64_t> &send_data) {
    this->reset();
    this->init_count(recv_ranks);
    const int num_recv_ranks = this->get_count();
    send_ranks.clear();
    send_ranks.resize(num_recv_ranks);
    send_data.clear();
    send_data.resize(num_recv_ranks);
    this->exchange_send_recv_counts(recv_ranks, recv_data, send_ranks,
                                    send_data);
  }

  /**
   * Free this instance. Collective on the communicator.
   */
  inline void free() {
    MPICHK(MPI_Win_free(&this->recv_win));
    this->allocated = false;
  }
};

} // namespace NESO::Particles

#endif
