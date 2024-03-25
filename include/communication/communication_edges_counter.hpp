#ifndef _NESO_PARTICLES_COMMUNICATION_EDGES_COUNTER_HPP_
#define _NESO_PARTICLES_COMMUNICATION_EDGES_COUNTER_HPP_

#include "communication_typedefs.hpp"
#include <cstdint>
#include <mpi.h>
#include <vector>

namespace NESO::Particles {

/**
 * TODO
 */
class CommunicationEdgesCounter {
protected:
  MPI_Comm comm;
  MPI_Win recv_win;
  int *recv_win_data;
  MPI_Request mpi_request;

public:
  /**
   * TODO
   */
  CommunicationEdgesCounter(MPI_Comm comm) : comm(comm) {
    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &this->recv_win_data, &this->recv_win));
  }

  /**
   * TODO
   */
  inline void reset() {
    recv_win_data[0] = 0;
    MPICHK(MPI_Ibarrier(this->comm, &this->mpi_request));
  }

  /**
   * TODO
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
   * TODO
   */
  inline int get_count() {
    MPICHK(MPI_Wait(&this->mpi_request, MPI_STATUS_IGNORE));
    return this->recv_win_data[0];
  }

  /**
   * TODO
   */
  inline void exchange_send_recv_counts(std::vector<int> &recv_ranks,
                                        std::vector<std::int64_t> &recv_data,
                                        std::vector<int> &send_ranks,
                                        std::vector<std::int64_t> &send_data) {
    const int num_recv_ranks = recv_ranks.size();
    const int num_send_ranks = send_ranks.size();

    std::vector<MPI_Request> send_requests(num_send_ranks);
    std::vector<MPI_Request> recv_requests(num_recv_ranks);

    for (int rankx = 0; rankx < num_send_ranks; rankx++) {
      MPICHK(MPI_Irecv(&send_data[rankx], 1, MPI_INT64_T, MPI_ANY_SOURCE, 127,
                       this->comm, &send_requests[rankx]));
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
      send_ranks.at(rankx) = rank;
    }

    MPICHK(
        MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUSES_IGNORE));
  }

  /**
   * TODO
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
   * TODO
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
   * TODO
   */
  inline void free() { MPICHK(MPI_Win_free(&this->recv_win)); }
};

} // namespace NESO::Particles

#endif
