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
  CommunicationEdgesCounter(MPI_Comm comm);

  /**
   * Reset the internal state. Collective on the communicator.
   */
  void reset();

  /**
   * Determine how many remote MPI ranks are to send/recv data with this MPI
   * rank. The number of ranks is returned by @ref get_count. Collective on the
   * communicator.
   *
   * @param ranks Vector of remote ranks which this rank will communicate with.
   */
  void init_count(std::vector<int> &ranks);

  /**
   * Get the number of remote ranks which will setup communication patterns
   * with this MPI rank. See @ref init_count which must be called before
   * calling this method.
   *
   * @returns Number of remote MPI ranks which had this MPI rank in the vector
   * they passed to @ref init_count.
   */
  int get_count();

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
  void exchange_send_recv_counts(std::vector<int> &recv_ranks,
                                 std::vector<std::int64_t> &recv_data,
                                 std::vector<int> &send_ranks,
                                 std::vector<std::int64_t> &send_data,
                                 const bool ordered = false);

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
  void exchange_send_recv_data(std::vector<int> &recv_ranks,
                               std::vector<std::int64_t> &recv_counts,
                               std::vector<void *> &recv_data,
                               std::vector<int> &send_ranks,
                               std::vector<std::int64_t> &send_counts,
                               std::vector<void *> &send_data);

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
  void get_remote_ranks(std::vector<int> &recv_ranks,
                        std::vector<std::int64_t> &recv_data,
                        std::vector<int> &send_ranks,
                        std::vector<std::int64_t> &send_data);

  /**
   * Free this instance. Collective on the communicator.
   */
  void free();
};

} // namespace NESO::Particles

#endif
