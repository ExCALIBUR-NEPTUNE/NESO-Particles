#ifndef _NESO_PARTICLES_COMMUNICATION_GLOBAL_MOVE_COMMUNICATION_HPP_
#define _NESO_PARTICLES_COMMUNICATION_GLOBAL_MOVE_COMMUNICATION_HPP_

#include "communication_typedefs.hpp"
#include <memory>
#include <mpi.h>

namespace NESO::Particles {

/**
 * This type wraps the MPI data structures that the global communication uses
 * that must be created and freed collectively on the communicator. By using
 * this class for the MPI dependent components the downstream components, e.g.
 * ParticleGroup, can simply use C++ destructors.
 */
struct GlobalMoveCommunication {
  /// Disable (implicit) copies.
  GlobalMoveCommunication(const GlobalMoveCommunication &st) = delete;
  /// Disable (implicit) copies.
  GlobalMoveCommunication &operator=(GlobalMoveCommunication const &a) = delete;

  /// MPI communicator used for particle transport.
  MPI_Comm comm;
  /// Is the MPI_Win allocated or not.
  bool recv_win_allocated;
  /// The MPI_Win object.
  MPI_Win recv_win;
  /// Data for MPI_Win, will be allocated for 1 int.
  int *recv_win_data;

  /**
   * Create new instance. Must be called collectively on the communicator.
   */
  GlobalMoveCommunication(MPI_Comm comm)
      : comm(comm), recv_win_allocated(false) {
    // Create a MPI_Win used to sum the number of remote ranks that will
    // send particles to this rank.
    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &this->recv_win_data, &this->recv_win));
    this->recv_win_allocated = true;
  }

  /**
   * Free the internal MPI data structures. Must be called collectively on the
   * communicator. Must be called before MPI_Finalize.
   */
  inline void free() {
    if (this->recv_win_allocated) {
      MPICHK(MPI_Win_free(&this->recv_win));
      this->recv_win_allocated = false;
    }
  }

  ~GlobalMoveCommunication() { this->free(); };
};

typedef std::shared_ptr<GlobalMoveCommunication>
    GlobalMoveCommunicationSharedPtr;

} // namespace NESO::Particles

#endif
