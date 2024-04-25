#ifndef _NESO_PARTICLES_COMM_PAIR_HPP
#define _NESO_PARTICLES_COMM_PAIR_HPP

#include "communication_typedefs.hpp"

namespace NESO::Particles {

/**
 * Wrapper around a pair of inter and intra MPI Comms for shared memory
 * purposes.
 */
class CommPair {

private:
  bool allocated = false;

public:
  /// Disable (implicit) copies.
  CommPair(const CommPair &st) = delete;
  /// Disable (implicit) copies.
  CommPair &operator=(CommPair const &a) = delete;

  /// Parent (i.e. global for the simulation) MPI communicator.
  MPI_Comm comm_parent;
  /// Communicator between one rank on each shared memory region.
  MPI_Comm comm_inter;
  /// Communicator between the ranks in a shared memory region.
  MPI_Comm comm_intra;
  /// MPI rank in the parent communicator.
  int rank_parent;
  /// MPI rank in the inter shared memory region communicator.
  int rank_inter;
  /// MPI rank within the shared memory region communicator.
  int rank_intra;
  /// Size of the parent communicator.
  int size_parent;
  /// Size of the inter shared memory communicator.
  int size_inter;
  /// Size of the intra shared memory communicator.
  int size_intra;

  ~CommPair(){};
  CommPair(){};

  /**
   * Create a new set of inter and intra shared memory region communicators
   * from a parent communicator. Must be called collectively on the parent
   * communicator.
   *
   * @param comm_parent MPI communicator to derive intra and inter communicators
   * from.
   */
  CommPair(MPI_Comm comm_parent) {
    this->comm_parent = comm_parent;

    int rank_parent;
    MPICHK(MPI_Comm_rank(comm_parent, &rank_parent))
    MPICHK(MPI_Comm_split_type(comm_parent, MPI_COMM_TYPE_SHARED, 0,
                               MPI_INFO_NULL, &this->comm_intra))

    int rank_intra;
    MPICHK(MPI_Comm_rank(this->comm_intra, &rank_intra))
    const int colour_intra = (rank_intra == 0) ? 1 : MPI_UNDEFINED;
    MPICHK(MPI_Comm_split(comm_parent, colour_intra, 0, &this->comm_inter))

    this->allocated = true;

    MPICHK(MPI_Comm_rank(this->comm_parent, &this->rank_parent))
    MPICHK(MPI_Comm_rank(this->comm_intra, &this->rank_intra))
    MPICHK(MPI_Comm_size(this->comm_parent, &this->size_parent))
    MPICHK(MPI_Comm_size(this->comm_intra, &this->size_intra))
    if (comm_inter != MPI_COMM_NULL) {
      MPICHK(MPI_Comm_rank(this->comm_inter, &this->rank_inter))
      MPICHK(MPI_Comm_size(this->comm_inter, &this->size_inter))
    }
  };

  /**
   * Free the communicators held by the CommPair instance.
   */
  void free() {
    int flag;
    MPICHK(MPI_Initialized(&flag))
    if (allocated && flag) {

      if ((this->comm_intra != MPI_COMM_NULL) &&
          (this->comm_intra != MPI_COMM_WORLD)) {
        MPICHK(MPI_Comm_free(&this->comm_intra))
        this->comm_intra = MPI_COMM_NULL;
      }

      if ((this->comm_inter != MPI_COMM_NULL) &&
          (this->comm_inter != MPI_COMM_WORLD)) {
        MPICHK(MPI_Comm_free(&this->comm_inter))
        this->comm_intra = MPI_COMM_NULL;
      }
    }
    this->allocated = false;
  }
};

} // namespace NESO::Particles

#endif
