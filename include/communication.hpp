#ifndef _NESO_PARTICLES_COMMUNICATION
#define _NESO_PARTICLES_COMMUNICATION

#include <mpi.h>

#include "typedefs.hpp"

#define _MACRO_STRING(x) #x
#define STR(x) _MACRO_STRING(x)
#define MPICHK(cmd) NESOASSERT(cmd == MPI_SUCCESS, "MPI ERROR");

namespace NESO::Particles {

/*
 * Wrapper around a pair of inter and intra MPI Comms for shared memory
 * purposes.
 */
class CommPair {

private:
  bool allocated = false;

public:
  MPI_Comm comm_parent, comm_inter, comm_intra;
  int rank_parent, rank_inter, rank_intra;
  int size_parent, size_inter, size_intra;

  CommPair(){};

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

  ~CommPair(){};
};

} // namespace NESO::Particles

#endif
