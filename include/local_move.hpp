#ifndef _NESO_PARTICLES_LOCAL_MOVE
#define _NESO_PARTICLES_LOCAL_MOVE

#include <CL/sycl.hpp>
#include <mpi.h>
#include <set>
#include <vector>

#include "cell_dat_compression.hpp"
#include "communication.hpp"
#include "compute_target.hpp"
#include "departing_particle_identification.hpp"
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

  BufferDeviceHost<int> dh_send_rank_map;
  BufferHost<int> h_send_rank_npart;
  BufferHost<int> h_recv_rank_npart;

  DepartingIdentify departing_identify;

  int in_flight_sends;
  int in_flight_recvs;

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
        h_recv_requests(sycl_target, 1), h_status(sycl_target, 1),
        dh_send_rank_map(sycl_target, sycl_target.comm_pair.size_parent),
        h_send_rank_npart(sycl_target, 1), h_recv_rank_npart(sycl_target, 1),
        departing_identify(sycl_target) {

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
    this->h_send_rank_npart.realloc_no_copy(this->num_remote_send_ranks);
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
    this->h_recv_rank_npart.realloc_no_copy(this->num_remote_recv_ranks);
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
    // prevent race condition on the send/recv from elsewhere
    MPICHK(MPI_Barrier(this->comm));
    // h_recv_ranks should now contain remote ranks that want to send to this
    // rank

    int index = 0;
    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {
      const int rank = this->h_send_ranks.ptr[rankx];
      this->dh_send_rank_map.h_buffer.ptr[rank] = index++;
    }
    this->dh_send_rank_map.host_to_device();
  };

  inline void set_mpi_rank_dat(ParticleDatShPtr<INT> mpi_rank_dat) {
    this->mpi_rank_dat = mpi_rank_dat;
  }

  inline void npart_exchange_sendrecv() {

    // setup recv
    for (int rankx = 0; rankx < this->num_remote_recv_ranks; rankx++) {
      MPICHK(MPI_Irecv(&this->h_recv_rank_npart.ptr[rankx], 1, MPI_INT,
                       this->h_recv_ranks.ptr[rankx], 43, this->comm,
                       &this->h_recv_requests.ptr[rankx]));
    }
    // setup send
    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {
      MPICHK(MPI_Isend(&this->h_send_rank_npart.ptr[rankx], 1, MPI_INT,
                       this->h_send_ranks.ptr[rankx], 43, this->comm,
                       &this->h_send_requests.ptr[rankx]));
    }
    // wait for sends
    MPICHK(MPI_Waitall(this->num_remote_send_ranks, this->h_send_requests.ptr,
                       this->h_status.ptr));
    // wait for recvs
    MPICHK(MPI_Waitall(this->num_remote_recv_ranks, this->h_recv_requests.ptr,
                       this->h_status.ptr));
  }

  inline void exchange_init() {
    // Get the packed particle data on the host
    auto h_send_buffer = this->particle_packer.get_packed_data_on_host(
        this->num_remote_send_ranks, this->h_send_rank_npart.ptr);

    auto h_send_offsets = this->particle_packer.h_send_offsets.ptr;
    auto h_recv_buffer = this->particle_unpacker.h_recv_buffer.ptr;
    auto h_recv_offsets = this->particle_unpacker.h_recv_offsets.ptr;
    const int num_bytes_per_particle =
        this->particle_packer.num_bytes_per_particle;

    NESOASSERT(this->particle_packer.num_bytes_per_particle ==
                   this->particle_unpacker.num_bytes_per_particle,
               "packer and unpacker disagree on size of a particle");

    // non-blocking recv of packed particle data
    this->in_flight_recvs = 0;
    for (int rankx = 0; rankx < this->num_remote_recv_ranks; rankx++) {
      const int tmp_num_recv = this->h_recv_rank_npart.ptr[rankx];
      if (tmp_num_recv > 0) {
        MPICHK(MPI_Irecv(&h_recv_buffer[h_recv_offsets[rankx]],
                         tmp_num_recv * num_bytes_per_particle, MPI_CHAR,
                         this->h_recv_ranks.ptr[rankx], 43, this->comm,
                         &this->h_recv_requests.ptr[this->in_flight_recvs++]));
      }
    }

    // non-blocking send of particle data
    this->in_flight_sends = 0;
    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {
      const int tmp_num_send = this->h_send_rank_npart.ptr[rankx];
      if (tmp_num_send > 0) {
        MPICHK(MPI_Isend(&h_send_buffer[h_send_offsets[rankx]],
                         tmp_num_send * num_bytes_per_particle, MPI_CHAR,
                         this->h_send_ranks.ptr[rankx], 43, this->comm,
                         &this->h_send_requests.ptr[this->in_flight_sends++]));
      }
    }
  }

  inline void exchange_finalise() {
    MPICHK(MPI_Waitall(this->in_flight_recvs, this->h_recv_requests.ptr,
                       this->h_status.ptr));

    MPICHK(MPI_Waitall(this->in_flight_sends, this->h_send_requests.ptr,
                       this->h_status.ptr));
  }

  inline void move() {
    auto t0 = profile_timestamp();

    // find particles leaving through the local interface
    this->departing_identify.identify(1);

    // reset the packer
    this->particle_packer.reset();

    for (int rankx = 0; rankx < this->num_remote_send_ranks; rankx++) {
      const int rank = this->h_send_ranks.ptr[rankx];
      this->h_send_rank_npart.ptr[rankx] =
          this->departing_identify.dh_send_counts_all_ranks.h_buffer.ptr[rank];
    }

    const int num_particles_leaving =
        this->departing_identify.dh_num_particle_send.h_buffer.ptr[0];

    // start the packing
    auto global_pack_event = this->particle_packer.pack(
        this->num_remote_send_ranks, this->h_send_rank_npart,
        this->dh_send_rank_map, num_particles_leaving,
        this->departing_identify.d_pack_cells,
        this->departing_identify.d_pack_layers_src,
        this->departing_identify.d_pack_layers_dst, this->particle_dats_real,
        this->particle_dats_int);

    // exchange the send/recv counts with neighbours
    this->npart_exchange_sendrecv();

    // allocate space to recv packed particles
    particle_unpacker.reset(this->num_remote_recv_ranks,
                            this->h_recv_rank_npart, this->particle_dats_real,
                            this->particle_dats_int);

    // wait for the local particles to be packed.
    global_pack_event.wait_and_throw();

    // send and recv packed particles from particle_packer.cell_dat to
    // particle_unpacker.h_recv_buffer using h_recv_offsets.

    // start exchange particles
    this->exchange_init();

    // remove the sent particles whilst the communication occurs
    this->layer_compressor.remove_particles(
        num_particles_leaving, this->departing_identify.d_pack_cells.ptr,
        this->departing_identify.d_pack_layers_src.ptr);

    // finalise exchange particles
    this->exchange_finalise();

    // Unpack the recv'd particles
    this->particle_unpacker.unpack(particle_dats_real, particle_dats_int);

    sycl_target.profile_map.inc("LocalMove", "Move", 1,
                                profile_elapsed(t0, profile_timestamp()));
  };
};

} // namespace NESO::Particles

#endif
