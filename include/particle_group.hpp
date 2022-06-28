#ifndef _NESO_PARTICLES_PARTICLE_GROUP
#define _NESO_PARTICLES_PARTICLE_GROUP

#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include "access.hpp"
#include "cell_dat.hpp"
#include "compute_target.hpp"
#include "domain.hpp"
#include "particle_dat.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

class ParticleGroup {
private:
  int ncell;
  int npart_local;
  BufferShared<INT> npart_cell;

  BufferDevice<INT> compress_cells_old;
  BufferDevice<INT> compress_cells_new;
  BufferDevice<INT> compress_layers_old;
  BufferDevice<INT> compress_layers_new;

  // these should be INT not int but hipsycl refused to do atomic refs on long
  // int
  BufferDevice<int> device_npart_cell;
  BufferDevice<int> device_move_counters;

  inline void compute_remove_compress_indicies(const int npart,
                                               const std::vector<INT> &cells,
                                               const std::vector<INT> &layers);
  template <typename T> inline void realloc_dat(ParticleDatShPtr<T> &dat) {
    dat->realloc(this->npart_cell);
  };
  template <typename T> inline void push_particle_spec(ParticleProp<T> prop) {
    this->particle_spec.push(prop);
  };

  // members for mpi communication
  BufferShared<int> s_send_ranks;
  BufferShared<int> s_recv_ranks;
  BufferShared<int> s_send_offsets;
  BufferShared<int> s_recv_offsets;
  BufferShared<int> s_num_ranks_send;
  BufferShared<int> s_num_ranks_recv;
  BufferShared<int> s_npart_send_recv;
  BufferShared<int> s_pack_cells;
  BufferShared<int> s_pack_layers_src;
  BufferShared<int> s_pack_layers_dst;
  // pack/unpack buffers
  BufferDevice<char> d_packing;
  BufferHost<char> h_packing;

public:
  Domain domain;
  SYCLTarget &sycl_target;

  std::map<Sym<REAL>, ParticleDatShPtr<REAL>> particle_dats_real{};
  std::map<Sym<INT>, ParticleDatShPtr<INT>> particle_dats_int{};

  // ParticleDat storing Positions
  std::shared_ptr<Sym<REAL>> position_sym;
  ParticleDatShPtr<REAL> position_dat;
  // ParticleDat storing cell ids
  std::shared_ptr<Sym<INT>> cell_id_sym;
  ParticleDatShPtr<INT> cell_id_dat;
  // ParticleDat storing MPI rank
  std::shared_ptr<Sym<INT>> mpi_rank_sym;
  ParticleDatShPtr<INT> mpi_rank_dat;

  // ParticleSpec of all the ParticleDats of this ParticleGroup
  ParticleSpec particle_spec;

  ParticleGroup(Domain domain, ParticleSpec &particle_spec,
                SYCLTarget &sycl_target)
      : domain(domain), sycl_target(sycl_target),
        ncell(domain.mesh.get_cell_count()), compress_cells_old(sycl_target, 1),
        compress_cells_new(sycl_target, 1), compress_layers_old(sycl_target, 1),
        compress_layers_new(sycl_target, 1), s_send_ranks(sycl_target, 1),
        s_recv_ranks(sycl_target, 1), s_send_offsets(sycl_target, 1),
        s_recv_offsets(sycl_target, 1), s_num_ranks_send(sycl_target, 1),
        s_num_ranks_recv(sycl_target, 1), s_pack_cells(sycl_target, 1),
        s_pack_layers_src(sycl_target, 1), s_pack_layers_dst(sycl_target, 1),
        s_npart_send_recv(sycl_target, 1), d_packing(sycl_target, 1),
        h_packing(sycl_target, 1), npart_cell(sycl_target, 1),
        device_npart_cell(sycl_target, domain.mesh.get_cell_count()),
        device_move_counters(sycl_target, domain.mesh.get_cell_count()) {

    this->npart_local = 0;
    this->npart_cell.realloc_no_copy(domain.mesh.get_cell_count());

    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->npart_cell.ptr[cellx] = 0;
    }
    for (auto &property : particle_spec.properties_real) {
      add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
    }
    for (auto &property : particle_spec.properties_int) {
      add_particle_dat(ParticleDat(sycl_target, property, this->ncell));
    }
    // Create a ParticleDat to store the MPI rank of the particles in.
    mpi_rank_sym = std::make_shared<Sym<INT>>("NESO_MPI_RANK");
    mpi_rank_dat =
        ParticleDat(sycl_target, ParticleProp(*mpi_rank_sym, 1), ncell);
    add_particle_dat(mpi_rank_dat);
  }
  ~ParticleGroup() {}

  inline void add_particle_dat(ParticleDatShPtr<REAL> particle_dat);
  inline void add_particle_dat(ParticleDatShPtr<INT> particle_dat);

  inline void add_particles();
  template <typename U> inline void add_particles(U particle_data);
  inline void add_particles_local(ParticleSet &particle_data);

  inline int get_npart_local() { return this->npart_local; }

  inline ParticleDatShPtr<REAL> &operator[](Sym<REAL> sym) {
    return this->particle_dats_real.at(sym);
  };
  inline ParticleDatShPtr<INT> &operator[](Sym<INT> sym) {
    return this->particle_dats_int.at(sym);
  };

  inline CellData<REAL> get_cell(Sym<REAL> sym, const int cell) {
    return particle_dats_real[sym]->cell_dat.get_cell(cell);
  }

  inline CellData<INT> get_cell(Sym<INT> sym, const int cell) {
    return particle_dats_int[sym]->cell_dat.get_cell(cell);
  }

  inline void remove_particles(const int npart, const std::vector<INT> &cells,
                               const std::vector<INT> &layers);

  inline INT get_npart_cell(const int cell) {
    return this->npart_cell.ptr[cell];
  }
  inline ParticleSpec &get_particle_spec() { return this->particle_spec; }
  inline void global_move();
  inline INT get_particle_loop_iter_range() {
    return this->domain.mesh.get_cell_count() *
           this->position_dat->cell_dat.get_nrow_max();
  }
  inline INT get_particle_loop_cell_stride() {
    return this->position_dat->cell_dat.get_nrow_max();
  }
  inline INT *get_particle_loop_npart_cell() { return this->npart_cell.ptr; }
  /*
   * Number of bytes required to store the data for one particle.
   */
  inline size_t particle_size();
};

inline void
ParticleGroup::add_particle_dat(ParticleDatShPtr<REAL> particle_dat) {
  this->particle_dats_real[particle_dat->sym] = particle_dat;
  // Does this dat hold particle positions?
  if (particle_dat->positions) {
    this->position_dat = particle_dat;
    this->position_sym = std::make_shared<Sym<REAL>>(particle_dat->sym.name);
  }
  realloc_dat(particle_dat);
  // TODO clean up this ParticleProp handling
  push_particle_spec(ParticleProp(particle_dat->sym, particle_dat->ncomp,
                                  particle_dat->positions));
}
inline void
ParticleGroup::add_particle_dat(ParticleDatShPtr<INT> particle_dat) {
  this->particle_dats_int[particle_dat->sym] = particle_dat;
  // Does this dat hold particle cell ids?
  if (particle_dat->positions) {
    this->cell_id_dat = particle_dat;
    this->cell_id_sym = std::make_shared<Sym<INT>>(particle_dat->sym.name);
  }
  realloc_dat(particle_dat);
  // TODO clean up this ParticleProp handling
  push_particle_spec(ParticleProp(particle_dat->sym, particle_dat->ncomp,
                                  particle_dat->positions));
}

inline void ParticleGroup::add_particles(){};
template <typename U>
inline void ParticleGroup::add_particles(U particle_data){

};

/*
 * Number of bytes required to store the data for one particle.
 */
inline size_t ParticleGroup::particle_size() {
  size_t s = 0;
  for (auto &dat : this->particle_dats_real) {
    s += dat.second->cell_dat.row_size();
  }
  for (auto &dat : this->particle_dats_int) {
    s += dat.second->cell_dat.row_size();
  }
  return s;
};

inline void ParticleGroup::add_particles_local(ParticleSet &particle_data) {
  // loop over the cells of the new particles and allocate more space in the
  // dats

  const int npart = particle_data.npart;
  const int npart_new = this->npart_local + npart;
  auto cellids = particle_data.get(*this->cell_id_sym);
  std::vector<INT> layers(npart_new);
  for (int px = 0; px < npart_new; px++) {
    auto cellindex = cellids[px];
    NESOASSERT((cellindex >= 0) && (cellindex < this->ncell),
               "Bad particle cellid)");

    layers[px] = this->npart_cell.ptr[cellindex]++;
  }

  for (auto &dat : this->particle_dats_real) {
    realloc_dat(dat.second);
    dat.second->append_particle_data(npart, particle_data.contains(dat.first),
                                     cellids, layers,
                                     particle_data.get(dat.first));
  }

  for (auto &dat : this->particle_dats_int) {
    realloc_dat(dat.second);
    dat.second->append_particle_data(npart, particle_data.contains(dat.first),
                                     cellids, layers,
                                     particle_data.get(dat.first));
  }

  this->npart_local = npart_new;

  // The append is async
  this->sycl_target.queue.wait();
}

inline void ParticleGroup::compute_remove_compress_indicies(
    const int npart, const std::vector<INT> &cells,
    const std::vector<INT> &layers) {

  compress_cells_old.realloc_no_copy(npart);
  compress_layers_old.realloc_no_copy(npart);
  compress_layers_new.realloc_no_copy(npart);
  const auto cell_count = domain.mesh.get_cell_count();
  NESOASSERT(cell_count <= this->npart_cell.size,
             "bad buffer lengths on cell count");
  device_npart_cell.realloc_no_copy(cell_count);

  auto npart_cell_ptr = npart_cell.ptr;
  sycl::buffer<INT> b_cells(cells.data(), sycl::range<1>{cells.size()});
  sycl::buffer<INT> b_layers(layers.data(), sycl::range<1>{layers.size()});

  auto device_npart_cell_ptr = device_npart_cell.ptr;
  auto device_move_counters_ptr = device_move_counters.ptr;

  auto compress_cells_old_ptr = compress_cells_old.ptr;
  auto compress_layers_old_ptr = compress_layers_old.ptr;
  auto compress_layers_new_ptr = compress_layers_new.ptr;

  INT ***cell_ids_ptr = this->cell_id_dat->cell_dat.device_ptr();
  this->sycl_target.queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(static_cast<size_t>(cell_count)),
                           [=](sycl::id<1> idx) {
                             device_npart_cell_ptr[idx] =
                                 static_cast<int>(npart_cell_ptr[idx]);
                             device_move_counters_ptr[idx] = 0;
                           });
      })
      .wait();

  this->sycl_target.queue
      .submit([&](sycl::handler &cgh) {
        auto a_cells = b_cells.get_access<sycl::access::mode::read>(cgh);
        auto a_layers = b_layers.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for<>(sycl::range<1>(static_cast<size_t>(npart)),
                           [=](sycl::id<1> idx) {
                             const auto cell = a_cells[idx];
                             const auto layer = a_layers[idx];

                             // Atomically do device_npart_cell_ptr[cell]--
                             sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                              sycl::memory_scope::device>
                                 element_atomic(device_npart_cell_ptr[cell]);
                             element_atomic.fetch_add(-1);

                             //// indicate this particle is removed by setting
                             /// the / cell index to -1
                             cell_ids_ptr[cell][0][layer] = -42;
                           });
      })
      .wait();

  this->sycl_target.queue
      .submit([&](sycl::handler &cgh) {
        auto a_cells = b_cells.get_access<sycl::access::mode::read>(cgh);
        auto a_layers = b_layers.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<>(
            sycl::range<1>(static_cast<size_t>(npart)), [=](sycl::id<1> idx) {
              const auto cell = a_cells[idx];
              const auto layer = a_layers[idx];

              // Is this layer less than the new cell count?
              // If so then there is a particle in a row greater than the cell
              // count to be copied into this layer.
              if (layer < device_npart_cell_ptr[cell]) {

                // If there are n rows to be filled in row indices less than the
                // new cell count then there are n rows greater than the new
                // cell count which are to be copied down. Atomically compute
                // which one of those rows this index copies.
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    element_atomic(device_move_counters_ptr[cell]);
                const INT source_row_offset = element_atomic.fetch_add(1);

                // find the source row counting up from new_cell_count to
                // old_cell_count potential source rows will have a cell index
                // >= 0. This index should pick the i^th source row where i was
                // computed from the atomic above.
                INT found_count = 0;
                INT source_row = -1;
                for (INT rowx = device_npart_cell_ptr[cell];
                     rowx < npart_cell_ptr[cell]; rowx++) {
                  // Is this a potential source row?
                  if (cell_ids_ptr[cell][0][rowx] > -1) {
                    if (source_row_offset == found_count++) {
                      source_row = rowx;
                      break;
                    }
                  }
                }
                compress_cells_old_ptr[idx] = cell;
                compress_layers_new_ptr[idx] = layer;
                compress_layers_old_ptr[idx] = source_row;

              } else {

                compress_cells_old_ptr[idx] = -1;
                compress_layers_new_ptr[idx] = -1;
                compress_layers_old_ptr[idx] = -1;
              }
            });
      })
      .wait_and_throw();
}

inline void ParticleGroup::remove_particles(const int npart,
                                            const std::vector<INT> &cells,
                                            const std::vector<INT> &layers) {

  compute_remove_compress_indicies(npart, cells, layers);

  auto compress_cells_old_ptr = compress_cells_old.ptr;
  auto compress_layers_old_ptr = compress_layers_old.ptr;
  auto compress_layers_new_ptr = compress_layers_new.ptr;
  for (auto &dat : particle_dats_real) {
    dat.second->copy_particle_data(
        npart, compress_cells_old_ptr, compress_cells_old_ptr,
        compress_layers_old_ptr, compress_layers_new_ptr);
    dat.second->set_npart_cells_device(device_npart_cell.ptr);
  }
  for (auto &dat : particle_dats_int) {
    dat.second->copy_particle_data(
        npart, compress_cells_old_ptr, compress_cells_old_ptr,
        compress_layers_old_ptr, compress_layers_new_ptr);
    dat.second->set_npart_cells_device(device_npart_cell.ptr);
  }

  auto npart_cell_ptr = npart_cell.ptr;
  auto device_npart_cell_ptr = device_npart_cell.ptr;
  const auto cell_count = domain.mesh.get_cell_count();
  this->sycl_target.queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(
        sycl::range<1>(static_cast<size_t>(cell_count)), [=](sycl::id<1> idx) {
          npart_cell_ptr[idx] = static_cast<INT>(device_npart_cell_ptr[idx]);
        });
  });

  // the move calls are async
  sycl_target.queue.wait_and_throw();

  for (auto &dat : particle_dats_real) {
    dat.second->trim_cell_dat_rows();
  }
  for (auto &dat : particle_dats_int) {
    dat.second->trim_cell_dat_rows();
  }
}

/*
 * Perform global move operation to send particles to the MPI ranks stored in
 * the NESO_MPI_RANK ParticleDat. Must be called collectively on the
 * communicator.
 */
inline void ParticleGroup::global_move() {

  const int comm_size = sycl_target.comm_pair.size_parent;
  const int comm_rank = sycl_target.comm_pair.rank_parent;
  s_send_ranks.realloc_no_copy(comm_size);
  s_recv_ranks.realloc_no_copy(comm_size);

  auto pl_iter_range = this->get_particle_loop_iter_range();
  auto pl_stride = this->get_particle_loop_cell_stride();
  auto pl_npart_cell = this->get_particle_loop_npart_cell();

  s_pack_cells.realloc_no_copy(pl_iter_range);
  s_pack_layers_src.realloc_no_copy(pl_iter_range);
  s_pack_layers_dst.realloc_no_copy(pl_iter_range);

  auto s_send_ranks_ptr = s_send_ranks.ptr;
  auto s_recv_ranks_ptr = s_recv_ranks.ptr;
  auto s_num_ranks_send_ptr = s_num_ranks_send.ptr;
  auto s_num_ranks_recv_ptr = s_num_ranks_recv.ptr;
  // auto s_recv_offsets_ptr = s_recv_offsets.ptr;
  auto s_pack_cells_ptr = s_pack_cells.ptr;
  auto s_pack_layers_src_ptr = s_pack_layers_src.ptr;
  auto s_pack_layers_dst_ptr = s_pack_layers_dst.ptr;
  auto s_npart_send_recv_ptr = s_npart_send_recv.ptr;

  // zero the send/recv counts
  this->sycl_target.queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(sycl::range<1>(comm_size), [=](sycl::id<1> idx) {
      s_send_ranks_ptr[idx] = 0;
      s_recv_ranks_ptr[idx] = 0;
    });
  });
  // zero the number of ranks involved with send/recv
  this->sycl_target.queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<>([=]() {
      s_num_ranks_send_ptr[0] = 0;
      s_num_ranks_recv_ptr[0] = 0;
      s_npart_send_recv_ptr[0] = 0;
    });
  });
  sycl_target.queue.wait_and_throw();
  // loop over all particles - for leaving particles atomically compute the
  // packing layer by incrementing the send count for the report rank and
  // increment the counter for the number of remote ranks to send to
  const INT INT_comm_size = static_cast<INT>(comm_size);
  const INT INT_comm_rank = static_cast<INT>(comm_rank);
  auto d_neso_mpi_rank = this->mpi_rank_dat->cell_dat.device_ptr();

  this->sycl_target.queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          const INT cellx = ((INT)idx) / pl_stride;
          const INT layerx = ((INT)idx) % pl_stride;

          if (layerx < pl_npart_cell[cellx]) {
            const INT owning_rank = d_neso_mpi_rank[cellx][0][layerx];

            // if rank is valid and not equal to this rank then this particle is
            // being sent somewhere
            if (((owning_rank >= 0) && (owning_rank < INT_comm_size)) &&
                (owning_rank != INT_comm_rank)) {
              // Increment the counter for the remote rank
              // reuse the recv ranks array to avoid mallocing more space
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device>
                  pack_layer_atomic(s_recv_ranks_ptr[owning_rank]);
              const int pack_layer = pack_layer_atomic.fetch_add(1);

              // increment the counter for number of sent particles (globally)
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device>
                  send_count_atomic(s_npart_send_recv_ptr[0]);
              const int send_index = send_count_atomic.fetch_add(1);

              // store the cell, source layer, packing layer
              s_pack_cells_ptr[send_index] = static_cast<int>(cellx);
              s_pack_layers_src_ptr[send_index] = static_cast<int>(layerx);
              s_pack_layers_dst_ptr[send_index] = pack_layer;

              // if the packing layer is zero then this is the first particle
              // found sending to the remote rank -> increment the number of
              // remote ranks and record this rank.
              if (pack_layer == 0) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    num_ranks_send_atomic(s_num_ranks_send_ptr[0]);
                const int rank_index = num_ranks_send_atomic.fetch_add(1);
                s_send_ranks_ptr[rank_index] = static_cast<int>(owning_rank);
              }
            }
          }
        });
      })
      .wait_and_throw();

  // The send offsets are the offsets in the packing buffer for each remote
  // rank. These offsets are computed using a cumulative sum over the send
  // counts for each of the remote ranks (where the send count is non-zero).
  const int num_remote_send_ranks = s_num_ranks_send_ptr[0];
  s_send_offsets.realloc_no_copy(num_remote_send_ranks + 1);
  auto s_send_offsets_ptr = s_send_offsets.ptr;
  s_send_offsets_ptr[0] = 0;
  for (int rx = 0; rx < num_remote_send_ranks; rx++) {
    const int rank = s_send_ranks_ptr[rx];
    const int to_send_count = s_recv_ranks_ptr[rank];
    s_send_offsets_ptr[rx + 1] = s_send_offsets_ptr[rx] + to_send_count;
  }

  const int num_particles_leaving = s_npart_send_recv_ptr[0];
  // allocate enough space to pack all particles to send
  const size_t send_nbytes = num_particles_leaving * this->particle_size();
  d_packing.realloc_no_copy(send_nbytes);
  h_packing.realloc_no_copy(send_nbytes);

  // We now have:
  // 1) n particles to send
  // 2) array length n of source cells
  // 3) array length n of source layers (the rows in the source cells)
  // 4) array length n of packing layers (the index in the packing buffer for
  //    each particle to send)
  // 5) m destination MPI ranks to send to
  // 6) array length m of destination ranks
  // 7) array length m+1 where entry i is the start of the allocated space to
  //    pack particle data to send to send_ranks[i].
}

} // namespace NESO::Particles

#endif
