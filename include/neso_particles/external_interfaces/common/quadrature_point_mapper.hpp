#ifndef _NESO_PARTICLES_QUADRATURE_POINT_MAPPER_H_
#define _NESO_PARTICLES_QUADRATURE_POINT_MAPPER_H_
#include "../../communication/communication_edges_counter.hpp"
#include "../../compute_target.hpp"
#include "../../containers/cell_dat_const.hpp"
#include "../../containers/tuple.hpp"
#include "../../particle_group_impl.hpp"
#include "../../particle_io.hpp"
#include "../../typedefs.hpp"
#include <map>
#include <stack>
#include <tuple>
#include <vector>

namespace NESO::Particles::ExternalCommon {

/**
 * TODO
 */
class QuadraturePointMapper {
protected:
  int ndim;
  int rank;
  int size;
  bool internal_points_added;

  struct Point {
    REAL coords[3];
    int adding_rank;
    int adding_index;
  };
  std::stack<Point> added_points;
  std::unique_ptr<CommunicationEdgesCounter> cec;
  std::unique_ptr<BufferDevice<REAL>> d_local_contribs;
  std::unique_ptr<BufferDevice<REAL>> d_remote_contribs;

  int npoint_local;
  int npoint_remote;

  std::map<int, std::pair<int, int>> map_recv_rank_start_end;

  // The ranks that added points this rank now owns.
  std::vector<int> adding_ranks;
  // The number of points that correspond to each rank.
  std::vector<std::int64_t> adding_indices_count;
  std::vector<std::int64_t> adding_num_bytes;
  std::vector<void *> adding_ptrs;

  // The ranks that now hold points this rank added
  std::vector<int> owning_ranks;
  // The number of points this rank added that are now on the remote rank.
  std::vector<std::int64_t> owning_indices_count;
  std::vector<std::int64_t> owning_num_bytes;
  std::vector<void *> owning_ptrs;
  std::vector<int> owning_point_indices;

  std::vector<REAL> recv_buffer;

  inline void make_contribs_buffers(int ncomp_local, int ncomp_remote) {
    ncomp_local = std::max(ncomp_local, 1);
    ncomp_remote = std::max(ncomp_remote, 1);

    if (!this->d_local_contribs) {
      this->d_local_contribs =
          std::make_unique<BufferDevice<REAL>>(this->sycl_target, ncomp_local);
    }
    this->d_local_contribs->realloc_no_copy(ncomp_local);
    if (!this->d_remote_contribs) {
      this->d_remote_contribs =
          std::make_unique<BufferDevice<REAL>>(this->sycl_target, ncomp_remote);
    }
    this->d_remote_contribs->realloc_no_copy(ncomp_remote);
  }

  template <typename T>
  inline std::pair<int, int> compute_num_bytes(const int ncomp) {
    auto lambda_realloc = [&](const auto &counts, auto &bytes) -> int {
      const int num_entries = counts.size();
      int count = 0;
      for (int ix = 0; ix < num_entries; ix++) {
        count += counts.at(ix);
        bytes.at(ix) =
            counts.at(ix) * static_cast<std::int64_t>(sizeof(T)) * ncomp;
      }
      return count;
    };
    return {lambda_realloc(this->adding_indices_count, this->adding_num_bytes),
            lambda_realloc(this->owning_indices_count, this->owning_num_bytes)};
  }

  template <typename T>
  inline void compute_offsets(const int ncomp, std::vector<T> &adding,
                              std::vector<T> &owning) {
    NESOASSERT(this->adding_ptrs.size() == this->adding_ranks.size(),
               "size missmatch");
    NESOASSERT(this->owning_ptrs.size() == this->owning_ranks.size(),
               "size missmatch");

    auto lambda_compute_offsets = [&](auto &counts, auto &data, auto &ptrs) {
      const int num_entries = counts.size();
      int offset = 0;
      for (int ix = 0; ix < num_entries; ix++) {
        const int count = counts.at(ix) * ncomp;
        // This is a T* hence the pointer arithmetic is for type T
        ptrs.at(ix) = static_cast<void *>(data.data() + offset);
        offset += count;
      }
      NESOASSERT(offset <= data.size(),
                 "Pointer arithmetic exceed alloced buffer.");
    };

    lambda_compute_offsets(this->adding_indices_count, adding,
                           this->adding_ptrs);
    lambda_compute_offsets(this->owning_indices_count, owning,
                           this->owning_ptrs);
  }

public:
  SYCLTargetSharedPtr sycl_target;
  DomainSharedPtr domain;
  ParticleGroupSharedPtr particle_group;

  QuadraturePointMapper() = default;

  /**
   * TODO
   */
  QuadraturePointMapper(SYCLTargetSharedPtr sycl_target, DomainSharedPtr domain)
      : sycl_target(sycl_target), domain(domain),
        ndim(domain->mesh->get_ndim()),
        rank(sycl_target->comm_pair.rank_parent),
        size(sycl_target->comm_pair.size_parent), internal_points_added(false) {
    NESOASSERT((0 < ndim) && (ndim < 4), "Bad number of dimensions.");
  }

  /**
   * TODO
   */
  inline void add_points_initialise() {
    NESOASSERT(this->added_points.empty(),
               "Point creation cannot occur more than once.");
  }

  /**
   * TODO
   */
  inline void add_point(const REAL *point) {
    Point p;
    p.adding_rank = this->rank;
    p.adding_index = this->added_points.size();
    for (int dx = 0; dx < this->ndim; dx++) {
      p.coords[dx] = point[dx];
    }
    this->added_points.push(p);
  }

  /**
   * TODO
   * Collective
   */
  inline void add_points_finalise() {
    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<INT>("ADDING_RANK_INDEX"), 4)};

    this->particle_group = std::make_shared<ParticleGroup>(
        this->domain, particle_spec, this->sycl_target);

    // Add the quadrature points as particles into the particle group.
    ParticleSet initial_distribution(this->added_points.size(), particle_spec);
    int index = 0;
    while (!this->added_points.empty()) {
      const auto t = this->added_points.top();
      initial_distribution[Sym<INT>("ADDING_RANK_INDEX")][index][0] =
          t.adding_rank;
      initial_distribution[Sym<INT>("ADDING_RANK_INDEX")][index][1] =
          t.adding_index;
      for (int dx = 0; dx < this->ndim; dx++) {
        initial_distribution[Sym<REAL>("P")][index][dx] = t.coords[dx];
      }
      this->added_points.pop();
      index++;
    }
    this->particle_group->add_particles_local(initial_distribution);
    this->npoint_local = index;
    this->particle_group->hybrid_move();
    this->particle_group->cell_move();

    // Communicate to the rank which added a quadrature point which rank now
    // holds the point.
    struct RemotePoint {
      int adding_rank;
      int adding_index;
      int cell;
      int layer;
      int write_index;
    };

    // Get on the host the remotely added quadrature points
    auto la_remote_indices = std::make_shared<LocalArray<RemotePoint>>(
        this->sycl_target, this->particle_group->get_npart_local());
    auto la_remote_indices_counter =
        std::make_shared<LocalArray<int>>(this->sycl_target, 1);
    la_remote_indices_counter->fill(0);
    auto la_tmp_index = std::make_shared<LocalArray<int>>(
        this->sycl_target, this->particle_group->get_npart_local());

    const int k_rank = this->rank;
    particle_loop(
        "QuadraturePointMapper::add_points_finalise_0", this->particle_group,
        [=](auto INDEX, auto ADDING_RANK_INDEX, auto REMOTE_INDICES,
            auto REMOTE_INDICES_COUNTER, auto TMP_INDEX) {
          // Was this point added by a remote rank?
          if (ADDING_RANK_INDEX.at(0) != k_rank) {
            const int tmp_index = REMOTE_INDICES_COUNTER.fetch_add(0, 1);
            RemotePoint p;
            p.adding_rank = ADDING_RANK_INDEX.at(0);
            p.adding_index = ADDING_RANK_INDEX.at(1);
            p.cell = INDEX.cell;
            p.layer = INDEX.layer;
            REMOTE_INDICES.at(tmp_index) = p;
            TMP_INDEX.at(INDEX.get_local_linear_index()) = tmp_index;
          }
        },
        Access::read(ParticleLoopIndex{}),
        Access::read(Sym<INT>("ADDING_RANK_INDEX")),
        Access::write(la_remote_indices),
        Access::add(la_remote_indices_counter), Access::write(la_tmp_index))
        ->execute();

    const int num_remote_points = la_remote_indices_counter->get().at(0);
    this->npoint_remote = num_remote_points;

    std::vector<RemotePoint> remote_points;
    remote_points.reserve(num_remote_points);
    auto la_points = la_remote_indices->get();
    for (int ix = 0; ix < num_remote_points; ix++) {
      remote_points.push_back(la_points.at(ix));
    }

    // Group the remote points by rank and determine an ordering such that
    // the "particles" can write their contribution into a contiguous block
    // for each remote rank.

    // setup the communication patterns
    std::map<int, std::vector<int>> map_ranks_to_adding_index;
    for (auto rp : remote_points) {
      map_ranks_to_adding_index[rp.adding_rank].push_back(rp.adding_index);
    }

    // The ranks that added points this rank now holds
    this->adding_ranks.reserve(map_ranks_to_adding_index.size());
    // The number of points this rank holds from that the remote rank added.
    this->adding_indices_count.reserve(map_ranks_to_adding_index.size());

    for (auto rp : map_ranks_to_adding_index) {
      adding_ranks.push_back(rp.first);
      adding_indices_count.push_back(rp.second.size());
    }

    // For each remote rank compute the start/end points in the packing buffer
    std::vector<int> adding_write_starts(adding_ranks.size());
    std::vector<int> adding_write_ends(adding_ranks.size());
    std::exclusive_scan(adding_indices_count.begin(),
                        adding_indices_count.end(), adding_write_starts.begin(),
                        0);
    for (int ix = 0; ix < adding_ranks.size(); ix++) {
      adding_write_ends.at(ix) =
          adding_write_starts.at(ix) + adding_indices_count.at(ix);
    }

    // For each rank (local or remote) compute the index in the packing buffer
    // each particle should write it's contribution to
    const int num_adding_ranks = adding_ranks.size();
    std::vector<int> rank_starts(this->size);
    for (int rx = 0; rx < num_adding_ranks; rx++) {
      const int rankx = adding_ranks.at(rx);
      rank_starts.at(rankx) = adding_write_starts.at(rx);
    }
    auto la_rank_starts =
        std::make_shared<LocalArray<int>>(this->sycl_target, rank_starts);
    particle_loop(
        "QuadraturePointMapper::add_points_finalise_1", this->particle_group,
        [=](auto INDEX, auto ADDING_RANK_INDEX, auto RANK_STARTS,
            auto TMP_INDEX, auto REMOTE_INDICES) {
          // Was this point added by a remote rank?
          const int rank = ADDING_RANK_INDEX.at(0);
          if (rank != k_rank) {
            const int write_index = RANK_STARTS.fetch_add(rank, 1);
            const auto tmp_index = TMP_INDEX.at(INDEX.get_local_linear_index());
            REMOTE_INDICES.at(tmp_index).write_index = write_index;
            ADDING_RANK_INDEX.at(2) = write_index;
          } else {
            ADDING_RANK_INDEX.at(2) = ADDING_RANK_INDEX.at(1);
          }
        },
        Access::read(ParticleLoopIndex{}),
        Access::write(Sym<INT>("ADDING_RANK_INDEX")),
        Access::add(la_rank_starts), Access::read(la_tmp_index),
        Access::write(la_remote_indices))
        ->execute();

    // re-get these structs to have the write indices
    remote_points.clear();
    remote_points.reserve(num_remote_points);
    la_remote_indices->get(la_points);
    la_remote_indices = nullptr;
    for (int ix = 0; ix < num_remote_points; ix++) {
      remote_points.push_back(la_points.at(ix));
    }
    la_points.clear();

    // Check the data structure after the indices are computed matches the
    // start end points we computed.
    const auto post_rank_starts = la_rank_starts->get();
    la_rank_starts = nullptr;
    for (int rx = 0; rx < num_adding_ranks; rx++) {
      const int rankx = adding_ranks.at(rx);
      NESOASSERT(adding_write_ends.at(rx) == post_rank_starts.at(rankx),
                 "Mismatch between the ends computed on the host and the index "
                 "computation on device");
      this->map_recv_rank_start_end[rankx] = {adding_write_starts.at(rx),
                                              adding_write_ends.at(rx)};
    }

    // Connect the remote points with the write indices for each rank
    this->cec = std::make_unique<CommunicationEdgesCounter>(
        this->sycl_target->comm_pair.comm_parent);

    this->cec->get_remote_ranks(this->adding_ranks, this->adding_indices_count,
                                this->owning_ranks, this->owning_indices_count);

    auto la_remote_point_indices =
        std::make_shared<LocalArray<int>>(this->sycl_target, num_remote_points);

    particle_loop(
        "QuadraturePointMapper::add_points_finalise_2", this->particle_group,
        [=](auto ADDING_RANK_INDEX, auto REMOTE_POINT_INDICES) {
          if (ADDING_RANK_INDEX.at(0) != k_rank) {
            const int remote_index = ADDING_RANK_INDEX.at(1);
            const int index = ADDING_RANK_INDEX.at(2);
            REMOTE_POINT_INDICES.at(index) = remote_index;
          }
        },
        Access::read(Sym<INT>("ADDING_RANK_INDEX")),
        Access::write(la_remote_point_indices))
        ->execute();

    auto adding_point_indices = la_remote_point_indices->get();
    la_remote_point_indices = nullptr;

    this->adding_ptrs.resize(this->adding_ranks.size());
    this->owning_ptrs.resize(this->owning_ranks.size());
    this->adding_num_bytes.resize(this->adding_ranks.size());
    this->owning_num_bytes.resize(this->owning_ranks.size());

    const auto num_entries = this->compute_num_bytes<int>(1);
    NESOASSERT(num_entries.first == adding_point_indices.size(),
               "Num entries mismatch.");
    this->owning_point_indices = std::vector<int>(num_entries.second);
    this->compute_offsets<int>(1, adding_point_indices,
                               this->owning_point_indices);

    this->cec->exchange_send_recv_data(
        this->adding_ranks, this->adding_num_bytes, this->adding_ptrs,
        this->owning_ranks, this->owning_num_bytes, this->owning_ptrs);

    // the vector owning_point_indices should now contain the point indices for
    // points which were added on this rank but the location is owned by a
    // remote rank

    // Compute an ordering for the locally added points in each cell for
    // evaluation.
    auto la_ordering = std::make_shared<LocalArray<int>>(
        this->sycl_target, this->domain->mesh->get_cell_count());
    la_ordering->fill(0);

    particle_loop(
        "QuadraturePointMapper::add_points_finalise_3", this->particle_group,
        [=](auto INDEX, auto ADDING_RANK_INDEX, auto ORDERING) {
          if (ADDING_RANK_INDEX.at(0) == k_rank) {
            const int cell = INDEX.cell;
            const int tmp_index = ORDERING.fetch_add(cell, 1);
            ADDING_RANK_INDEX.at(3) = tmp_index;
          } else {
            ADDING_RANK_INDEX.at(3) = -1;
          }
        },
        Access::read(ParticleLoopIndex{}),
        Access::write(Sym<INT>("ADDING_RANK_INDEX")), Access::add(la_ordering))
        ->execute();

    this->internal_points_added = true;
  }

  /**
   * TODO
   */
  inline Sym<REAL> get_sym(const int ncomp) {
    NESOASSERT((0 < ncomp) && (ncomp < 1000), "Bad number of components.");
    std::string name = "contrib_" + std::to_string(ncomp);
    auto sym = Sym<REAL>(name);
    if (!this->particle_group->contains_dat(sym)) {
      this->particle_group->add_particle_dat(
          ParticleDat(this->sycl_target, ParticleProp(sym, ncomp),
                      this->domain->mesh->get_cell_count()));
    }
    return sym;
  }

  /**
   * TODO
   */
  inline void get(const int ncomp, std::vector<REAL> &output) {
    auto sym = this->get_sym(ncomp);

    const int ncomp_local = this->npoint_local * ncomp;
    const int ncomp_remote = this->npoint_remote * ncomp;
    this->make_contribs_buffers(ncomp_local, ncomp_remote);

    const int k_rank = this->rank;
    const int k_ncomp = ncomp;
    auto k_local = this->d_local_contribs->ptr;
    auto k_remote = this->d_remote_contribs->ptr;

    particle_loop(
        "QuadraturePointMapper::get", this->particle_group,
        [=](auto ADDING_RANK_INDEX, auto CONTRIB) {
          const int start = ADDING_RANK_INDEX.at(2) * k_ncomp;
          if (ADDING_RANK_INDEX.at(0) == k_rank) {
            for (int cx = 0; cx < k_ncomp; cx++) {
              k_local[start + cx] = CONTRIB.at(cx);
            }
          } else {
            for (int cx = 0; cx < k_ncomp; cx++) {
              k_remote[start + cx] = CONTRIB.at(cx);
            }
          }
        },
        Access::read(Sym<INT>("ADDING_RANK_INDEX")), Access::read(sym))
        ->execute();

    if (output.size() != ncomp_local) {
      output.resize(ncomp_local);
    }

    auto e0 = this->sycl_target->queue.memcpy(
        output.data(), this->d_local_contribs->ptr, ncomp_local * sizeof(REAL));

    std::vector<REAL> send_buffer(ncomp_remote);
    auto e1 = this->sycl_target->queue.memcpy(send_buffer.data(),
                                              this->d_remote_contribs->ptr,
                                              ncomp_remote * sizeof(REAL));

    // Need to exchange and populate the points where the evaluation point is
    // not owned by the rank which added the point.
    auto sizes = this->compute_num_bytes<REAL>(ncomp);
    this->recv_buffer.resize(sizes.second);
    e1.wait_and_throw();
    this->compute_offsets<REAL>(ncomp, send_buffer, recv_buffer);

    // actually exchange the data
    this->cec->exchange_send_recv_data(
        this->adding_ranks, this->adding_num_bytes, this->adding_ptrs,
        this->owning_ranks, this->owning_num_bytes, this->owning_ptrs);

    // copy the recieved data into the right index
    const int num_entries = this->owning_point_indices.size();
    e0.wait_and_throw();
    for (int ix = 0; ix < num_entries; ix++) {
      const int dst_index = this->owning_point_indices.at(ix);
      for (int cx = 0; cx < ncomp; cx++) {
        output.at(dst_index * ncomp + cx) =
            this->recv_buffer.at(ix * ncomp + cx);
      }
    }
  }

  /**
   * TODO
   */
  inline void set(const int ncomp, std::vector<REAL> &input) {
    auto sym = this->get_sym(ncomp);
    const int ncomp_local = this->npoint_local * ncomp;
    NESOASSERT(input.size() >= ncomp_local, "Input vector is too small.");
    this->make_contribs_buffers(ncomp_local, 0);

    auto k_local = this->d_local_contribs->ptr;
    this->sycl_target->queue
        .memcpy(k_local, input.data(), ncomp_local * sizeof(REAL))
        .wait_and_throw();

    const int k_rank = this->rank;
    const int k_ncomp = ncomp;
    particle_loop(
        "QuadraturePointMapper::set", this->particle_group,
        [=](auto ADDING_RANK_INDEX, auto CONTRIB) {
          const int start = ADDING_RANK_INDEX.at(2) * k_ncomp;
          if (ADDING_RANK_INDEX.at(0) == k_rank) {
            for (int cx = 0; cx < k_ncomp; cx++) {
              CONTRIB.at(cx) = k_local[start + cx];
            }
          }
        },
        Access::read(Sym<INT>("ADDING_RANK_INDEX")), Access::write(sym))
        ->execute();
  }

  /**
   * TODO
   */
  inline void free() {
    if (this->particle_group) {
      this->particle_group->free();
      this->particle_group = nullptr;
    }
    if (this->cec) {
      this->cec->free();
      this->cec = nullptr;
    }
  }
  /**
   * @returns true if points have been added to the mapper.
   */
  inline bool points_added() { return this->internal_points_added; }

  /**
   * TODO collective
   */
  inline void write_to_disk(std::string filename) {
    H5Part h5part(filename, this->particle_group, Sym<REAL>("P"),
                  Sym<INT>("CELL_ID"), Sym<INT>("ADDING_RANK_INDEX"));
    h5part.write();
    h5part.close();
  }
};

typedef std::shared_ptr<QuadraturePointMapper> QuadraturePointMapperSharedPtr;

namespace {
template <std::size_t N> struct InterpTupleType {};

template <> struct InterpTupleType<1> { using type = Tuple::Tuple<REAL>; };
template <> struct InterpTupleType<2> {
  using type = Tuple::Tuple<REAL, REAL>;
};
template <> struct InterpTupleType<3> {
  using type = Tuple::Tuple<REAL, REAL, REAL>;
};

template <typename PTYPE>
inline void assemble_args(PTYPE &P, Tuple::Tuple<REAL> &args) {
  Tuple::get<0>(args) = P.at(0);
}
template <typename PTYPE>
inline void assemble_args(PTYPE &P, Tuple::Tuple<REAL, REAL> &args) {
  Tuple::get<0>(args) = P.at(0);
  Tuple::get<1>(args) = P.at(1);
}
template <typename PTYPE>
inline void assemble_args(PTYPE &P, Tuple::Tuple<REAL, REAL, REAL> &args) {
  Tuple::get<0>(args) = P.at(0);
  Tuple::get<1>(args) = P.at(1);
  Tuple::get<2>(args) = P.at(2);
}

template <typename... ARGS, typename DST_DAT, typename FUNC_TOP>
inline void apply_args_inner(const int index, Tuple::Tuple<ARGS...> &args,
                             DST_DAT &dst, FUNC_TOP func_top) {
  dst.at(index) = Tuple::apply(func_top, args);
}
template <typename... ARGS, typename DST_DAT, typename FUNC_TOP,
          typename... FUNC>
inline void apply_args_inner(int index, Tuple::Tuple<ARGS...> &args,
                             DST_DAT &dst, FUNC_TOP func_top, FUNC... func) {
  dst.at(index) = Tuple::apply(func_top, args);
  index++;
  apply_args_inner(index, args, dst, func...);
}
template <typename... ARGS, typename DST_DAT, typename... FUNC>
inline void apply_args(Tuple::Tuple<ARGS...> &args, DST_DAT &dst,
                       FUNC... func) {
  apply_args_inner(0, args, dst, func...);
}

} // namespace

/**
 * Helper function to evaluate functions at the points in a
 * QuadraturePointMapper.
 *
 * @param qpm QuadraturePointMapper to evaluate functions at.
 * @param func Parameter pack with function per component. These functions must
 * be device copyable types (i.e. not function pointers). Functions wrapped in
 * a lambda function should work.
 */
template <std::size_t NDIM, std::size_t NCOMP, typename... FUNC>
inline void interpolate(QuadraturePointMapperSharedPtr qpm, FUNC... func) {

  static_assert((0 < NDIM) && (NDIM < 4), "Only implemented for NDIM=1,2,3");
  NESOASSERT(NDIM == qpm->particle_group->domain->mesh->get_ndim(),
             "Number of dimensions missmatch.");

  static_assert(
      sizeof...(FUNC) == NCOMP,
      "Number of passed functions does not match the number of components.");

  particle_loop(
      qpm->particle_group,
      [=](auto POS, auto Q) {
        typename InterpTupleType<NDIM>::type args;
        assemble_args(POS, args);
        apply_args(args, Q, func...);
      },
      Access::read(qpm->particle_group->position_dat),
      Access::write(qpm->get_sym(NCOMP)))
      ->execute();
}

/**
 * Helper function to evaluate functions at the points in a
 * QuadraturePointMapper.
 *
 * @param qpm QuadraturePointMapper to evaluate functions at.
 * @param ncomp Number of components of the space to interpolate into.
 * @param component Component of the space to interpolate into.
 * @param func Function to interpolate. This function must be a device copyable
 * type (i.e. not function pointers). Functions wrapped in a lambda function
 * should work.
 */
template <std::size_t NDIM, typename FUNC>
inline void interpolate(QuadraturePointMapperSharedPtr qpm, const int ncomp,
                        const int component, FUNC func) {

  static_assert((0 < NDIM) && (NDIM < 4), "Only implemented for NDIM=1,2,3");
  NESOASSERT(NDIM == qpm->particle_group->domain->mesh->get_ndim(),
             "Number of dimensions missmatch.");

  particle_loop(
      qpm->particle_group,
      [=](auto POS, auto Q) {
        typename InterpTupleType<NDIM>::type args;
        assemble_args(POS, args);
        Q.at(component) = Tuple::apply(func, args);
      },
      Access::read(qpm->particle_group->position_dat),
      Access::write(qpm->get_sym(ncomp)))
      ->execute();
}

} // namespace NESO::Particles::ExternalCommon

#endif
