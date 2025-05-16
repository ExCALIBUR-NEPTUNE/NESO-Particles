#ifndef _NESO_PARTICLES_QUADRATURE_POINT_MAPPER_H_
#define _NESO_PARTICLES_QUADRATURE_POINT_MAPPER_H_
#include "../../communication/communication_edges_counter.hpp"
#include "../../compute_target.hpp"
#include "../../containers/tuple.hpp"
#include "../../particle_group_impl.hpp"
#include "../../particle_io.hpp"
#include "../../typedefs.hpp"
#include <map>
#include <stack>
#include <vector>

namespace NESO::Particles::ExternalCommon {

/**
 * Type to aid deposition and evaluation to external libraries. We assume that
 * the external library provides a set of points at which function samples
 * should be provided, for deposition, and at which function evaluations can be
 * read, for evaluation.
 *
 * Points must be added before calling other methods. To add points
 * add_points_initialise must be called before calling add_point for each
 * quadrature point. Finally, after all quadrature points have been added,
 * add_points_finalise must be called.
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

  void make_contribs_buffers(int ncomp_local, int ncomp_remote);

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
      NESOASSERT(static_cast<std::size_t>(offset) <= data.size(),
                 "Pointer arithmetic exceed alloced buffer.");
    };

    lambda_compute_offsets(this->adding_indices_count, adding,
                           this->adding_ptrs);
    lambda_compute_offsets(this->owning_indices_count, owning,
                           this->owning_ptrs);
  }

public:
  /// The compute device and communicator for the points.
  SYCLTargetSharedPtr sycl_target;
  /// The domain on which the quadrature points exist.
  DomainSharedPtr domain;
  /// The internal particle group that stores the quadrature points.
  ParticleGroupSharedPtr particle_group;

  QuadraturePointMapper() = default;

  /**
   * Create a quadrature point mapper from a compute device and a domain.
   *
   * @param sycl_target Compute device (and MPI communicator).
   * @param domain Domain which contains the quadrature points.
   */
  QuadraturePointMapper(SYCLTargetSharedPtr sycl_target,
                        DomainSharedPtr domain);

  /**
   * Start the procedure to add points to the quadrature point mapper. This
   * method must be called and must be called collectively on the communicator.
   */
  inline void add_points_initialise() {
    NESOASSERT(this->added_points.empty(),
               "Point creation cannot occur more than once.");
  }

  /**
   * Add a quadrature point to the mapper. This method must be called for each
   * quadrature point. This method does not need to be called collectively on
   * the communicator.
   */
  void add_point(const REAL *point);

  /**
   * End the procedure to add quadrature points. This method must be called and
   * must be called collectively on the communicator.
   */
  void add_points_finalise();

  /**
   * @returns the Sym that corresponds to the ParticleDat for a given number of
   * components.
   */
  Sym<REAL> get_sym(const int ncomp);

  /**
   * Get the internal representation evaluated at the added points.
   *
   * @param[in] ncomp The number of components of the representation to get.
   * @param[in, out] output Output vector for evaluations.
   */
  void get(const int ncomp, std::vector<REAL> &output);

  /**
   * Set the evaluations at the points.
   *
   * @param ncomp The number of components at each evaluation point to set.
   * @param input The point evaluations to set.
   */
  void set(const int ncomp, std::vector<REAL> &input);

  /**
   * Free the data structure. This must be called and must be called
   * collectively on the communicator.
   */
  void free();

  /**
   * @returns true if points have been added to the mapper.
   */
  bool points_added();

  /**
   * Write the quadrature points to disk.
   *
   * @param filename Filename for h5part file. Should have the extension h5part.
   */
  void write_to_disk(std::string filename);
};

typedef std::shared_ptr<QuadraturePointMapper> QuadraturePointMapperSharedPtr;

namespace {
template <std::size_t N> struct InterpTupleType {};

template <> struct InterpTupleType<1> {
  using type = Tuple::Tuple<REAL>;
};
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
