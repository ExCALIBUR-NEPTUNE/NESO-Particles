#ifndef _NESO_PARTICLES_LOOP_PARTICLE_LOOP_ARGS_HPP_
#define _NESO_PARTICLES_LOOP_PARTICLE_LOOP_ARGS_HPP_

#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "../cell_dat.hpp"
#include "../compute_target.hpp"
#include "../containers/descendant_products.hpp"
#include "../containers/global_array.hpp"
#include "../containers/local_array.hpp"
#include "../containers/local_memory_block.hpp"
#include "../containers/local_memory_interlaced.hpp"
#include "../containers/nd_local_array.hpp"
#include "../containers/particle_set_device.hpp"
#include "../containers/product_matrix.hpp"
#include "../containers/rng/kernel_rng.hpp"
#include "../containers/sym_vector.hpp"
#include "../containers/sym_vector_impl.hpp"
#include "../containers/tuple.hpp"
#include "../particle_dat.hpp"
#include "../particle_spec.hpp"
#include "../sycl_typedefs.hpp"
#include "../typedefs.hpp"
#include "cell_info_npart.hpp"
#include "kernel.hpp"
#include "particle_loop_base.hpp"
#include "particle_loop_index.hpp"
#include "pli_particle_dat.hpp"

namespace NESO::Particles {

#ifdef NESO_PARTICLES_PARTICLE_LOOP_ARGS_STATS

struct ParticleLoopArgsStats {
  std::map<std::vector<std::size_t>, int> indices_to_count;
  std::map<std::type_index, std::size_t> arg_to_index;
  std::map<std::size_t, std::string> index_to_name;

  std::size_t index{0};
  inline std::size_t get_index(std::type_index t) {
    if (this->arg_to_index.count(t)) {
      return this->arg_to_index.at(t);
    } else {
      const std::size_t c = index++;
      this->arg_to_index[t] = c;
      return c;
    }
  }

  ParticleLoopArgsStats() {}
  ~ParticleLoopArgsStats() {

    std::vector<std::vector<std::size_t>> keys;
    for (auto &ix : this->indices_to_count) {
      keys.push_back(ix.first);
    }

    std::vector<std::size_t> idx(this->indices_to_count.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t i, std::size_t j) {
      return this->indices_to_count.at(keys.at(i)) <
             this->indices_to_count.at(keys.at(j));
    });

    // Find a unique filename and write an empty file to stop other ranks
    // finding the same name.
    std::size_t file_index = 0;
    std::FILE *fh = nullptr;
    std::string filename;
    do {
      filename =
          "particle_loop_args_stats_" + std::to_string(file_index++) + ".json";
      fh = fopen(filename.c_str(), "wx");
    } while (fh == nullptr);
    std::fclose(fh);

    // Write the output json
    std::ofstream fs;
    fs.open(filename, std::ios::out);
    fs << "{" << std::endl;
    fs << "\t\"index_to_mangled_names\" : {" << std::endl;
    for (std::size_t ix = 0; ix < this->index - 1; ix++) {
      fs << "\"" << std::to_string(ix) << "\" : \""
         << this->index_to_name.at(ix) << "\"," << std::endl;
    }
    fs << "\"" << std::to_string(this->index - 1) << "\" : \""
       << this->index_to_name.at(this->index - 1) << "\"" << std::endl;
    fs << "\t}," << std::endl;

    fs << "\t\"loop_count_to_args_count\" : [" << std::endl;

    for (std::size_t xx = 0; xx < idx.size() - 1; xx++) {
      fs << "\t\t[";
      std::size_t ix = idx.at(xx);
      auto &key = keys.at(ix);
      for (auto kx : key) {
        fs << std::to_string(kx) << ", ";
      }
      fs << std::to_string(this->indices_to_count.at(key));
      fs << "]," << std::endl;
    }
    fs << "\t\t[";
    std::size_t ix = idx.at(idx.size() - 1);
    auto &key = keys.at(ix);
    for (auto kx : key) {
      fs << std::to_string(kx) << ", ";
    }
    fs << std::to_string(this->indices_to_count.at(key));
    fs << "]" << std::endl;

    fs << "\t]" << std::endl;
    fs << "}" << std::endl;
  }
};

extern ParticleLoopArgsStats particle_loop_args_stats;

template <typename ARG>
inline void
particle_loop_args_stats_increment_inner(std::vector<std::size_t> &key, ARG &) {
  const std::size_t arg_index = particle_loop_args_stats.get_index(typeid(ARG));
  key.push_back(arg_index);
  particle_loop_args_stats.index_to_name[arg_index] = typeid(ARG).name();
}

template <typename... ARGS>
inline void particle_loop_args_stats_increment(ARGS... args) {
  std::vector<std::size_t> key;
  (particle_loop_args_stats_increment_inner(key, args), ...);
  particle_loop_args_stats.indices_to_count[key]++;
}

#endif

class ParticleSubGroup;

namespace ParticleLoopImplementation {
/**
 * Catch all for args passed as shared ptrs
 */
template <template <typename> typename T, typename U, typename V>
struct LoopParameter<T<std::shared_ptr<U>>, V> {
  using type = typename LoopParameter<T<U>, V>::type;
};
/**
 * Catch Access::Reduction around shared pointers.
 */
template <template <typename> typename T, typename U, typename OP>
struct LoopParameter<Access::Reduction<std::shared_ptr<T<U>>, OP>> {
  using type = typename LoopParameter<Access::Reduction<T<U>, OP>>::type;
};
/**
 * Catch all for args passed as shared ptrs
 */
template <template <typename> typename T, typename U, typename V>
struct KernelParameter<T<std::shared_ptr<U>>, V> {
  using type = typename KernelParameter<T<U>, V>::type;
};
/**
 * Catch Access::Reduction around shared pointers.
 */
template <template <typename> typename T, typename U, typename OP>
struct KernelParameter<Access::Reduction<std::shared_ptr<T<U>>, OP>> {
  using type = typename KernelParameter<Access::Reduction<T<U>, OP>>::type;
};

} // namespace ParticleLoopImplementation

namespace {

/**
 *  This is a metafunction which is passed a input data type and returns the
 *  LoopParamter type that corresponds to the input type (by using the structs
 * defined above).
 */
template <class T>
using loop_parameter_t =
    typename ParticleLoopImplementation::LoopParameter<T, std::true_type>::type;

/**
 *  Function to map access descriptor and data structure type to kernel type.
 */
template <class T>
using kernel_parameter_t =
    typename ParticleLoopImplementation::KernelParameter<T,
                                                         std::true_type>::type;

template <template <typename> typename T, typename U>
inline void check_is_sym_inner([[maybe_unused]] T<U> arg) {
  static_assert(std::is_same<T<U>, Sym<U>>::value == false,
                "Sym based arguments cannot be passed to ParticleLoop with a "
                "ParticleDat iterator. Pass the ParticleDatSharedPtr instead.");
}
template <typename T> inline void check_is_sym_inner([[maybe_unused]] T arg) {}
template <template <typename> typename T, typename U>
inline void check_is_sym_outer(T<U> arg) {
  check_is_sym_inner(arg.obj);
}

} // namespace

/**
 * This class describes how the loop arguments should behave.
 *
 * * Methods that discuss "loop" arguments are executed host side.
 * * Methods that describe "kernel" arguments are executed device side and must
 *   meet the sycl requirements for a device function.
 */
template <typename... ARGS> class ParticleLoopArgs {
protected:
  /**
   * Method to compute access to a type wrapped in a shared_ptr.
   */
  template <template <typename> typename T, typename U>
  inline auto create_loop_arg_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, T<std::shared_ptr<U>> a);
  /**
   * Method to compute access to a type not wrapper in a shared_ptr
   */
  template <template <typename> typename T, typename U>
  inline auto create_loop_arg_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, T<U> a);

  /**
   * Method to compute access to a Reduction type.
   */
  template <template <typename> typename T, typename U, typename OP>
  inline auto create_loop_arg_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Reduction<std::shared_ptr<T<U>>, OP> a);

  /**
   * Method to compute access to a type wrapped in a shared_ptr.
   */
  template <template <typename> typename T, typename U>
  static inline auto post_loop_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      T<std::shared_ptr<U>> a);
  /**
   * Method to compute access to a type not wrapper in a shared_ptr
   */
  template <template <typename> typename T, typename U>
  static inline auto post_loop_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info, T<U> a);

  /**
   * Pre loop cast for reduction access
   */
  template <template <typename> typename T, typename U, typename OP>
  static inline void post_loop_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      Access::Reduction<std::shared_ptr<T<U>>, OP> a);

  /**
   * Method to compute access to a type wrapped in a shared_ptr.
   */
  template <template <typename> typename T, typename U>
  inline void
  pre_loop_cast(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                T<std::shared_ptr<U>> a);
  /**
   * Method to compute access to a type not wrapper in a shared_ptr
   */
  template <template <typename> typename T, typename U>
  inline void
  pre_loop_cast(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                T<U> a);

  /**
   * Pre loop cast for reduction access
   */
  template <template <typename> typename T, typename U, typename OP>
  inline void
  pre_loop_cast(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                Access::Reduction<std::shared_ptr<T<U>>, OP> a);

  /**
   * Method to compute access to a type wrapped in a shared_ptr.
   */
  template <template <typename> typename T, typename U>
  static inline std::size_t local_mem_loop_cast(T<std::shared_ptr<U>> a) {
    T<U *> c = {a.obj.get()};
    return ParticleLoopImplementation::get_required_local_num_bytes(c);
  }
  /**
   * Method to compute access to a Reduction type wrapped in a shared_ptr.
   */
  template <template <typename> typename T, typename U, typename OP>
  static inline std::size_t
  local_mem_loop_cast(Access::Reduction<std::shared_ptr<T<U>>, OP> a) {
    return ParticleLoopImplementation::get_required_local_num_bytes(a);
  }
  /**
   * Method to compute access to a type not wrapper in a shared_ptr
   */
  template <template <typename> typename T, typename U>
  static inline std::size_t local_mem_loop_cast(T<U> a) {
    T<U *> c = {&a.obj};
    return ParticleLoopImplementation::get_required_local_num_bytes(c);
  }

  /// The types of the parameters for the outside loops.
  using loop_parameter_type = Tuple::Tuple<loop_parameter_t<ARGS>...>;
  /// The types of the arguments passed to the kernel.
  using kernel_parameter_type = Tuple::Tuple<kernel_parameter_t<ARGS>...>;
  /// Tuple of the arguments passed to the ParticleLoop on construction.
  std::tuple<ARGS...> args;

  /// Recursively assemble the tuple args.
  template <size_t INDEX, typename U> inline void unpack_args(U a0) {
    std::get<INDEX>(this->args) = a0;
  }
  template <size_t INDEX, typename U, typename... V>
  inline void unpack_args(U a0, V... args) {
    std::get<INDEX>(this->args) = a0;
    this->unpack_args<INDEX + 1>(args...);
  }

  /// Recursively assemble the outer loop arguments.
  template <size_t INDEX, size_t SIZE, typename PARAM>
  inline void create_loop_args_inner(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, PARAM &loop_args) {
    if constexpr (INDEX < SIZE) {
      Tuple::get<INDEX>(loop_args) =
          create_loop_arg_cast(global_info, cgh, std::get<INDEX>(this->args));
      create_loop_args_inner<INDEX + 1, SIZE>(global_info, cgh, loop_args);
    }
  }
  inline void create_loop_args(
      sycl::handler &cgh, loop_parameter_type &loop_args,
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) {
    create_loop_args_inner<0, sizeof...(ARGS)>(global_info, cgh, loop_args);
  }

  inline void apply_pre_loop(
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info) {
    auto cast_wrapper = [&](auto t) { pre_loop_cast(&global_info, t); };
    auto pre_loop_caller = [&](auto... as) { (cast_wrapper(as), ...); };
    std::apply(pre_loop_caller, this->args);
  }

  inline std::size_t get_local_size_args(SYCLTargetSharedPtr sycl_target,
                                         std::string name) {

    // Loop over the args and add how many local bytes they each require.
    std::size_t num_bytes = 0;
    auto lambda_size_add = [&](auto argx) {
      num_bytes += this->local_mem_loop_cast(argx);
    };
    auto lambda_size = [&](auto... as) { (lambda_size_add(as), ...); };
    std::apply(lambda_size, this->args);

    // The amount of local space on the device and required number of local
    // bytes gives an upper bound on local size.
    std::size_t local_size =
        sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;
    local_size = sycl_target->get_num_local_work_items(num_bytes, local_size);

    sycl_target->profile_map.set("ParticleLoop::" + name, "local_size",
                                 local_size, 0.0);
    return local_size;
  }

  ParticleLoopArgs(ARGS... args) {
    this->unpack_args<0>(args...);
#ifdef NESO_PARTICLES_PARTICLE_LOOP_ARGS_STATS
    particle_loop_args_stats_increment(args...);
#endif
  }
};

} // namespace NESO::Particles

#endif
