#ifndef _NESO_PARTICLES_PARTICLE_LOOP_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_H_

#include "../compute_target.hpp"
#include "../containers/tuple.hpp"
#include "../particle_group.hpp"
#include "../particle_spec.hpp"
#include <CL/sycl.hpp>
#include <cstdlib>
#include <typeinfo>

namespace NESO::Particles::Access {

template <typename T> struct AccessGeneric { T obj; };

template <typename T> struct Read : AccessGeneric<T> {};

template <typename T> struct Write : AccessGeneric<T> {};

template <typename T> inline Read<T> read(T t) { return Read<T>{t}; }

template <typename T> inline Write<T> write(T t) { return Write<T>{t}; }

} // namespace NESO::Particles::Access

namespace NESO::Particles::Access::ParticleDat {

template <typename T> struct Read {
  T **ptr;
  int layer;
  T &operator[](const int component) { return ptr[component][this->layer]; }
};

template <typename T> struct Write {
  T **ptr;
  int layer;
  T &operator[](const int component) { return ptr[component][this->layer]; }
};

} // namespace NESO::Particles::Access::ParticleDat

namespace NESO::Particles {

namespace {

template <typename SPEC> struct LoopParameter { using type = void *; };
template <typename SPEC> struct LoopParameter<Access::Read<Sym<SPEC>>> {
  using type = SPEC ***;
};
template <typename SPEC> struct LoopParameter<Access::Write<Sym<SPEC>>> {
  using type = SPEC ***;
};
template <class T> using loop_parameter_t = typename LoopParameter<T>::type;

template <typename SPEC> struct KernelParameter { using type = void; };
template <typename SPEC> struct KernelParameter<Access::Read<Sym<SPEC>>> {
  using type = Access::ParticleDat::Read<SPEC>;
};
template <typename SPEC> struct KernelParameter<Access::Write<Sym<SPEC>>> {
  using type = Access::ParticleDat::Write<SPEC>;
};
template <class T> using kernel_parameter_t = typename KernelParameter<T>::type;

template <typename SPEC>
inline void create_kernel_arg(const int cellx, const int layerx, SPEC ***rhs,
                              Access::ParticleDat::Read<SPEC> &lhs) {
  lhs.layer = layerx;
  lhs.ptr = rhs[cellx];
}
template <typename SPEC>
inline void create_kernel_arg(const int cellx, const int layerx, SPEC ***rhs,
                              Access::ParticleDat::Write<SPEC> &lhs) {
  lhs.layer = layerx;
  lhs.ptr = rhs[cellx];
}

} // namespace

template <typename KERNEL, typename... ARGS> class ParticleLoop {

protected:
  using loop_parameter_type = Tuple::Tuple<loop_parameter_t<ARGS>...>;
  using kernel_parameter_type = Tuple::Tuple<kernel_parameter_t<ARGS>...>;

  std::tuple<ARGS...> args;

  template <size_t INDEX, typename U> inline void unpack_args(U a0) {
    std::get<INDEX>(this->args) = a0;
  }

  template <size_t INDEX, typename U, typename... V>
  inline void unpack_args(U a0, V... args) {
    std::get<INDEX>(this->args) = a0;
    this->unpack_args<INDEX + 1>(args...);
  }

  template <typename SPEC> inline auto get_loop_arg(Access::Read<Sym<SPEC>> a) {
    auto sym = a.obj;
    return this->particle_group->get_dat(sym)->cell_dat.device_ptr();
  }

  template <typename SPEC>
  inline auto get_loop_arg(Access::Write<Sym<SPEC>> a) {
    auto sym = a.obj;
    return this->particle_group->get_dat(sym)->cell_dat.device_ptr();
  }

  template <size_t INDEX, size_t SIZE, typename PARAM>
  inline void create_loop_args_inner(PARAM &loop_args) {
    if constexpr (INDEX < SIZE) {
      Tuple::get<INDEX>(loop_args) = get_loop_arg(std::get<INDEX>(this->args));
      create_loop_args_inner<INDEX + 1, SIZE>(loop_args);
    }
  }

  inline void create_loop_args(loop_parameter_type &loop_args) {
    create_loop_args_inner<0, sizeof...(ARGS)>(loop_args);
  }

  template <size_t INDEX, size_t SIZE>
  static inline constexpr void
  create_kernel_args_inner(const int cellx, const int layerx,
                           const loop_parameter_type &loop_args,
                           kernel_parameter_type &kernel_args) {

    if constexpr (INDEX < SIZE) {
      auto arg = Tuple::get<INDEX>(loop_args);
      create_kernel_arg(cellx, layerx, arg, Tuple::get<INDEX>(kernel_args));
      create_kernel_args_inner<INDEX + 1, SIZE>(cellx, layerx, loop_args,
                                                kernel_args);
    }
  }

  static inline constexpr void
  create_kernel_args(const int cellx, const int layerx,
                     const loop_parameter_type &loop_args,
                     kernel_parameter_type &kernel_args) {

    create_kernel_args_inner<0, sizeof...(ARGS)>(cellx, layerx, loop_args,
                                                 kernel_args);
  }

  ParticleGroupSharedPtr particle_group;
  KERNEL kernel;

public:
  ParticleLoop(ParticleGroupSharedPtr particle_group, KERNEL kernel,
               ARGS... args)
      : particle_group(particle_group), kernel(kernel) {
    this->unpack_args<0>(args...);
  };

  inline void execute(EventStack &event_stack) {

    auto h_npart_cell = this->particle_group->position_dat->h_npart_cell;
    const int ncell = this->particle_group->position_dat->ncell;

    loop_parameter_type loop_args;
    create_loop_args(loop_args);
    auto k_kernel = this->kernel;

    for (int cellh = 0; cellh < ncell; cellh++) {
      const int cellx = cellh;
      const int num_particles = h_npart_cell[cellh];

      if (num_particles > 0) {
        auto iteration_set = get_nd_range_peel_1d(num_particles, 256);

        event_stack.push(this->particle_group->sycl_target->queue.submit(
            [&](sycl::handler &cgh) {
              cgh.parallel_for<>(
                  iteration_set.loop_main, [=](sycl::nd_item<1> idx) {
                    const int layerx =
                        static_cast<int>(idx.get_global_linear_id());
                    kernel_parameter_type kernel_args;
                    create_kernel_args(cellx, layerx, loop_args, kernel_args);
                    Tuple::apply(k_kernel, kernel_args);
                  });
            }));
        if (iteration_set.peel_exists) {
          const std::size_t k_offset = iteration_set.offset;
          event_stack.push(this->particle_group->sycl_target->queue.submit(
              [&](sycl::handler &cgh) {
                cgh.parallel_for<>(
                    iteration_set.loop_peel, [=](sycl::nd_item<1> idx) {
                      const size_t layers =
                          idx.get_global_linear_id() + k_offset;
                      const int layerx = static_cast<int>(layers);
                      if (layerx < num_particles) {
                        kernel_parameter_type kernel_args;
                        create_kernel_args(cellx, layerx, loop_args,
                                           kernel_args);
                        Tuple::apply(k_kernel, kernel_args);
                      }
                    });
              }));
        }
      }
    }
  }

  inline void execute() {
    EventStack es;
    this->execute(es);
    es.wait();
  }
};

} // namespace NESO::Particles

#endif
