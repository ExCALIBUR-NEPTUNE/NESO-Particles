#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

namespace NESO::Particles::Tuple {

template <std::size_t INDEX, typename U> struct TupleImpl {
  U value;
  TupleImpl() = default;
  U &get() { return value; }
  const U &get_const() const { return value; }
};

template <size_t INDEX, typename... V> struct TupleBaseRec {
  TupleBaseRec() = default;
};

template <size_t INDEX, typename U, typename... V>
struct TupleBaseRec<INDEX, U, V...> : TupleImpl<INDEX, U>,
                                      TupleBaseRec<INDEX + 1, V...> {
  TupleBaseRec() = default;
};

template <typename U, typename... V> struct Tuple : TupleBaseRec<0, U, V...> {
  Tuple() = default;
};

template <size_t INDEX, typename T, typename... U> struct GetIndexType {
  using type = typename GetIndexType<INDEX - 1, U...>::type;
};

template <typename T, typename... U> struct GetIndexType<0, T, U...> {
  using type = T;
};

template <size_t INDEX, typename... U> auto &get(Tuple<U...> &u) {
  return static_cast<
             TupleImpl<INDEX, typename GetIndexType<INDEX, U...>::type> &>(u)
      .get();
}

template <size_t INDEX, typename... U> const auto &get(const Tuple<U...> &u) {
  return static_cast<const TupleImpl<
      INDEX, typename GetIndexType<INDEX, U...>::type> &>(u)
      .get_const();
}

} // namespace NESO::Particles::Tuple

TEST(ParticleLoop, Tuple) {
  using Tuple0 = Tuple::Tuple<int, int64_t, double>;
  static_assert(std::is_trivially_copyable<Tuple0>::value == true);
  static_assert(std::is_same<Tuple::GetIndexType<0, int, int64_t, double>::type,
                             int>::value == true);
  static_assert(std::is_same<Tuple::GetIndexType<1, int, int64_t, double>::type,
                             int64_t>::value == true);
  static_assert(std::is_same<Tuple::GetIndexType<2, int, int64_t, double>::type,
                             double>::value == true);

  Tuple0 t;

  auto t0 = static_cast<Tuple::TupleImpl<0, int> *>(&t);
  t0->value = 42;
  auto t1 = static_cast<Tuple::TupleImpl<2, double> *>(&t);
  t1->value = 3.14;

  auto to_test_0 = static_cast<Tuple::TupleImpl<0, int> *>(&t)->value;
  auto to_test_1 = static_cast<Tuple::TupleImpl<2, double> *>(&t)->value;
  EXPECT_EQ(to_test_0, 42);
  EXPECT_EQ(to_test_1, 3.14);

  Tuple::get<0>(t) = 43;
  Tuple::get<2>(t) = 3.141;

  EXPECT_EQ(Tuple::get<0>(t), 43);
  EXPECT_EQ(Tuple::get<2>(t), 3.141);
}

namespace NESO::Particles::Apply {

template <size_t...> struct IntSequence {};

template <size_t N, size_t... S> struct GenerateIntSequence {
  using type = typename GenerateIntSequence<N - 1, N - 1, S...>::type;
};

template <size_t... S> struct GenerateIntSequence<0, S...> {
  using type = IntSequence<S...>;
};

template <typename KERNEL, size_t... S, typename... ARGS>
auto apply_inner(KERNEL &kernel, IntSequence<S...>,
                 Tuple::Tuple<ARGS...> &args) {
  return kernel(Tuple::get<S>(args)...);
}

template <typename KERNEL, typename... ARGS>
auto apply(KERNEL kernel, Tuple::Tuple<ARGS...> &args) {
  return apply_inner(
      kernel, typename GenerateIntSequence<sizeof...(ARGS)>::type(), args);
}

} // namespace NESO::Particles::Apply

TEST(ParticleLoop, Apply) {
  using Tuple0 = Tuple::Tuple<int, int64_t, double>;
  Tuple0 t;
  Tuple::get<0>(t) = -42;
  Tuple::get<1>(t) = 43;
  Tuple::get<2>(t) = 3.141;

  int aa;
  int64_t bb;
  double cc;

  const int to_test = Apply::apply(
      [&](const int a, const int64_t b, const double c) {
        aa = a;
        bb = b;
        cc = c;
        return 53;
      },
      t);

  EXPECT_EQ(to_test, 53);
  EXPECT_EQ(get<0>(t), aa);
  EXPECT_EQ(get<1>(t), bb);
  EXPECT_EQ(get<2>(t), cc);
}

template <typename T> struct ParticleDatAccess {
  T **ptr;
  int layer;
  T &operator[](const int component) { return ptr[component][this->layer]; }
};

template <typename SPEC> struct KernelParameter { using type = void; };
template <typename SPEC> struct KernelParameter<Sym<SPEC>> {
  using type = ParticleDatAccess<SPEC>;
};

template <typename SPEC> struct LoopParameter { using type = void *; };
template <typename SPEC> struct LoopParameter<Sym<SPEC>> {
  using type = SPEC ***;
};

template <class T> using kernel_parameter_t = typename KernelParameter<T>::type;
template <class T> using loop_parameter_t = typename LoopParameter<T>::type;

template <typename SPEC>
inline void create_kernel_arg(const int cellx, const int layerx, SPEC ***rhs,
                              ParticleDatAccess<SPEC> &lhs) {
  lhs.layer = layerx;
  lhs.ptr = rhs[cellx];
}

template <typename KERNEL, typename... ARGS> class ParticleLoop {

protected:
  using kernel_parameter_type = Tuple::Tuple<kernel_parameter_t<ARGS>...>;
  using loop_parameter_type = Tuple::Tuple<loop_parameter_t<ARGS>...>;

  std::tuple<ARGS...> args;

  template <size_t INDEX, typename U> inline void unpack_args(U a0) {
    std::get<INDEX>(this->args) = a0;
  }

  template <size_t INDEX, typename U, typename... V>
  inline void unpack_args(U a0, V... args) {
    std::get<INDEX>(this->args) = a0;
    this->unpack_args<INDEX + 1>(args...);
  }

  template <typename SPEC> inline auto get_loop_arg(Sym<SPEC> sym) {
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

  /*
  inline void execute() {

    auto pl_iter_range =
        this->particle_group->position_dat->get_particle_loop_iter_range();
    auto pl_stride =
        this->particle_group->position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell =
        this->particle_group->position_dat->get_particle_loop_npart_cell();

    loop_parameter_type loop_args;
    create_loop_args(loop_args);

    auto k_kernel = this->kernel;

    this->particle_group->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                kernel_parameter_type kernel_args;

                create_kernel_args(cellx, layerx, loop_args, kernel_args);

                Apply::apply(k_kernel, kernel_args);

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
  }
  */


};

TEST(ParticleLoop, Base2) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto domain = std::make_shared<Domain>(mesh);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);

  const int N = 10;

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }

  A->add_particles_local(initial_distribution);

  A->print(Sym<REAL>("P"), Sym<REAL>("V"), Sym<INT>("ID"));

  ParticleLoop particle_loop(
      A,
      [=](ParticleDatAccess<REAL> P, ParticleDatAccess<REAL> V,
          ParticleDatAccess<INT> ID) {
        P[0] += V[0];
        ID[0] = -42;
      },
      Sym<REAL>("P"), Sym<REAL>("V"), Sym<INT>("ID"));

  //particle_loop.execute();

  A->print(Sym<REAL>("P"), Sym<REAL>("V"), Sym<INT>("ID"));

  A->free();
  mesh->free();
}
