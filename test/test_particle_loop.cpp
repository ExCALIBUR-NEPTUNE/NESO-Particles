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

TEST(ParticleLoop, Call) {
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
template <typename U> struct KernelParameter<Sym<U>> {
  using type = ParticleDatAccess<U>;
};

template <class T> using kernel_parameter_t = typename KernelParameter<T>::type;

template <typename KERNEL, typename... ARGS> class ParticleLoop {
protected:
  using kernel_parameter_types = std::tuple<kernel_parameter_t<ARGS>...>;
  std::tuple<ARGS...> args;

  template <size_t INDEX, typename U> inline void unpack_args(U a0) {
    std::get<INDEX>(this->args) = a0;
  }

  template <size_t INDEX, typename U, typename... V>
  inline void unpack_args(U a0, V... args) {
    std::get<INDEX>(this->args) = a0;
    this->unpack_args<INDEX + 1>(args...);
  }

  ParticleGroupSharedPtr particle_group;
  KERNEL kernel;

public:
  ParticleLoop(ParticleGroupSharedPtr particle_group, KERNEL kernel,
               ARGS... args)
      : particle_group(particle_group), kernel(kernel) {
    this->unpack_args<0>(args...);
  };

  inline void execute() {

    auto pl_iter_range =
        this->particle_group->position_dat->get_particle_loop_iter_range();
    auto pl_stride =
        this->particle_group->position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell =
        this->particle_group->position_dat->get_particle_loop_npart_cell();

    this->particle_group->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(pl_iter_range),
                             [=](sycl::id<1> idx) {
                               NESO_PARTICLES_KERNEL_START
                               const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                               const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                               NESO_PARTICLES_KERNEL_END
                             });
        })
        .wait_and_throw();
  }
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

  A->add_particles(initial_distribution);

  ParticleLoop particle_loop(
      A,
      [=](ParticleDatAccess<REAL> P, ParticleDatAccess<REAL> V,
          ParticleDatAccess<INT> ID) { P[0] += V[0] * 0.001; },
      Sym<REAL>("P"), Sym<REAL>("V"), Sym<INT>("ID"));

  particle_loop.execute();

  A->free();
  mesh->free();
}

/*

template <typename SPECIALISATION>
struct ParticleLoopKernel {
  inline void pre_kernel_v(sycl::handler &cgh){
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.pre_kernel(cgh);
  }
  inline void kernel_v(const INT cellx, const INT layerx) const {
    auto &underlying = static_cast<const SPECIALISATION &>(*this);
    underlying.kernel(cellx, layerx);
  }

  inline void pre_kernel(sycl::handler &cgh){
    printf("Base pre_kernel\n");
  }
  inline void kernel(const INT cellx, const INT layerx) const {
    printf("Base kernel\n");
  }
};


template <typename LAMBDA_TYPE>
struct ParticleLoopKernelLambda :
ParticleLoopKernel<ParticleLoopKernelLambda<LAMBDA_TYPE>> { LAMBDA_TYPE
lambda_kernel;

  ParticleLoopKernelLambda(LAMBDA_TYPE lambda_kernel) :
lambda_kernel(lambda_kernel) {

    lambda_kernel(-3, -3);
    this->lambda_kernel(-4, -4);
    this->kernel(-5, -5);


  };

  inline void kernel(const INT cellx, const INT layerx) const {
    lambda_kernel(cellx, layerx);
  }


};




template <typename T, typename U>
inline sycl::event particle_loop(
    ParticleDatSharedPtr<T> particle_dat,
    //ParticleLoopKernel<U> kernel
    U kernel
  ) {

  auto pl_iter_range = particle_dat->get_particle_loop_iter_range();
  auto pl_stride = particle_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell = particle_dat->get_particle_loop_npart_cell();
  auto sycl_target = particle_dat->sycl_target;


  kernel.kernel(-6, -6);
  //kernel.kernel_v(-1, -1);

  return sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        kernel.pre_kernel_v(cgh);
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          NESO_PARTICLES_KERNEL_START
          const INT cellx = NESO_PARTICLES_KERNEL_CELL;
          const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
          kernel.kernel(cellx, layerx);
          NESO_PARTICLES_KERNEL_END
        });
      });
}




TEST(ParticleLoop, Base) {

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

  auto k_P = A->get_dat(Sym<REAL>("P"))->cell_dat.device_ptr();
  auto k_V = A->get_dat(Sym<REAL>("V"))->cell_dat.device_ptr();
  auto pl_iter_range = A->position_dat->get_particle_loop_iter_range();
  auto pl_stride = A->position_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell = A->position_dat->get_particle_loop_npart_cell();

  printf("%ld, %ld\n", k_V, k_P);

  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          NESO_PARTICLES_KERNEL_START
          const INT cellx = NESO_PARTICLES_KERNEL_CELL;
          const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
          printf("T ndim %d, %ld, %ld, %d, %d, %ld, %f\n", ndim, k_V, k_P,
cellx, layerx, k_V[cellx], k_V[cellx][0][layerx]); NESO_PARTICLES_KERNEL_END
        });
      }).wait_and_throw();


  //struct Foo : ParticleLoopKernel<Foo> {
  //  inline void kernel(const INT cellx, const INT layerx) const {
  //
  //  }
  //};


  auto inner = [=](const INT cellx, const INT layerx){
      printf("ndim %d, %ld, %ld, %d, %d, %ld\n", ndim, k_V, k_P, cellx, layerx,
k_V); for(int dimx=0 ; dimx<ndim ; dimx++){
        //k_V[cellx][dimx][layerx] = 0.001 * k_P[cellx][dimx][layerx];
      }
    };

  inner(-2, -2);


  ParticleLoopKernelLambda bar(
    inner
  );

  auto e = particle_loop(A->position_dat, bar);
  e.wait_and_throw();

  A->free();
  mesh->free();
}
*/
