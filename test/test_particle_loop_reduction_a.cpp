#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>
#include <type_traits>

using namespace NESO::Particles;

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common(const int N = 10093) {
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;

  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 4),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int rank = sycl_target->comm_pair.rank_parent;
  const INT id_offset = rank * N;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);

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
    initial_distribution[Sym<INT>("ID")][px][0] = px + id_offset;
  }

  A->add_particles_local(initial_distribution);
  parallel_advection_initialisation(A, 16);

  auto ccb = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, A->position_dat, A->cell_id_dat);

  ccb->execute();
  A->cell_move();

  return A;
}

} // namespace

namespace NESO::Particles {
namespace Access {

template <typename T> struct IsReduction {
  using flag = std::false_type;
};
template <typename T, typename OP> struct IsReduction<Reduction<T, OP>> {
  using flag = std::true_type;
};
template <typename... ARGS> struct HasReduction {
  static constexpr bool value{(
      std::is_same_v<std::true_type, typename IsReduction<ARGS>::flag> || ...)};
};

} // namespace Access

template <typename KERNEL, typename... ARGS>
class ParticleLoopReduction : public ParticleLoop<KERNEL, ARGS...> {};

// TODO
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop_TODO(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                   ARGS... args) {

  if constexpr (Access::HasReduction<ARGS...>::value) {
    nprint("Contains reduction.");
    return particle_loop(particle_group, kernel, args...);
    return nullptr;
  } else {
    return particle_loop(particle_group, kernel, args...);
  }
}
} // namespace NESO::Particles

TEST(ParticleLoopReduction, base) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto Vto_test =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 3, 1);

  particle_loop_TODO(
      A,
      [=](auto INDEX) {

      },
      Access::read(ParticleLoopIndex{}))
      ->execute();

  auto loop1 = particle_loop_TODO(
      A,
      [=](auto INDEX, auto CDC) {

      },
      Access::read(ParticleLoopIndex{}),
      Access::reduce(Vto_test, Kernel::plus<REAL>{}));

  auto reduce = Access::reduce(Vto_test, Kernel::plus<REAL>{});
  auto read = Access::read(Vto_test);

  auto foo = ParticleLoopImplementation::LoopParameter<decltype(reduce)>{};
  auto bar = ParticleLoopImplementation::KernelParameter<decltype(reduce)>{};

  A->free();
  sycl_target->free();
  mesh->free();
}
