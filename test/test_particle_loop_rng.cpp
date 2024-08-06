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

TEST(ParticleLoopRNG, base) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  INT count = 0;
  auto seq_lambda = [&]() -> INT { return count++; };

  const int seq_ncomp = 2;
  auto seq_kernel = host_per_particle_block_rng<INT>(seq_lambda, seq_ncomp);

  particle_loop(
      A,
      [=](auto INDEX, auto RNG, auto LOOP_INDEX) {
        LOOP_INDEX.at(0) = INDEX.get_loop_linear_index();
        LOOP_INDEX.at(1) = RNG.at(INDEX, 0);
        LOOP_INDEX.at(2) = RNG.at(INDEX, 1);
      },
      Access::read(ParticleLoopIndex{}), Access::read(seq_kernel),
      Access::write(Sym<INT>("LOOP_INDEX")))
      ->execute();

  const int npart_local = A->get_npart_local();
  for (int cx = 0; cx < cell_count; cx++) {
    auto loop_index = A->get_cell(Sym<INT>("LOOP_INDEX"), cx);
    for (int rx = 0; rx < loop_index->nrow; rx++) {
      ASSERT_EQ(loop_index->at(rx, 0), loop_index->at(rx, 1));
      ASSERT_EQ(loop_index->at(rx, 0) + npart_local, loop_index->at(rx, 2));
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopRNG, base_single_cell) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  INT count = 0;
  auto seq_lambda = [&]() -> INT { return count++; };

  const int seq_ncomp = 2;
  auto seq_kernel = host_per_particle_block_rng<INT>(seq_lambda, seq_ncomp);

  particle_loop(
      A,
      [=](auto INDEX, auto RNG, auto LOOP_INDEX) {
        LOOP_INDEX.at(0) = INDEX.get_loop_linear_index();
        LOOP_INDEX.at(1) = RNG.at(INDEX, 0);
        LOOP_INDEX.at(2) = RNG.at(INDEX, 1);
      },
      Access::read(ParticleLoopIndex{}), Access::read(seq_kernel),
      Access::write(Sym<INT>("LOOP_INDEX")))
      ->execute(cell_count - 1);

  for (int cx = 0; cx < cell_count - 1; cx++) {
    auto loop_index = A->get_cell(Sym<INT>("LOOP_INDEX"), cx);
    for (int rx = 0; rx < loop_index->nrow; rx++) {
      ASSERT_EQ(loop_index->at(rx, 0), 0);
    }
  }

  auto loop_index = A->get_cell(Sym<INT>("LOOP_INDEX"), cell_count - 1);
  for (int rx = 0; rx < loop_index->nrow; rx++) {
    ASSERT_EQ(loop_index->at(rx, 0), loop_index->at(rx, 1));
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopRNG, zero_components) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  INT count = 0;
  auto seq_lambda = [&]() -> INT { return count++; };

  auto seq_kernel = host_per_particle_block_rng<INT>(seq_lambda, 0);

  particle_loop(
      A, [=]([[maybe_unused]] auto INDEX, [[maybe_unused]] auto RNG) {},
      Access::read(ParticleLoopIndex{}), Access::read(seq_kernel))
      ->execute();

  // This sampling function should never actually be called.
  ASSERT_EQ(count, 0);

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopRNG, uniform) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng_state(52234234 + rank);
  const REAL a = 10.0;
  const REAL b = 20.0;
  std::uniform_real_distribution<> rng_dist(a, b);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng_state); };

  const int rng_ncomp = 3;
  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, rng_ncomp);

  particle_loop(
      A,
      [=](auto INDEX, auto RNG, auto V) {
        for (int cx = 0; cx < rng_ncomp; cx++) {
          V.at(cx) = RNG.at(INDEX, cx);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng_kernel),
      Access::write(Sym<REAL>("V")))
      ->execute();

  for (int cx = 0; cx < cell_count; cx++) {
    auto V = A->get_cell(Sym<REAL>("V"), cx);
    for (int rx = 0; rx < V->nrow; rx++) {
      for (int vx = 0; vx < 3; vx++) {
        ASSERT_TRUE((a <= V->at(rx, vx)) && (V->at(rx, vx) <= b));
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopRNG, uniform_sub_group) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng_state(52234234 + rank);
  const REAL a = 10.0;
  const REAL b = 20.0;
  std::uniform_real_distribution<> rng_dist(a, b);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng_state); };

  const int rng_ncomp = 3;
  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, rng_ncomp);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  auto zeroer = particle_loop(
      A,
      [=](auto V) {
        for (int cx = 0; cx < 3; cx++) {
          V.at(cx) = 0.0;
        }
      },
      Access::write(Sym<REAL>("V")));
  zeroer->execute();

  auto assigner = particle_loop(
      aa,
      [=](auto INDEX, auto RNG, auto V) {
        for (int cx = 0; cx < rng_ncomp; cx++) {
          V.at(cx) = RNG.at(INDEX, cx);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng_kernel),
      Access::write(Sym<REAL>("V")));
  assigner->execute();

  for (int cx = 0; cx < cell_count; cx++) {
    auto ID = A->get_cell(Sym<INT>("ID"), cx);
    auto V = A->get_cell(Sym<REAL>("V"), cx);
    for (int rx = 0; rx < V->nrow; rx++) {
      for (int vx = 0; vx < 3; vx++) {
        if (ID->at(rx, 0) % 2 == 0) {
          ASSERT_TRUE((a <= V->at(rx, vx)) && (V->at(rx, vx) <= b));
        } else {
          ASSERT_NEAR(V->at(rx, vx), 0, 1.0e-15);
        }
      }
    }
  }

  zeroer->execute();
  const int cm1 = cell_count - 1;
  assigner->execute(cm1);

  for (int cx = 0; cx < cell_count; cx++) {
    auto ID = A->get_cell(Sym<INT>("ID"), cx);
    auto V = A->get_cell(Sym<REAL>("V"), cx);
    for (int rx = 0; rx < V->nrow; rx++) {
      for (int vx = 0; vx < 3; vx++) {
        if ((ID->at(rx, 0) % 2 == 0) && (cx == cm1)) {
          ASSERT_TRUE((a <= V->at(rx, vx)) && (V->at(rx, vx) <= b));
        } else {
          ASSERT_NEAR(V->at(rx, vx), 0, 1.0e-15);
        }
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopRNG, uniform_atomic_block) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;
  const int rank = sycl_target->comm_pair.rank_parent;

  auto zeroer = particle_loop(
      A,
      [=](auto V) {
        for (int cx = 0; cx < 3; cx++) {
          V.at(cx) = 0.0;
        }
      },
      Access::write(Sym<REAL>("V")));
  zeroer->execute();

  std::mt19937 rng_state(52234234 + rank);
  const REAL a = 10.0;
  const REAL b = 20.0;
  std::uniform_real_distribution<> rng_dist(a, b);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng_state); };
  const int rng_ncomp = 3;

  // KernelRNG is the type common to all device rngs.
  // AtomicBlockRNG is the type that has a buffer with an atomic index.
  // REAL is the value type.
  std::shared_ptr<KernelRNG<AtomicBlockRNG<REAL>>> rng_device_kernel =
      host_atomic_block_kernel_rng<REAL>(rng_lambda, rng_ncomp);

  particle_loop(
      A,
      [=](auto INDEX, Access::KernelRNG::Read<AtomicBlockRNG<REAL>> RNG,
          auto V) {
        for (int cx = 0; cx < rng_ncomp; cx++) {
          V.at(cx) = RNG.at(INDEX, cx);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng_device_kernel),
      Access::write(Sym<REAL>("V")))
      ->execute();

  ASSERT_TRUE(rng_device_kernel->valid_internal_state());

  for (int cx = 0; cx < cell_count; cx++) {
    auto V = A->get_cell(Sym<REAL>("V"), cx);
    for (int rx = 0; rx < V->nrow; rx++) {
      for (int vx = 0; vx < 3; vx++) {
        ASSERT_TRUE((a <= V->at(rx, vx)) && (V->at(rx, vx) <= b));
      }
    }
  }

  auto cast_rng_device_kernel =
      std::dynamic_pointer_cast<HostAtomicBlockKernelRNG<REAL>>(
          rng_device_kernel);
  ASSERT_TRUE(cast_rng_device_kernel != nullptr);
  cast_rng_device_kernel->suppress_warnings = true;

  particle_loop(
      A,
      [=](auto INDEX, auto RNG, auto V) {
        for (int cx = 0; cx < rng_ncomp; cx++) {
          V.at(cx) = RNG.at(INDEX, cx);
        }
        for (int cx = 0; cx < rng_ncomp; cx++) {
          V.at(cx) = RNG.at(INDEX, cx);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng_device_kernel),
      Access::write(Sym<REAL>("V")))
      ->execute();

  ASSERT_FALSE(rng_device_kernel->valid_internal_state());

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoopRNG, uniform_atomic_block_sampler) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;
  const int rank = sycl_target->comm_pair.rank_parent;

  auto zeroer = particle_loop(
      A,
      [=](auto V) {
        for (int cx = 0; cx < 3; cx++) {
          V.at(cx) = 0.0;
        }
      },
      Access::write(Sym<REAL>("V")));
  zeroer->execute();

  int count = 0;
  auto rng_lambda = [&]() -> int { return count++; };
  const int rng_ncomp = 8;
  auto rng_device_kernel =
      host_atomic_block_kernel_rng<REAL>(rng_lambda, rng_ncomp);

  auto cast_rng_device_kernel =
      std::dynamic_pointer_cast<HostAtomicBlockKernelRNG<REAL>>(
          rng_device_kernel);

  auto la = std::make_shared<LocalArray<int>>(sycl_target, 1024);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) == 0; }, Access::read(Sym<INT>("ID")));

  if (aa->get_npart_local() > 0) {
    ASSERT_EQ(aa->get_npart_local(), 1);

    std::set<int> seen_values;
    int sample_count = 0;

    int num_values = 32;
    cast_rng_device_kernel->set_num_random_numbers(num_values);

    auto lambda_loop = [&]() {
      particle_loop(
          aa,
          [=](auto INDEX, auto RNG, auto LA) {
            for (int ix = 0; ix < num_values; ix++) {
              LA.at(ix) = RNG.at(INDEX, ix);
            }
          },
          Access::read(ParticleLoopIndex{}), Access::read(rng_device_kernel),
          Access::write(la))
          ->execute();
    };

    // Run the plain kernel
    ASSERT_TRUE(rng_device_kernel->valid_internal_state());
    auto hla = la->get();
    sample_count += num_values;
    for (int ix = 0; ix < num_values; ix++) {
      seen_values.insert(hla.at(ix));
    }
    ASSERT_EQ(seen_values.size(), sample_count);

    // rerun plain kernel
    lambda_loop();
    ASSERT_TRUE(rng_device_kernel->valid_internal_state());
    hla = la->get();
    sample_count += num_values;
    for (int ix = 0; ix < num_values; ix++) {
      seen_values.insert(hla.at(ix));
    }
    ASSERT_EQ(seen_values.size(), sample_count);

    num_values = 4;
    lambda_loop();
    ASSERT_TRUE(rng_device_kernel->valid_internal_state());
    hla = la->get();
    sample_count += num_values;
    for (int ix = 0; ix < num_values; ix++) {
      seen_values.insert(hla.at(ix));
    }
    ASSERT_EQ(seen_values.size(), sample_count);

    num_values = 2;
    lambda_loop();
    ASSERT_TRUE(rng_device_kernel->valid_internal_state());
    hla = la->get();
    sample_count += num_values;
    for (int ix = 0; ix < num_values; ix++) {
      seen_values.insert(hla.at(ix));
    }
    ASSERT_EQ(seen_values.size(), sample_count);

    num_values = 4;
    lambda_loop();
    ASSERT_TRUE(rng_device_kernel->valid_internal_state());
    hla = la->get();
    sample_count += num_values;
    for (int ix = 0; ix < num_values; ix++) {
      seen_values.insert(hla.at(ix));
    }
    ASSERT_EQ(seen_values.size(), sample_count);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

namespace {
template <typename RNG_TYPE> struct RNGConsumer {

  std::mt19937 rng_state;
  const REAL a = 10.0;
  const REAL b = 20.0;
  std::uniform_real_distribution<> rng_dist;
  std::shared_ptr<RNG_TYPE> rng;

  RNGConsumer() { this->rng = std::make_shared<RNG_TYPE>(); }
  inline void setup() {
    this->rng_state = std::mt19937(52234234);
    const REAL a = 10.0;
    const REAL b = 20.0;
    this->rng_dist = std::uniform_real_distribution<>(a, b);
    auto rng_lambda = [&]() -> REAL { return this->rng_dist(this->rng_state); };
    this->rng = std::make_shared<RNG_TYPE>(rng_lambda, 3);
  }

  inline void execute(ParticleGroupSharedPtr particle_group) {
    particle_loop(
        particle_group,
        [=](auto INDEX, auto V, typename RNG_TYPE::KernelType RNG) {
          for (int cx = 0; cx < 3; cx++) {
            V.at(cx) = RNG.at(INDEX, cx);
          }
        },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("V")),
        Access::read(this->rng))
        ->execute();
  }
};
} // namespace

TEST(ParticleLoopRNG, type_casting) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;
  const int rank = sycl_target->comm_pair.rank_parent;

  RNGConsumer<HostAtomicBlockKernelRNG<REAL>> rng;
  rng.setup();
  rng.execute(A);

  A->free();
  sycl_target->free();
  mesh->free();
}
