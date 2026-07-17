#include "include/test_neso_particles.hpp"

namespace {

template <typename T> struct RNGTestDevice : RNGGenerationFunction<T> {

  std::mt19937 rng;

  std::set<T> sampled_values;
  std::uniform_real_distribution<REAL> dist{
      std::uniform_real_distribution<REAL>(1.0, 2.0)};

  virtual inline void
  draw_random_samples(SYCLTargetSharedPtr sycl_target, T *d_ptr,
                      const std::size_t num_numbers,
                      [[maybe_unused]] const int block_size) override {
    std::vector<T> samples;
    samples.reserve(num_numbers);
    for (std::size_t ix = 0; ix < num_numbers; ix++) {
      samples.push_back(this->dist(this->rng));
    }

    auto e0 = sycl_target->queue.memcpy(d_ptr, samples.data(),
                                        num_numbers * sizeof(T));

    for (std::size_t ix = 0; ix < num_numbers; ix++) {
      this->sampled_values.insert(samples.at(ix));
    }

    e0.wait_and_throw();
  }
};

} // namespace

TEST(ParticleLoopRNGDevice, base_function_only) {
  auto [A, sycl_target_t, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto sycl_target = sycl_target_t;
  auto lambda_sampler = [&](REAL *d_ptr, const std::size_t num_numbers) -> int {
    std::mt19937 rng;
    std::uniform_real_distribution<REAL> dist{
        std::uniform_real_distribution<REAL>(1.0, 2.0)};
    std::vector<REAL> samples;
    samples.reserve(num_numbers);
    for (std::size_t ix = 0; ix < num_numbers; ix++) {
      samples.push_back(dist(rng));
    }
    sycl_target->queue.memcpy(d_ptr, samples.data(), num_numbers * sizeof(REAL))
        .wait_and_throw();
    return 0;
  };

  auto rng_function =
      make_rng_generation_function<GenericDeviceRNGGenerationFunction, REAL>(
          lambda_sampler);
  auto rng = host_atomic_block_kernel_rng<REAL>(rng_function, 4);
  auto rng_orig = std::dynamic_pointer_cast<RNGTestDevice<REAL>>(rng_function);

  particle_loop(
      A,
      [=](auto INDEX, auto RNG, auto V) {
        for (int dx = 0; dx < 3; dx++) {
          bool valid;
          V.at(dx) = RNG.at(INDEX, dx, &valid);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng),
      Access::write(Sym<REAL>("V")))
      ->execute();

  for (int cellx = 0; cellx < cell_count_t; cellx++) {
    auto V = A->get_cell(Sym<REAL>("V"), cellx);
    const int nrow = V->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int dx = 0; dx < 3; dx++) {
        ASSERT_TRUE(V->at(rowx, dx) >= 1.0);
        ASSERT_TRUE(V->at(rowx, dx) <= 2.0);
      }
    }
  }

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticleLoopRNGDevice, base_block) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto rng_function = make_rng_generation_function<RNGTestDevice, REAL>();
  auto rng = host_per_particle_block_rng<REAL>(rng_function, 4);
  auto rng_orig = std::dynamic_pointer_cast<RNGTestDevice<REAL>>(rng_function);

  particle_loop(
      A,
      [=](auto INDEX, auto RNG, auto V) {
        for (int dx = 0; dx < 3; dx++) {
          bool valid;
          V.at(dx) = RNG.at(INDEX, dx, &valid);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng),
      Access::write(Sym<REAL>("V")))
      ->execute();

  for (int cellx = 0; cellx < cell_count_t; cellx++) {
    auto V = A->get_cell(Sym<REAL>("V"), cellx);
    const int nrow = V->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int dx = 0; dx < 3; dx++) {
        ASSERT_TRUE(V->at(rowx, dx) >= 1.0);
        ASSERT_TRUE(V->at(rowx, dx) <= 2.0);
        ASSERT_TRUE(rng_orig->sampled_values.count(V->at(rowx, dx)));
      }
    }
  }

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticleLoopRNGDevice, base_atomic) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto rng_function = make_rng_generation_function<RNGTestDevice, REAL>();
  auto rng = host_atomic_block_kernel_rng<REAL>(rng_function, 4);
  auto rng_orig = std::dynamic_pointer_cast<RNGTestDevice<REAL>>(rng_function);

  particle_loop(
      A,
      [=](auto INDEX, auto RNG, auto V) {
        for (int dx = 0; dx < 3; dx++) {
          bool valid;
          V.at(dx) = RNG.at(INDEX, dx, &valid);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng),
      Access::write(Sym<REAL>("V")))
      ->execute();

  for (int cellx = 0; cellx < cell_count_t; cellx++) {
    auto V = A->get_cell(Sym<REAL>("V"), cellx);
    const int nrow = V->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int dx = 0; dx < 3; dx++) {
        ASSERT_TRUE(V->at(rowx, dx) >= 1.0);
        ASSERT_TRUE(V->at(rowx, dx) <= 2.0);
        ASSERT_TRUE(rng_orig->sampled_values.count(V->at(rowx, dx)));
      }
    }
  }

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticleLoopRNGDevice, atomic_sampling_correctness) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  const int rng_ncomp = 10;

  int state0 = 0;
  auto rng_lambda = [&]() { return state0++; };

  auto rng_atomic_kernel =
      host_atomic_block_kernel_rng<INT>(rng_lambda, rng_ncomp);
  rng_atomic_kernel->max_factor = 100;

  auto ae = particle_sub_group(A, []() { return false; });
  ASSERT_EQ(ae->get_npart_local(), 0);
  auto a00 = particle_sub_group(
      A, [=](auto index) { return (index.cell == 0) && (index.layer == 0); },
      Access::read(ParticleLoopIndex{}));
  ASSERT_EQ(a00->get_npart_local(), 1);

  auto a01 = particle_sub_group(
      A, [=](auto index) { return (index.cell <= 1) && (index.layer == 0); },
      Access::read(ParticleLoopIndex{}));
  ASSERT_EQ(a01->get_npart_local(), 2);

  particle_loop(
      ae, [=]([[maybe_unused]] auto RNG) {}, Access::read(rng_atomic_kernel))
      ->execute();

  ASSERT_EQ(state0, 0);
  ASSERT_TRUE(rng_atomic_kernel->valid_internal_state());

  auto d_samples = std::make_shared<BufferDevice<int>>(sycl_target, 20);
  auto k_samples = d_samples->ptr;
  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  auto l0 = particle_loop(
      a00, [=]([[maybe_unused]] auto RNG) {}, Access::read(rng_atomic_kernel));

  l0->execute();
  ASSERT_EQ(state0, 10);
  ASSERT_TRUE(rng_atomic_kernel->valid_internal_state());

  l0->execute();
  ASSERT_EQ(state0, 10);
  ASSERT_TRUE(rng_atomic_kernel->valid_internal_state());

  auto l1 = particle_loop(
      a00,
      [=](auto INDEX, auto RNG) {
        bool valid = false;
        for (int ix = 0; ix < 4; ix++) {
          const int v = RNG.at(INDEX, ix, &valid);
          k_samples[ix] = v;
          NESO_KERNEL_ASSERT(valid, k_ep);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng_atomic_kernel));

  l1->execute();
  ASSERT_TRUE(rng_atomic_kernel->valid_internal_state());
  ASSERT_FALSE(ep.get_flag());
  ASSERT_EQ(state0, 10);

  std::vector<int> h_correct(20);
  std::vector<int> h_samples = d_samples->get();

  for (int ix = 0; ix < 4; ix++) {
    ASSERT_EQ(h_samples.at(ix), ix);
  }

  // [10, 11, 12, 13, 4, 5, 6, 7, 8, 9]
  l1->execute();
  ASSERT_TRUE(rng_atomic_kernel->valid_internal_state());
  ASSERT_FALSE(ep.get_flag());
  ASSERT_EQ(state0, 14);

  h_samples = d_samples->get();
  for (int ix = 0; ix < 4; ix++) {
    ASSERT_EQ(h_samples.at(ix), ix + 10);
  }

  auto l2 = particle_loop(
      a00,
      [=](auto INDEX, auto RNG) {
        bool valid = false;
        for (int ix = 0; ix < 10; ix++) {
          const int v = RNG.at(INDEX, ix, &valid);
          k_samples[ix] = v;
          NESO_KERNEL_ASSERT(valid, k_ep);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng_atomic_kernel));

  // [14, 15, 16, 17, 4, 5, 6, 7, 8, 9]
  l2->execute();
  ASSERT_TRUE(rng_atomic_kernel->valid_internal_state());
  ASSERT_FALSE(ep.get_flag());
  ASSERT_EQ(state0, 18);

  h_correct = {14, 15, 16, 17, 4, 5, 6, 7, 8, 9};

  h_samples = d_samples->get();
  for (int ix = 0; ix < 10; ix++) {
    ASSERT_EQ(h_samples.at(ix), h_correct.at(ix));
  }

  //[18, 19, ... , 36, 37]
  particle_loop(
      a01,
      [=](auto INDEX, auto RNG) {
        const auto lid = INDEX.get_loop_linear_index();
        bool valid = false;
        for (int ix = 0; ix < 10; ix++) {
          k_samples[lid * 10 + ix] = RNG.at(INDEX, ix, &valid);
          NESO_KERNEL_ASSERT(valid, k_ep);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(rng_atomic_kernel))
      ->execute();
  ASSERT_TRUE(rng_atomic_kernel->valid_internal_state());
  ASSERT_FALSE(ep.get_flag());
  ASSERT_EQ(state0, 38);

  std::set<int> s_correct;
  for (int ix = 18; ix < 38; ix++) {
    s_correct.insert(ix);
  }

  std::set<int> s_to_test;
  h_samples = d_samples->get();
  for (int ix = 0; ix < 20; ix++) {
    s_to_test.insert(h_samples.at(ix));
  }

  ASSERT_EQ(s_to_test, s_correct);

  sycl_target->free();
  A->domain->mesh->free();
}
