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
