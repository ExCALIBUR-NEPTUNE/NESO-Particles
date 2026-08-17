#ifndef _NESO_PARTICLES_CONTAINERS_RNG_RNG_GENERATION_FUNCTION_H_
#define _NESO_PARTICLES_CONTAINERS_RNG_RNG_GENERATION_FUNCTION_H_

#include "../../compute_target.hpp"

namespace NESO::Particles {

/**
 * This type defines the absract interface for creating RNG values in a block.
 */
template <typename T> struct RNGGenerationFunction {
  virtual ~RNGGenerationFunction() = default;

  /**
   * Draw values from a host generation function and copy them to a device
   * buffer.
   *
   * @param[in] sycl_target Compute device.
   * @param[in, out] d_ptr Pointer to buffer in which to place values.
   * @param[in] num_numbers Number of values to draw from generation function.
   * @param[in] block_size Block size to use when copying into device buffer.
   */
  virtual inline void draw_random_samples(SYCLTargetSharedPtr sycl_target,
                                          T *d_ptr,
                                          const std::size_t num_numbers,
                                          const int block_size) = 0;
};

/**
 * Helper function to create a RNGGenerationFunction object from child type.
 * Called like:
 *
 *  make_rng_generation_function<RNG_TYPE, VALUE_TYPE>(ARGS...);
 *
 * @param args Args to pass to downstream constructor.
 */
template <template <typename> typename RNG_TYPE, typename VALUE_TYPE,
          typename... ARGS>
inline std::shared_ptr<RNGGenerationFunction<VALUE_TYPE>>
make_rng_generation_function(ARGS... args) {
  std::shared_ptr<RNGGenerationFunction<VALUE_TYPE>> ptr =
      std::dynamic_pointer_cast<RNGGenerationFunction<VALUE_TYPE>>(
          std::make_shared<RNG_TYPE<VALUE_TYPE>>(args...));
  NESOASSERT(ptr != nullptr,
             "Could not cast pointers for RNGGenerationFunction.");
  return ptr;
}

} // namespace NESO::Particles

#endif
