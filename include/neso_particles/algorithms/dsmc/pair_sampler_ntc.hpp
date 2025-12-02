#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_PAIR_SAMPLER_NTC_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_PAIR_SAMPLER_NTC_HPP_

#include "../../compute_target.hpp"
#include "../../containers/rng/host_rng_common.hpp"
#include "../../pair_loop/cellwise_pair_list_block.hpp"
#include "../../particle_sub_group/particle_sub_group.hpp"

namespace NESO::Particles {

namespace DSMC {

/**
 * Interface to sample pairs of particles for NTC style DSMC.
 */
class PairSamplerNTC : public CellwisePairListBlockInterface {
protected:
  std::shared_ptr<BufferDevice<int>> d_wave_counts;
  int pair_count{0};
  std::vector<int> h_pair_counts;
  std::shared_ptr<BufferDevice<int>> d_pair_counts;
  std::vector<INT> h_pair_counts_es;
  std::shared_ptr<BufferDevice<INT>> d_pair_counts_es;
  int block_size{0};
  std::shared_ptr<BufferDevice<int>> d_pair_list;
  CellwisePairListBlockDevice d_pair_list_block_device;

  std::int64_t version_a{0};
  std::int64_t version_b{0};
  ParticleSubGroupWeakPtr particle_sub_group_a;
  ParticleSubGroupWeakPtr particle_sub_group_b;

public:
  /// Disable (implicit) copies.
  PairSamplerNTC(const PairSamplerNTC &st) = delete;
  /// Disable (implicit) copies.
  PairSamplerNTC &operator=(PairSamplerNTC const &a) = delete;

  virtual ~PairSamplerNTC() = default;

  SYCLTargetSharedPtr sycl_target{nullptr};
  int cell_count{0};
  std::shared_ptr<RNGGenerationFunction<REAL>> rng_generation_function{nullptr};

  PairSamplerNTC(
      SYCLTargetSharedPtr sycl_target, const int cell_count,
      std::shared_ptr<RNGGenerationFunction<REAL>> rng_generation_function);

  void sample(ParticleSubGroupSharedPtr sub_group_a,
              ParticleSubGroupSharedPtr sub_group_b,
              std::vector<int> &new_sample_counts);

  virtual CellwisePairListBlockDevice get_pair_list() override;
};

} // namespace DSMC
} // namespace NESO::Particles

#endif
