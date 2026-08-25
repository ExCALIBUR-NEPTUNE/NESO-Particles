#include <neso_particles/algorithms/unseen_value_extractor.hpp>

namespace NESO::Particles {

UnseenValueExtractor::UnseenValueExtractor(SYCLTargetSharedPtr sycl_target)
    : sycl_target(sycl_target), known_bounds(false) {

  this->d_tree = std::make_shared<BlockedBinaryTree<
      INT, NodeType, NESO_PARTICLES_UNSEEN_VALUE_EXTRACTOR_NODE_WIDTH>>(
      this->sycl_target);
}

UnseenValueExtractor::UnseenValueExtractor(SYCLTargetSharedPtr sycl_target,
                                           const INT bound_lower,
                                           const INT bound_upper)
    : UnseenValueExtractor(sycl_target) {

  this->known_bounds = false;
  this->bound_lower = bound_lower;
  this->bound_upper = bound_upper;
  NESOASSERT(bound_lower < bound_upper, "Bad bounds passed.");
  const INT bound_range = bound_upper - bound_lower;
  const INT bound_range_limit = this->sycl_target->parameters->get_env_size_t(
      "NESO_PARTICLES_UNSEEN_VALUE_EXTRACTOR_MAX_RANGE", 8000000);

  if (bound_range <= bound_range_limit) {
    this->known_bounds = true;
    this->d_known_bound_seen_values =
        std::make_shared<BufferDevice<int>>(this->sycl_target, bound_range);
    this->sycl_target->queue
        .fill<int>(this->d_known_bound_seen_values->ptr, 0, bound_range)
        .wait_and_throw();
  }
}

bool UnseenValueExtractor::using_known_bounds() { return this->known_bounds; }

template std::set<INT>
UnseenValueExtractor::extract(std::shared_ptr<ParticleGroup> group,
                              Sym<INT> sym, const int component,
                              const bool is_ephemeral);

template std::set<INT>
UnseenValueExtractor::extract(std::shared_ptr<ParticleSubGroup> group,
                              Sym<INT> sym, const int component,
                              const bool is_ephemeral);

} // namespace NESO::Particles
