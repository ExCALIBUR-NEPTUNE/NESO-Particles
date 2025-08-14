#include <neso_particles/algorithms/unseen_value_extractor.hpp>

namespace NESO::Particles {

UnseenValueExtractor::UnseenValueExtractor(SYCLTargetSharedPtr sycl_target)
    : sycl_target(sycl_target) {

  this->d_tree =
      std::make_shared<BlockedBinaryTree<INT, NodeType>>(this->sycl_target);
}

template std::set<INT>
UnseenValueExtractor::extract(std::shared_ptr<ParticleGroup> group,
                              Sym<INT> sym, const int component,
                              const bool is_ephemeral);

template std::set<INT>
UnseenValueExtractor::extract(std::shared_ptr<ParticleSubGroup> group,
                              Sym<INT> sym, const int component,
                              const bool is_ephemeral);

} // namespace NESO::Particles
