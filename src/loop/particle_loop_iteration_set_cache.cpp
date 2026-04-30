#include <neso_particles/loop/particle_loop_iteration_set_cache.hpp>

namespace NESO::Particles::ParticleLoopImplementation {

ParticleLoopIterationSetCache::ParticleLoopIterationSetCache(
    SYCLTargetSharedPtr sycl_target, const std::size_t ncell, int *h_npart_cell,
    int *d_npart_cell)
    : sycl_target(sycl_target), ncell(ncell), h_npart_cell(h_npart_cell),
      d_npart_cell(d_npart_cell) {}

void ParticleLoopIterationSetCache::clear() { this->cache.clear(); }

const std::vector<ParticleLoopBlockHost> &ParticleLoopIterationSetCache::get(
    const std::size_t cell_start, const std::size_t cell_end, std::size_t nbin,
    std::size_t local_size, const std::size_t num_bytes_local,
    const std::size_t stride, std::size_t *iteration_set_size) {

  if (this->cache.size() > cache_size_limit) {
    this->clear();
  }

  const cache_key_type key = {cell_start, cell_end, nbin, local_size, stride};

  if (!this->cache.count(key)) {
    this->cache[key] = std::make_unique<ParticleLoopBlockIterationSet>(
        this->sycl_target, this->ncell, this->h_npart_cell, this->d_npart_cell);
    this->cache[key]->get_generic(cell_start, cell_end, nbin, local_size,
                                  num_bytes_local, stride);
  }

  if (iteration_set_size != nullptr) {
    *iteration_set_size = this->cache[key]->iteration_set_size;
  }
  return this->cache[key]->iteration_set;
}
} // namespace NESO::Particles::ParticleLoopImplementation
