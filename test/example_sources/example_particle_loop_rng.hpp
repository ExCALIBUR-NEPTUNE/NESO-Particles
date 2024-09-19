// clang-format off
inline void particle_loop_rng(
    // Input ParticleGroup - we will loop over all particles in this
    // ParticleGroup.
    ParticleGroupSharedPtr particle_group
) {
  
  // Number of RNG values required per particle in the ParticleLoop.
  const int num_components = 3;
  
  // Create a RNG on the host with the required distribution.
  std::mt19937 rng_state(52234234);
  std::normal_distribution<REAL> rng_dist(0, 1.0);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng_state); };
  
  // Create a PerParticleBlockRNG instance from the required distribution.
  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, num_components);
  
  // Create a ParticleLoop which samples the distribution num_components times
  // per particle.
  auto loop = particle_loop(
    "rng_example", 
    particle_group,
    [=](auto INDEX, auto DIST, auto V){
      for(int dimx=0 ; dimx<num_components ; dimx++){
        V.at(dimx) = DIST.at(INDEX, dimx);
      }
    },
    // A ParticleLoopIndex facilities access to the RNG values.
    Access::read(ParticleLoopIndex{}),
    // A KernelRNG may only be accessed in Read mode.
    Access::read(rng_kernel),
    // Output particle property.
    Access::write(Sym<REAL>("V"))
  );
  
  loop->execute();

  return; 
}
// clang-format on
