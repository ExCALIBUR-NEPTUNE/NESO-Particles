// clang-format off
inline void particle_sub_group_creation(
    // Input ParticleGroup - we will select a subset of the particles in this
    // group.
    ParticleGroupSharedPtr particle_group
) {
  
  // Create new sub group
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(
    // The source ParticleGroup
    particle_group,

    // The second argument is a lambda which follows the rules of ParticleLoop
    // kernel lambdas with the exception that this lambda returns true for
    // particles which are in the sub group and false for particles which are
    // not. This lambda may only access particle data and may only access
    // particle data with read-only access descriptors.
    [=](auto ID) {
      return (ID[0] % 2 == 0);
    }, 

    // The remaining arguments are the ParticleLoop compliant arguments for the
    // kernel with selects particles.
    Access::read(Sym<INT>("ID"))
  );

  return; 
}
// clang-format on
