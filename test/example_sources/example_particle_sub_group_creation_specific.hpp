// clang-format off
inline void particle_sub_group_creation_specific(
    // Input ParticleGroup - we will select a subset of the particles in this
    // group.
    ParticleGroupSharedPtr particle_group
) {
  
  /**
   * Create a naive copy of the ParticleGroup using the base sub group
   * function.
   */
  auto a0 = particle_sub_group(particle_group, [=](){return true;});

  /**
   * Creates a copy of a0.
   */
  auto a1 = particle_sub_group(a0);
  
  /**
   * Creates a lightweight reference to the whole particle group.
   */
  auto a2 = particle_sub_group(particle_group);

  /**
   * Creates a copy of a2. As a2 is a ParticleGroup this also creates a
   * lightweight reference.
   */
  auto a3 = particle_sub_group(a2);

  /**
   * Select all particles for all cells within a cell range from a
   * ParticleGroup.
   */
  auto c0 = particle_sub_group(particle_group, 0, 1);

  /**
   * Select all particles for all cells within a cell range from a
   * ParticleSubGroup.
   */
  auto c1 = particle_sub_group(a0, 0, 1);

  /**
   * Discard n particles then select any remaining particles from a particle
   * group.
   */
  const int n = 4;
  auto d0 = particle_sub_group_discard(particle_group, n);
  
  /**
   * Discard n particles then select any remaining particles from a particle
   * sub group. 
   */
  auto d1 = particle_sub_group_discard(d0, n);

  /**
   * Select only the first, at most, n particles from each cell from the source
   * ParticleGroup.
   */
  auto t0 = particle_sub_group_truncate(particle_group, n);

  /**
   * Select only the first, at most, n particles from each cell from the source
   * ParticleGroup.
   */
  auto t1 = particle_sub_group_truncate(a0, n);
}
// clang-format on
