// clang-format off
inline void particle_sub_group_loop(
    ParticleGroupSharedPtr particle_group
) {
  
  // Create a ParticleSubGroup from even values of ID.
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(
    particle_group,
    [=](auto ID) {
      return (ID[0] % 2 == 0);
    }, 
    Access::read(Sym<INT>("ID"))
  );
  
  // Perform a position update style kernel on particles with even values of
  // ID[0].
  auto loop = particle_loop(
    particle_sub_group,
    [=](auto V, auto P){
      P[0] += 0.001 * V[0];
      P[1] += 0.001 * V[1];
    },
    Access::read(Sym<REAL>("V")),
    Access::write(Sym<REAL>("P"))
  );

  loop->execute();

  return; 
}
// clang-format on
