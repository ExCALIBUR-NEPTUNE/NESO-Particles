// clang-format off
inline void global_array_example(
    ParticleGroupSharedPtr particle_group
) {
  
  // Create a new GlobalArray on the same SYCLTarget as the particle group.
  auto global_array = std::make_shared<GlobalArray<REAL>>(
    particle_group->sycl_target, // Compute target to use.
    1,                           // Number of elements in the array.
    0                            // Initial value for elements.
  );
  
  auto loop = particle_loop(
    "global_array_example",
    particle_group,
    [=](auto V, auto GA){
      // Kinetic energy of this particle.
      const REAL kinetic_energy = 
        0.5 * (V[0] * V[0] + V[1] * V[1]);
      // Increment the first component of the global array with the
      // contribution from this particle.
      GA.add(0, kinetic_energy);
    },
    // Particle property access descriptor.
    Access::read(Sym<REAL>("V")),
    // GlobalArray access descriptor.
    Access::add(global_array)
  );
  
  // Execute the loop. This must be called collectively on the MPI communicator
  // of the SYCLTarget as the add operation is collective.
  loop->execute();

  // Get the contents of the global array in a std::vector. The first element
  // would be the kinetic energy of all particles in the ParticleGroup.
  auto vec0 = global_array->get();

  return; 
}
// clang-format on
