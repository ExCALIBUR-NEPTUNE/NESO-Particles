// clang-format off
inline void advection_example(
    // Input ParticleGroup - we will loop over all particles in this
    // ParticleGroup.
    ParticleGroupSharedPtr particle_group
) {
  
  // These constants are captured by value into the kernel lambda.
  const int ndim = 2;
  const REAL dt = 0.001;
  
  auto loop = particle_loop(
    "advection_example", // Optional name for the loop - for profiling.

    particle_group,      // Iteration set is defined as all particles in this
                         // ParticleGroup.

    // This lambda defines the kernel to be executed for
    // all particles in the iteration set.
    // The [=] captures the ndim and dt variables by value. A [&] would capture
    // these variables by reference.
    [=](auto P, auto V){ // These parameters have a one-to-one correspondence
                         // with the loop arguments that follow the kernel in
                         // the call to particle_loop.
      
      // Loop over the P particle property and update the values with values
      // read from the V property.
      for(int dimx=0 ; dimx<ndim ; dimx++){
        P[dimx] += dt * V[dimx];
      }
    },
    // The remaining arguments passed to the particle loop are the link between
    // the kernel parameters and the data structures which the loop accesses.
    // Each argument is passed as a combination of an access descriptor and a
    // data structure. 
    
    // In this example the data structure is a ParticleDat and it is referenced
    // by the symbol for the ParticleDat in the ParticleGroup. Here we indicate
    // that the "P" ParticleDat is accessed in a write mode.
    Access::write(Sym<REAL>("P")),
    // Here we pass the V ParticleDat to the particle loop and indicate that
    // these particle properties are accessed in a read-only mode in the
    // particle loop.
    Access::read(Sym<REAL>("V"))
  );
  
  // Execute the ParticleLoop. A ParticleLoop can be executed multiple times
  // without reconstruction.
  loop->execute();

  return; 
}
// clang-format on
