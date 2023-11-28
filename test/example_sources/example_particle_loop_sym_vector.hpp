// clang-format off
inline void sym_vector_example(
    ParticleGroupSharedPtr particle_group
) {
  
  // These constants are captured by value into the kernel lambda.
  const int ndim = 2;
  const REAL dt = 0.001;
  
  auto loop = particle_loop(
    "sym_vector_example",
    particle_group,
    [=](auto particle_index, auto dats_vector){
      for(int dimx=0 ; dimx<ndim ; dimx++){
        // P has index 0 in dats_vector as it is first in the sym_vector.
        // V has index 1 in dats_vector as it is second.
        dats_vector.at(0, particle_index, dimx) += 
          dt * dats_vector.at(1, particle_index, dimx);
      }     
    },
    // The loop index data structure simplifies indexing into the SymVector 
    // in the kernel.
    Access::read(ParticleLoopIndex{}),
    // We state that all ParticleDats in the SymVector are write access.
    Access::write(
      // Helper function to create a SymVector.
      sym_vector<REAL>(
        particle_group,
        // This argument may also be a std::vector of Syms.
        {Sym<REAL>("P"), Sym<REAL>("V")}
      )
    )
  );

  // Execute the loop.
  loop->execute();
  return; 
}
// clang-format on
