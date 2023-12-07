// clang-format off
inline void advection_example_loop_index(
    ParticleGroupSharedPtr particle_group
) {
  
  auto loop = particle_loop(
    "loop_index_example",
    particle_group,
    [=](auto index){
      // Dummy output variable we store the indices in for this example.
      [[maybe_unused]] INT tmp;
      // The cell containing the particle.
      tmp = index.cell; 
      // The row (layer) containing the particle.
      tmp = index.layer; 
      // The linear index of the particle on the calling MPI rank. 
      // This index is in [0, A->get_npart_local()).
      tmp = index.get_local_linear_index(); 
      // The linear index of the particle in the current ParticleLoop. 
      // This index is in [0, <size of ParticleLoop iteration set>).
      tmp = index.get_loop_linear_index(); 
    },
    // Note the extra {} that creates an instance of the type.
    Access::read(ParticleLoopIndex{})
  );

  loop->execute();
  return; 
}
// clang-format on
