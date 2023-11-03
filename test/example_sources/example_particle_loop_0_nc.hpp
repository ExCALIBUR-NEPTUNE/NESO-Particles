// clang-format off
inline void advection_example_no_comments(
    ParticleGroupSharedPtr particle_group
) {
  
  const int ndim = 2;
  const REAL dt = 0.001;
  
  auto loop = particle_loop(
    "advection_example",
    particle_group,
    [=](auto P, auto V){
      for(int dimx=0 ; dimx<ndim ; dimx++){
        // .at is an alternative access method
        P.at(dimx) += dt * V.at(dimx);
      }
    },
    Access::write(Sym<REAL>("P")),
    Access::read(Sym<REAL>("V"))
  );

  // Launch the particle loop.
  loop->submit();

  // Wait for execution of the particle loop to complete.
  loop->wait();
  return; 
}
// clang-format on
