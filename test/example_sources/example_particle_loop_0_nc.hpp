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
        P[dimx] += dt * V[dimx];
      }
    },
    Access::write(Sym<REAL>("P")),
    Access::read(Sym<REAL>("V"))
  );

  loop->execute();
  return; 
}
// clang-format on
