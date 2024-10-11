// clang-format off

/*
NESO_PARTICLES_PROFILING_REGION should be defined before NESO-Particles is 
included.

#define NESO_PARTICLES_PROFILING_REGION
#include <neso_particles.hpp>
*/

inline void profile_regions_example(
    // Input ParticleGroup - we will loop over all particles in this
    // ParticleGroup.
    ParticleGroupSharedPtr particle_group
) {
  // Reference to the ProfileMap for the SYCLTarget inside the ParticleGroup.
  // This is the ProfileMap which will be used for the ParticleLoops that have
  // this ParticleGroup as an iteration set.
  auto & profile_map = particle_group->sycl_target->profile_map;
  const int rank = particle_group->sycl_target->comm_pair.rank_parent;
  
  // Create some user written loops. The name of the loop is used in the
  // profiling.
  
  // Example where the name "pbc" and the time taken is recorded.
  auto loop_pbc = particle_loop(
    "pbc",
    particle_group,
    [=](auto P){
      P.at(0) = Kernel::fmod(P.at(0) + 8.0, 8.0);
      P.at(1) = Kernel::fmod(P.at(1) + 8.0, 8.0);
    },
    Access::write(Sym<REAL>("P"))
  );

  // Example where the kernel can be combined with additional metadata.
  const REAL dt = 0.001;
  auto loop_advect = particle_loop(
    "advect",
    particle_group,
    Kernel::Kernel(
      [=](auto P, auto V){
        P.at(0) += dt * V.at(0);
        P.at(1) += dt * V.at(1);
      },
      Kernel::Metadata(
        Kernel::NumBytes(6),
        Kernel::NumFLOP(4)
      )
    ),
    Access::write(Sym<REAL>("P")),
    Access::read(Sym<REAL>("V"))
  );
 
  // Enable recording of events and regions in the ProfileMap (default 
  // disabled).
  profile_map.enable();
  
  // Users can define their own regions and add them to the ProfileMap. The
  // region starts on creation of the object.
  auto r = ProfileRegion("NameFirstPart", "NameSecondPart");
  
  // Do something to time here.

  // End the region to profile
  r.end();
  // Add our custom region to the ProfileMap
  profile_map.add_region(r);
  
  // Run something to profile and record the regions from internal
  // implementations and ParticleLoops.
  for(int stepx=0 ; stepx<20 ; stepx++){
    loop_advect->execute();
    loop_pbc->execute();
    particle_group->hybrid_move();
    particle_group->cell_move();
  }

  // Disable recording of events and regions.
  profile_map.disable();
  // Write the regions and events to a json file with name
  // regions_example.rank.json.
  profile_map.write_events_json("regions_example", rank);
}
// clang-format on
