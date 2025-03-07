// clang-format off
inline void particle_loop_example_cell_info_npart(
    ParticleGroupSharedPtr particle_group
) {
  particle_loop(
    "loop_cell_info_npart_example",
    particle_group,
    [=](auto cell_info_npart){
      // Dummy output variable we store the particle count in 
      // for this example.
      [[maybe_unused]] INT tmp = cell_info_npart.get();
    },
    // Note the extra {} that creates an instance of the type.
    Access::read(CellInfoNPart{})
  )->execute();
}
// clang-format on
