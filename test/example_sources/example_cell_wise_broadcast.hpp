// clang-format off
inline void cellwise_broadcast(
    ParticleGroupSharedPtr particle_group
) {
  // Get the number of cells on this MPI rank.
  const int cell_count = particle_group->domain->mesh->get_cell_count();

  std::vector<INT> h_cell_values(cell_count);
  for(int ix=0 ; ix<cell_count ; ix++){
    h_cell_values.at(ix) = ix + 1;
  }

  // All particles in cell i will have ID.at(0) set to i + 1.
  cellwise_broadcast(particle_group, Sym<INT>("ID"), 0, h_cell_values);

  // The cellwise_broadcast function can also be called with a ParticleSubGroup.
}
// clang-format on
