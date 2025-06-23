// clang-format off
inline void cell_dat_const_reduction_example(
    ParticleGroupSharedPtr particle_group
) {
  // Get the number of cells on this MPI rank.
  const int cell_count = particle_group->domain->mesh->get_cell_count();

  // Create a 3x1 matrix for cell_count cells.
  auto g1 = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);
  // Set the initial values.
  g1->fill(0.0);
  
  particle_loop(
    "cell_dat_const_reduction_example",
    particle_group,
    [=](auto V, auto GA){
      // Increment the matrix in each cell with the velocities of particles in
      // that cell. These values can be modified prior to calling combine as 
      // required.
      GA.combine(0, 0, V.at(0));
      GA.combine(1, 0, V.at(1));
      GA.combine(2, 0, V.at(2));
    },
    // Particle property access descriptor.
    Access::read(Sym<REAL>("V")),
    // CellDatConst access descriptor for reduction.
    Access::reduce(g1, Kernel::plus<REAL>())
  )->execute();
  

  // Alternatively use the helper function to reduce particle dat values cell 
  // wise.

  // Create an Output CellDatConst with dimensions equal to the particle data 
  // we will reduce.
  auto g2 = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);
  g2->fill(0.0);

  // This function call is a specialised implementation of the above particle 
  // loop.
  reduce_dat_components_cellwise(particle_group, Sym<REAL>("V"), g2,
                                 Kernel::plus<REAL>());
  
}
// clang-format on
