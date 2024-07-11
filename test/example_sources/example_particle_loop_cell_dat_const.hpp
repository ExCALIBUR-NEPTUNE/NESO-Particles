// clang-format off
inline void cell_dat_const_example(
    ParticleGroupSharedPtr particle_group
) {
  
  // Get the number of cells on this MPI rank.
  const int cell_count = particle_group->domain->mesh->get_cell_count();

  // Create a 3x1 matrix for cell_count cells.
  auto g1 = std::make_shared<CellDatConst<REAL>>(
      particle_group->sycl_target, cell_count, 3, 1);
  
  // For each cell get the current matrix and zero the values then write back
  // to the data structure.
  for (int cx = 0; cx < cell_count; cx++) {
    auto cell_data = g1->get_cell(cx);
    cell_data->at(0, 0) = 0.0;
    cell_data->at(1, 0) = 0.0;
    cell_data->at(2, 0) = 0.0;
    g1->set_cell(cx, cell_data);
  }

  // Alternatively all entries in all cells of the CellDatConst may be filled
  // with a value.
  g1->fill(0.0);
  
  auto loop = particle_loop(
    "cell_dat_const_example",
    particle_group,
    [=](auto V, auto GA){
      // Increment the matrix in each cell with the velocities of particles in
      // that cell.
      GA.fetch_add(0, 0, V[0]);
      GA.fetch_add(1, 0, V[1]);
      GA.fetch_add(2, 0, V[2]);
    },
    // Particle property access descriptor.
    Access::read(Sym<REAL>("V")),
    // CellDatConst access descriptor.
    Access::add(g1)
  );
  
  // Execute the loop.
  loop->execute();

  // After loop execution the 3x1 matrix in each cell will contain the sum of
  // the particle velocities in each cell.
  for (int cx = 0; cx < cell_count; cx++) {
    auto cell_data = g1->get_cell(cx);
    // use cell data
  }

  return; 
}
// clang-format on
