// clang-format off
inline void advection_example_low_level(
    // Input ParticleGroup - we will loop over all particles in this
    // ParticleGroup. Note that all the objects required in this example can 
    // be obtained from just a ParticleDatSharedPtr.
    ParticleGroupSharedPtr pg
) {
  
  // These constants are captured by value into the kernel lambda.
  const int ndim = 2;
  const REAL dt = 0.001;
  
  // [A] Start an access window for the particle properties V and P.
  auto d_V = Access::direct_get(Access::read(pg->get_dat(Sym<REAL>("V"))));
  auto d_P = Access::direct_get(Access::write(pg->get_dat(Sym<REAL>("P"))));
  
  // Use a helper class to create an iteration set for all particles. Using
  // these helper functions/classes is not compulsory and the user is free to
  // create and use their own method of accessing the pointers provided by
  // direct_get.
  ParticleLoopImplementation::ParticleLoopBlockIterationSet 
    block_iteration_set(pg->get_dat(Sym<REAL>("P")));

  // The iteration set here is a vector of "blocks". The union of these blocks
  // defines an iteration set which covers all particles.
  auto iteration_set = block_iteration_set.get_all_cells();
  
  // We push all the sycl events into this helper object then wait on all of
  // them.
  EventStack event_stack;

  for (auto &block : iteration_set) {
    const auto block_device = block.block_device;
    event_stack.push(pg->sycl_target->queue.parallel_for<>(
        block.loop_iteration_set, [=](sycl::nd_item<2> idx) {
          
          // Get the particle cell and layer for this workitem.
          std::size_t cell;
          std::size_t layer;
          block_device.get_cell_layer(idx, &cell, &layer);

          // The ParticleLoopBlockIterationSet may produce blocks which 
          // overflow the actual number of particles in the iteration set. the
          // block_device instance has a helper function to determine if this
          // item is required. The corresponding host instance, here "block",
          // contains a member bool variable 
          // "block.layer_bounds_check_required" which can be inspected to 
          // determine if the kernel must perform this check. If this bool is
          // inspected then the user will have two parallel loop types, one 
          // which has this conditional and one which does not. 
          if (block_device.work_item_required(cell, layer)) {
            
            // This is the actual kernel that touches particle data in this 
            // example.
            for(int dx=0 ; dx<ndim ; dx++){
              d_P[cell][dx][layer] += dt * d_V[cell][dx][layer];
            }

          }
        }));
  }
  event_stack.wait();

  // [B] End the access window for the particle properties V and P. These 
  // functions must be called.
  Access::direct_restore(Access::read(pg->get_dat(Sym<REAL>("V"))), d_V);
  Access::direct_restore(Access::write(pg->get_dat(Sym<REAL>("P"))), d_P);
}
// clang-format on
