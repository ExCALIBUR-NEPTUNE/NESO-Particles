// clang-format off
inline void nd_local_array_example(
    ParticleGroupSharedPtr particle_group
) {
  
  // Create a new 2D NDLocalArray to accumulate values into.
  auto local_array_add = std::make_shared<NDLocalArray<INT, 2>>(
    particle_group->sycl_target, // Compute target to use.
    2,                           // Number of elements in the array for 
                                 // dimension 0.
    1                            // Number of elements in the array for 
                                 // dimension 1.
  );
  // Fill the array with the same value for all elements.
  local_array_add->fill(0);
  
  // Create a 2D NDLocalArray for read access.
  auto local_array_read = std::make_shared<NDLocalArray<REAL, 2>>(
    particle_group->sycl_target, // Compute device.
    1,                           // Number of elements in dimension 0. 
    3                            // Number of elements in dimension 1. 
  );
  
  // Get the contents of an NDLocalArray on the host.
  // Note uninitialised data here.
  auto v = local_array_read->get();
  // The data in the array is linearised such that the indices run from slowest 
  // to fastest.
  v.at(0) = 1.0; // Element at index (0,0).
  v.at(1) = 2.0; // Element at index (0,1).
  v.at(2) = 3.0; // Element at index (0,2).
  
  // Copy the host std::vector back into the NDLocalArray
  local_array_read->set(v);
  
  auto loop = particle_loop(
    "local_array_example",
    particle_group,
    // LA_ADD is the parameter for the NDLocalArray with "add" access and 
    // LA_READ is the parameter for the NDLocalArray with read-only access.
    [=](auto ID, auto V, auto LA_ADD, auto LA_READ){
      
      // Increment the (0, 0) component of LA_ADD by 1.
      LA_ADD.fetch_add(0, 0, 1);
      // Increment the (1, 0) component of LA_ADD with the entry in ID[0].
      LA_ADD.fetch_add(1, 0, ID[0]);
      
      // Read from LA_READ and assign the values to the V particle component.
      V[0] = LA_READ.at(0, 0);
      V[1] = LA_READ.at(0, 1);
      V[2] = LA_READ.at(0, 2);
    },
    // Particle property access descriptors.
    Access::read(Sym<INT>("ID")),
    Access::write(Sym<REAL>("V")),
    // NDLocalArray access descriptors.
    Access::add(local_array_add),
    Access::read(local_array_read)
  );

  // Execute the loop.
  loop->execute();
  
  // Get the contents of the local array in a std::vector.
  auto vec0 = local_array_add->get();
}
// clang-format on
