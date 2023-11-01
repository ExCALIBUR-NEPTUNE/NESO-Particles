// clang-format off
inline void local_array_example(
    ParticleGroupSharedPtr particle_group
) {
  
  // Create a new LocalArray on the same SYCLTarget as the particle group.
  auto local_array_add = std::make_shared<LocalArray<REAL>>(
    particle_group->sycl_target, // Compute target to use.
    2,                           // Number of elements in the array.
    0                            // Initial value for elements.
  );
  
  // Create a local array using initial values from a std::vector.
  std::vector<REAL> d0 = {1.0, 2.0, 3.0};
  auto local_array_read = std::make_shared<LocalArray<REAL>>(
    particle_group->sycl_target, // Compute device.
    d0                           // Initial values and size definition.
  );
  
  auto loop = particle_loop(
    "local_array_example",
    particle_group,
    // LA_ADD is the parameter for the LocalArray with "add" access and LA_READ
    // is the parameter for the LocalArray with read-only access.
    [=](auto ID, auto V, auto LA_ADD, auto LA_READ){
      
      // Increment the first component of LA_ADD by 1.
      LA_ADD.add(0, 1);
      // Increment the second component of LA_ADD with the entry in ID[0].
      LA_ADD.add(1, ID[0]);
      
      // Read from LA_READ and assign the values to the V particle component.
      V[0] = LA_READ.at(0);
      V[1] = LA_READ.at(1);
      V[2] = LA_READ.at(2);
    },
    // Particle property access descriptors.
    Access::read(Sym<INT>("ID")),
    Access::write(Sym<REAL>("V")),
    // LocalArray access descriptors.
    Access::add(local_array_add),
    Access::read(local_array_read)
  );
  
  // Get the contents of the local array in a std::vector.
  auto vec0 = local_array_add->get();

  loop->execute();
  return; 
}
// clang-format on
