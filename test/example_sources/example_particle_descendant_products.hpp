// clang-format off
inline void descendant_products_example(
    ParticleGroupSharedPtr particle_group
) {
  
  /* For this example we assume that the particle group has the following REAL
     and INT properties with the specified number of components:
    
     REAL
     ----
      P 2 (positions)
      V 3
      Q 1
    
     INT
     ---
     CELL_ID 1 (cell id)
     ID 1
   */
  
  /*
   Example print output with 5 particles in particle_group:

  particle_group->print(
    Sym<REAL>("P"),
    Sym<REAL>("V"),
    Sym<REAL>("Q"),
    Sym<INT>("CELL_ID"),
    Sym<INT>("ID")
  );
    
================================================================================
------- 194 -------
| P | V | Q | CELL_ID | ID |
| 0.503483 3.132274 | -1.029170 -0.238606 0.833977 | 1.000000 | 194 | 3 |
------- 205 -------
| P | V | Q | CELL_ID | ID |
| 3.443890 3.179283 | -1.879651 -0.262682 -0.862215 | 1.000000 | 205 | 2 |
------- 217 -------
| P | V | Q | CELL_ID | ID |
| 2.443217 3.438988 | 1.305861 -1.304251 -0.096116 | 1.000000 | 217 | 1 |
------- 285 -------
| P | V | Q | CELL_ID | ID |
| 3.271273 4.276710 | -0.101299 -0.826377 0.081399 | 1.000000 | 285 | 0 |
------- 419 -------
| P | V | Q | CELL_ID | ID |
| 0.993615 6.648731 | -0.338175 0.151852 -1.346172 | 1.000000 | 419 | 4 |
================================================================================
  */

  // We create a DescendantProducts with the following specification.
  // The number of components for a property must match the number of
  // components in the ParticleGroup the products are added to.
  auto product_spec = product_matrix_spec(
    ParticleSpec(
      ParticleProp(Sym<REAL>("V"), 3),
      ParticleProp(Sym<REAL>("Q"), 1),
      ParticleProp(Sym<INT>("ID"), 1)
    )
  );

  /* Re-visiting the properties in the particle group and the product
     specification:

    REAL
    ----
    P: Is a ParticleGroup property not in the product spec therefore the values
       will be copied from the parent particles.
    V: Is defined in the product spec and the values set in the
       DescendantProducts will be used for the new particles.
    Q: Is defined in the product spec and the values set in the
       DescendantProducts will be used for the new particles.
    
    INT
    ---
    CELL_ID: Is a ParticleGroup property not in the product spec therefore the
             values will be copied from the parent particles.
    ID: Is defined in the product spec and the values set in the
    DescendantProducts will be used for the new particles.
  */
  
  // Create a DescendantProducts with the above product spec for at most 2
  // products per parent particle.
  const int num_products_per_particle = 2;
  auto dp = std::make_shared<DescendantProducts>(
    particle_group->sycl_target, 
    product_spec,
    num_products_per_particle
  );

  // Define a ParticleLoop which creates the products from the parent
  // particles.
  auto loop = particle_loop(
    particle_group,
    [=](
      auto DP, auto parent_index, auto V, auto Q, auto ID
    ){
      for(int childx=0 ; childx<num_products_per_particle ; childx++){
        // Enable this product by calling set_parent
        DP.set_parent(parent_index, childx);
        
        // The V property was the first REAL product we specified and therefore
        // has property index 0 for at_real.
        const int V_index = 0;
        for(int dimx=0 ; dimx<3 ; dimx++){
          // Copy V from parent but negate the sign.
          DP.at_real(parent_index, childx, V_index, dimx) = -1.0 * V.at(dimx);
        }

        // The Q property was the second REAL product specified and hence has
        // property index 1 for set_real.
        const int Q_index = 1;
        // Simply copy the parent Q value in this kernel.
        DP.at_real(parent_index, childx, Q_index, 0) = Q.at(0);

        // The ID property is the first INT property we specified and hence has
        // index 0 for at_int.
        const int ID_index = 0;
        // Copy parent ID but modify it.
        DP.at_int(parent_index, childx, ID_index, 0) = -1 * ID.at(0)
          - 100 * childx;
      }
    },
    Access::write(dp),
    Access::read(ParticleLoopIndex{}),
    Access::read(Sym<REAL>("V")),
    Access::read(Sym<REAL>("Q")),
    Access::read(Sym<INT>("ID"))
  );

  // Before a loop is executed that accesses a DescendantProducts data
  // structure the reset method must be called with the number of particles in
  // the iteration set of the loop.
  dp->reset(particle_group->get_npart_local());
  
  // Execute the loop to create the products.
  loop->execute();
  
  // Finally add the new products to the ParticleGroup.
  particle_group->add_particles_local(dp);

  /* Example print output with 5 particles in particle_group:

  particle_group->print(
    Sym<REAL>("P"),
    Sym<REAL>("V"),
    Sym<REAL>("Q"),
    Sym<INT>("CELL_ID"),
    Sym<INT>("ID")
  );

================================================================================
------- 194 -------
| P | V | Q | CELL_ID | ID |
| 0.503483 3.132274 | -1.029170 -0.238606 0.833977 | 1.000000 | 194 | 3 |
| 0.503483 3.132274 | 1.029170 0.238606 -0.833977 | 1.000000 | 194 | -3 |
| 0.503483 3.132274 | 1.029170 0.238606 -0.833977 | 1.000000 | 194 | -103 |
------- 205 -------
| P | V | Q | CELL_ID | ID |
| 3.443890 3.179283 | -1.879651 -0.262682 -0.862215 | 1.000000 | 205 | 2 |
| 3.443890 3.179283 | 1.879651 0.262682 0.862215 | 1.000000 | 205 | -102 |
| 3.443890 3.179283 | 1.879651 0.262682 0.862215 | 1.000000 | 205 | -2 |
------- 217 -------
| P | V | Q | CELL_ID | ID |
| 2.443217 3.438988 | 1.305861 -1.304251 -0.096116 | 1.000000 | 217 | 1 |
| 2.443217 3.438988 | -1.305861 1.304251 0.096116 | 1.000000 | 217 | -1 |
| 2.443217 3.438988 | -1.305861 1.304251 0.096116 | 1.000000 | 217 | -101 |
------- 285 -------
| P | V | Q | CELL_ID | ID |
| 3.271273 4.276710 | -0.101299 -0.826377 0.081399 | 1.000000 | 285 | 0 |
| 3.271273 4.276710 | 0.101299 0.826377 -0.081399 | 1.000000 | 285 | -100 |
| 3.271273 4.276710 | 0.101299 0.826377 -0.081399 | 1.000000 | 285 | 0 |
------- 419 -------
| P | V | Q | CELL_ID | ID |
| 0.993615 6.648731 | -0.338175 0.151852 -1.346172 | 1.000000 | 419 | 4 |
| 0.993615 6.648731 | 0.338175 -0.151852 1.346172 | 1.000000 | 419 | -4 |
| 0.993615 6.648731 | 0.338175 -0.151852 1.346172 | 1.000000 | 419 | -104 |
================================================================================
  */

  return; 
}
// clang-format on
