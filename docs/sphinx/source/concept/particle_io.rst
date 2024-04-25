***********
Particle IO
***********

Standard Output
===============

Particle properties can be printed to stdout on each rank by calling `ParticleGroup::print`.
This print method takes as input the `Sym` instances that correspond to the properties to print.

.. code-block:: cpp
   :caption: Example printing particle data. 

    particle_group->print(
      Sym<REAL>("P"),
      Sym<REAL>("V"),
      Sym<REAL>("Q"),
      Sym<INT>("CELL_ID"),
      Sym<INT>("ID")
    );
    /*
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

HDF5
====

We use the [H5PART]_ file format as an HDF5 output format for particle trajectories. 
This format allows for post processing in languages with HDF5 bindings. 
We provide a class `H5Part` to facilitate writing particle properties to file at each time step. All methods of the `H5Part` class are collective over the MPI communicator of the `ParticleGroup`.

.. code-block:: cpp
   :caption: Example creating a H5Part trajectory. 

    H5Part h5part(
        "particle_trajectory.h5part", 
        particle_group, 
        Sym<REAL>("P"),  // The positions are always recorded as part of the 
                         // format. Passing them again here may trade space
                         // for convenience.
        Sym<REAL>("V"), 
        Sym<INT>("ID")   // Particle data is unordered between time steps. To
                         // track a particular particle through the trajectory 
                         // the output must contain a unique identifier for 
                         // each particle.
    );

    // Write the first time step.
    h5part.write();
    // Write a second time step.
    h5part.write();

    // Finally close the h5part file.
    h5part.close();


The `write` call may optionally be passed an integer to indicate the time step, this may be incompatible with the native H5Part reader in Paraview.
Note that the `close` method must be called for the file to be readable.

.. code-block:: python
   :caption: Prototype Python implementation to read H5Part files for post processing.

   import h5py
   import numpy as np

   h = h5py.File("particle_trajectory.h5part", "r")
   # NB these keys are not in chronological order - ordered by string value
   unordered_keys = h.keys()
   # Reorder into chronological order
   timestep_ordered_keys = sorted(unordered_keys, key=lambda x: int(x.split("#")[-1]))
   
   # Step#0
   # 	 <KeysViewHDF5 ['ID_0', 'P_0', 'P_1', 'V_0', 'V_1', 'x', 'y']>
   # Step#1
   # 	 <KeysViewHDF5 ['ID_0', 'P_0', 'P_1', 'V_0', 'V_1', 'x', 'y']>
   
   for stepx in timestep_ordered_keys:
       print(stepx)
       hx = h[stepx]
       print("\t", hx.keys())
       
       # Get the first component of V as a numpy array. This array contains the
       # V[0] values for all particles. Note that the ordering of particle data is
       # consistent between datasets within a time step. It is NOT consistent
       # between time steps (for example consider particles which are added and
       # removed from the system). To reorder particles for analysis ensure that
       # the particle system has a unique ID for particles.
       array_x = np.array(hx.get("V_0"))


Paraview
========

H5Part trajectories can be natively opened in Paraview. 
Opening a ``.h5part`` file should only involve opening the file in the Paraview file open dialogue or passing the file as a command line argument to Paraview. 
Note that by default the render view in Paraview may select a 3D view for a 2D simulation, in this case toggle the render view into 2D mode by clicking the small button with the text "3D" at the top left of the rendered view (on the toolbar directly below the tab bar).


.. [H5PART] H5Part: A Portable High Performance Parallel Data Interface for Particle Simulations, doi: 10.1109/PAC.2005.1591740. `IEEE <https://ieeexplore.ieee.org/document/1591740>`_ `CERN <https://accelconf.web.cern.ch/p05/papers/fpat083.pdf>`_.
