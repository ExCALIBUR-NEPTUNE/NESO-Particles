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


Paraview
========

H5Part trajectories can be natively opened in Paraview. 
Opening a ``.h5part`` file should only involve opening the file in the Paraview file open dialogue or passing the file as a command line argument to Paraview. 
Note that by default the render view in Paraview may select a 3D view for a 2D simulation, in this case toggle the render view into 2D mode by clicking the small button with the text "3D" at the top left of the rendered view (on the toolbar directly below the tab bar).


.. [H5PART] H5Part: A Portable High Performance Parallel Data Interface for Particle Simulations, doi: 10.1109/PAC.2005.1591740. `IEEE <https://ieeexplore.ieee.org/document/1591740>`_ `CERN <https://accelconf.web.cern.ch/p05/papers/fpat083.pdf>`_.
