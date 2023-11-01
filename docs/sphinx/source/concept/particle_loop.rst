*************
Particle Loop
*************

Introduction
============

In NESO-Particles (NP) a "Particle Loop" is a loop over all particles in a collection of particles.
For each particle in the iteration set a user provided kernel is executed.
This user provided kernel may access the data stored on each particle (ParticleDats) and access global data.
For each data structure accessed by the kernel the user must provide an access descriptor.
This access descriptor describes exactly how the data will be accessed, e.g. read-only.
The provided kernel must be written such that the result of execution of the loop is independent of the execution order of the loop, i.e. parallel and unsequenced in C++ terminology.
The particle loop abstraction follows the particle loop abstraction in [SAUNDERS2018]_.

Advection Example
~~~~~~~~~~~~~~~~~

In the code listings below we present and describe an example ParticleLoop.
This example loop assumes that a ParticleGroup has been created with "P" and "V" particle properties which are real valued and have at least two components.

.. literalinclude:: ../example_sources/example_particle_loop_0.hpp
   :language: cpp
   :caption: Example of a ParticleLoop which performs a simple advection operation.


.. literalinclude:: ../example_sources/example_particle_loop_0_nc.hpp
   :language: cpp
   :caption: Duplicate of the advection ParticleLoop example with the comments removed.


Additional Data Structures
==========================

In the advection example we described a particle loop that only accessed particle data.
In addition to particle data particle loops may access additional data structures which are not tied to a particular particle.

The "LocalArray" data structure is an array type which is local to the MPI rank on which it is defined, i.e. no MPI communication occurs. 
The local array type can be considered similar to the std::vector type.
The "GlobalArray" type is an array type where a copy of the array is stored on each MPI rank and access to the array is collective over the MPI communicator on which the global array is defined.
The "CellDatConst" type is a matrix type defined with fixed size (the "Const" part of the name) and data type for each cell in a mesh. 

These additional data structures must be passed to a particle loop with an access mode which is commutative.
In practice only commutative access modes are defined for these data structures and attempting to pass them to a particle loop with an inappropriate access descriptor will result in a compile time error.

LocalArray
~~~~~~~~~~

.. literalinclude:: ../example_sources/example_particle_loop_local_array.hpp
   :language: cpp
   :caption: Particle loop example where a local array is incremented by each particle and another local array is read by each particle. 


.. [SAUNDERS2018] A domain specific language for performance portable molecular dynamics algorithms. `CPC <https://doi.org/10.1016/j.cpc.2017.11.006>`_ , `arXiv <https://arxiv.org/abs/1704.03329>`_.
