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

In the code listings below we present and describe an example ParticleLoop.
This example loop assumes that a ParticleGroup has been created with "P" and "V" particle properties which are real valued and have at least two components.

.. literalinclude:: ../example_sources/example_particle_loop_0.hpp
   :language: cpp
   :caption: Example of a ParticleLoop which performs a simple advection operation.


.. literalinclude:: ../example_sources/example_particle_loop_0_nc.hpp
   :language: cpp
   :caption: Duplicate of the advection ParticleLoop example with the comments removed.


.. [SAUNDERS2018] A domain specific language for performance portable molecular dynamics algorithms. `CPC <https://doi.org/10.1016/j.cpc.2017.11.006>`_ , `arXiv <https://arxiv.org/abs/1704.03329>`_.
