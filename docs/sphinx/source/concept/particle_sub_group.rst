******************
Particle Sub Group
******************

Introduction
============

A "ParticleSubGroup" is a wrapper class around a ParticleGroup which selects a subset of the particles in the ParticleGroup.
A particle sub group may be passed to a ParticleLoop to define the iteration set for the ParticleLoop.

Creation
========

In the listing below we demonstrate the creation of a ParticleSubGroup which is defined as the subset of particles in a ParticleGroup which have an even integer in the first component of a ParticleDat called "ID".
This subset selection is specified by the user by providing a kernel lambda.
The creation of the sub group is automatically performed as required by objects which accept a ParticleSubGroup as input.

This subset selection procedure is automatically performed each time an up-to-date ParticleSubGroup is required. i.e. users may update the particle properties used in the subset selection process, or move particles between cells and MPI ranks, and expect the ParticleSubGroup to be updated to reflect these changes.
This automatic reevaluation process assumes that particle data access is through the methods provided by NESO-Particles. 
If these data access methods are circumvented to modify particle data then no guarantee is given that a ParticleSubGroup correctly represents the desired subset of particles.

.. literalinclude:: ../example_sources/example_particle_sub_group_creation.hpp
   :language: cpp
   :caption: Create a ParticleSubGroup from the subset of particles where the first component of the ID ParticleDat is even.

Particle Loop
=============

A ParticleSubGroup is a valid iteration set for a ParticleLoop.
In the following example we execute a particle loop over all particles with an even ID.

.. literalinclude:: ../example_sources/example_particle_sub_group_loop.hpp
   :language: cpp
   :caption: Create and execute a ParticleLoop over particles which are members of a ParticleSubGroup.



