.. contents::

Introduction
============

NESO-Particles is a particle library designed to enable particle transport over unstructured, and potentially high-order, meshes in a performance portable implementation.
:numref:`mesh-with-coarse` illustrates an example mesh in black.
In parallel computation such unstructured meshes are typically decomposed across many MPI ranks where each rank stores and can access a relatively small proportion of the overall mesh.
In Finite Element Methods (FEM) such meshes are used in conjunction with finite elements to define function spaces on which numerical solutions to Partial Differential Equations (PDEs) can be sought.

.. _mesh-with-coarse:
.. figure:: figures/mesh_with_coarse_overlay.svg
   :class: with-border
   :height: 240 pt

   Unstructured mesh (black) with coarse overlay grid (blue).

Of particular interest is the class of Particle In Cell (PIC) algorithms that seek to numerically solve PDEs via mesh and particle based approximations.
In these PIC schemes data is transferred between particle and mesh based representations at each time step and this data transfer may account for a significant proportion of the overall runtime.
This implementation supports storing particle data on a per cell basis to allow efficient coupling between mesh and particle representations.

An additional challenge this implementation tackles is the efficient transfer of particles between the subdomains that result from the domain decomposition approach.
There are numerous methods for tracking particles between cells, none of which are computationally cheap.
Mapping arbitrary points in space to cells in the mesh, and other methods to track particles, is particularly expensive on an unstructured, and potentially high-order, mesh. 
In contrast to typical Molecular Dynamics (MD) implementations we implement particle transfer mechanisms that can transfer particles "globally" as to accommodate very fast moving particles.

Particle Data
-------------




Transfer of Particle Data Globally
----------------------------------
