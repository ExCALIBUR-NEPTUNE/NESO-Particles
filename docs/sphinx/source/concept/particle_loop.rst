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

The particle loop can be executed asynchronously via calls to "submit" and "wait".

.. literalinclude:: ../example_sources/example_particle_loop_0_nc.hpp
   :language: cpp
   :caption: Duplicate of the advection ParticleLoop example with the comments removed and asynchronous execution.

.. list-table:: ParticleDat<T> Access
   :header-rows: 1

   * - Access Descriptors
     - Kernel Types
     - Notes
   * - Read
     - Access::ParticleDat::Read<T>
     - Components accessed for the particle via .at or subscript which returns a const reference.
   * - Write
     - Access::ParticleDat::Write<T>
     - Components accessed for the particle via .at or subscript which returns a modifiable reference.


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

The local array type is local to each MPI rank. No communication between MPI ranks is performed by the particle loop. 
The local array can be accessed in particle loops with "read" and "add" access modes.

.. literalinclude:: ../example_sources/example_particle_loop_local_array.hpp
   :language: cpp
   :caption: Particle loop example where a local array is incremented by each particle and another local array is read by each particle. 

.. list-table:: LocalArray<T> Access
   :header-rows: 1

   * - Access Descriptors
     - Kernel Types
     - Notes
   * - Read
     - Access::LocalArray::Read<T>
     - Components accessed for each array element via .at or subscript which returns a const reference.
   * - Add
     - Access::LocalArray::Add<T>
     - fetch_add(index, value) atomically increments the element referenced by the index with the passed value. Returns the previous value stored at the index. Users may assume that the memory region accessed is the same for all invocations of the kernel. Hence adding 1 for each particle gives a total ordering for each participating calls.


GlobalArray
~~~~~~~~~~~

Unlike the local array, a global array is intended to have identical values across MPI ranks.
The global array can be accessed in "read" and "add" access modes.
When accessed with "add" access modes the particle loop must be executed collectively across all MPI ranks.
On completion of the loop the local entries of each global array are globally combined (Allreduce).

.. literalinclude:: ../example_sources/example_particle_loop_global_array.hpp
   :language: cpp
   :caption: Particle loop example where a global array is incremented by each particle. 

.. list-table:: GlobalArray<T> Access
   :header-rows: 1

   * - Access Descriptors
     - Kernel Types
     - Notes
   * - Read
     - Access::GlobalArray::Read<T>
     - Components accessed for each array element via .at or subscript which returns a const reference.
   * - Add
     - Access::GlobalArray::Add<T>
     - add(index, value) atomically increments the element referenced by the index with the passed value. On loop execution completion values are all-reduced across all MPI ranks.


CellDatConst
~~~~~~~~~~~~

The CellDatConst data structure stores a constant sized matrix per mesh cell.
When accessed from a particle loop both the read and add access descriptors expose the matrix which corresponds to the cell in which the particle resides.

.. literalinclude:: ../example_sources/example_particle_loop_cell_dat_const.hpp
   :language: cpp
   :caption: Particle loop example where a CellDatConst is accessed by the ParticleLoop. 

.. list-table:: CellDatConst<T> Access
   :header-rows: 1

   * - Access Descriptors
     - Kernel Types
     - Notes
   * - Read
     - Access::CellDatConst::Read<T>
     - Components accessed for each array element via .at or subscript which returns a const reference.
   * - Add
     - Access::CellDatConst::Add<T>
     - fetch_add(index, value) atomically increments the element referenced by the index with the passed value. Returns the previous value stored at the index. Users may assume that the memory region accessed is the same for all invocations of the kernel. Hence adding 1 for each particle gives a total ordering for each participating calls.


.. [SAUNDERS2018] A domain specific language for performance portable molecular dynamics algorithms. `CPC <https://doi.org/10.1016/j.cpc.2017.11.006>`_ , `arXiv <https://arxiv.org/abs/1704.03329>`_.
