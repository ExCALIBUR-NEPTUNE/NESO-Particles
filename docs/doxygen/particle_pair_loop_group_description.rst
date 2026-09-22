This section contains documentation for particle pair looping.
Particle pair looping is a looping type similar to particle loop except that
the kernel operates on two particles.

.. list-table::  Constructs that are passable to Particle Pair Loops
   :header-rows: 1

   * - Construct
     - Access Modes
   * - CellDatConst
     - Read, Write, Add, Min, Max
   * - DescendantProducts
     - Write
   * - LocalArray
     - Read, Write, Add
   * - MaskArray
     - Read, Write
   * - NDLocalArray
     - Read, Write, Add, Max, Min
   * - ParticleMask
     - Read, Write
   * - SymVector
     - Read, Write
   * - ParticlePairLoopIndex
     - Read
   * - KernelRNG
     - Read
   * - TupleRNG
     - Read

