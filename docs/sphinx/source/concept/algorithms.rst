********************
High Level Functions
********************

We provide various high-level free functions to implement generic operations.
Typically these operations are not captured directly by the looping abstractions and attempts to implement these operations without writing a SYCL implementation may result in subpar performance.

CellDatConst Arithmetic
=======================

The ``cell_dat_const_loop_element_wise`` applies a given scalar function elementwise to the set of ``CellDatConst`` instances provided as arguments and assigns the outputs to the output ``CellDatConst``. 
The output may be one of the arguments.

.. literalinclude:: ../example_sources/example_cell_dat_const_loop_element_wise.hpp
   :language: cpp
   :caption: Example use of a elementwise loop and function application for ``CellDatConst`` instances.

Cellwise Broadcast
===================

Set the specified component and property on all particles to the value in the passed array at the index that corresponds to the cell of the particle.

.. literalinclude:: ../example_sources/example_cell_wise_broadcast.hpp
   :language: cpp
   :caption: Example use of a cellwise broadcast.







