******************************
Low Level Particle Data Access
******************************

Particle Data Access Window
===========================

Sometimes it is desirable to directly access particle data, e.g. in bespoke parallel loops for a particular operation on a particular architecture.
We provide a mechanism to directly access the pointers for particle data whilst respecting the access descriptors.
Access descriptors are used internally for processes such as cache invalidation.
Ignoring the access descriptors and using the provided pointers to access particle data outside of the access window will result in an incorrect program.

For a given `ParticleDat`, an access window is opened by a call to `direct_get` and closed by a call to `direct_restore`.
A pair of calls to `direct_get` and `direct_restore` must be made with the same access type.
The closing call to `direct_restore` must be made with the object returned from `direct_get`.
In the example below we implement a forward Euler kernel by using an access window.

.. literalinclude:: ../example_sources/example_particle_loop_low_level_0.hpp
   :language: cpp
   :caption: Example creation of an access window between comments labelled [A] and [B].




