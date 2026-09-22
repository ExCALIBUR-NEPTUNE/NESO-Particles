******************
Particle Pair Loop
******************

Introduction
============

Particle Pair Loops are a looping construct that visits two particles for each kernel invocation.
The form of the pair loop is similar to the particle loop; there is an iteration set, a kernel and a set of arguments with access descriptors.

The iteration set defines how the pairs are defined.
In the example below the iteration set is a pair list which explicitly contains the pairs of particles.
The second component of the pair loop is the pairwise kernel.
Like particle loop kernels, the pair wise kernel often takes the form of a C++ lambda function.

.. warning::
   The kernel function is compiled for and ran on the compute device. Variables
   used by the kernel, e.g. captured by a lambda, must be copyable to the
   device. Kernels which are lambda functions, like the examples here, should
   use a copy capture (``=``) rather than a capture by reference (``&``). Data
   in host allocated memory should be passed to the kernel via a construct such
   as ``LocalArray``.

Finally we pass the arguments to the pair loop.
As for particle loop, there is a one-to-one correspondence between the arguments listed after the kernel in the pair loop call and the parameters defined for the kernel function.
Some constructs can only be passed to particle loops and not particle pair loops and the reverse holds for some pair loop only constructs.

Arguments are wrapped in access descriptors and passed to the particle pair loop call.
In addition to the access descriptors which describe the access type, and hence kernel parameter type, pair loop arguments are wrapped in an additional access descriptor which specifies which particle the argument corresponds to.
These additional access descriptors are ``Access::A()`` and ``Access::B()`` where ``A`` refers to the first particle of the pair and ``B`` refers to the second particle of the pair.

Example Pair Loop
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../example_sources/example_particle_pair_loop_simple.hpp
   :language: cpp
   :caption: Example of a Particle Pair Loop which swaps the quantity Q if the particles are close.



