*****************
Particle Products
*****************

Overview
========

We consider a scenario where N existing particles are to each create M new particles.
These new particles will ultimately be added to the `ParticleGroup` which contains the original N particles and hence each of the NM new particles require all of the properties to be set to sensible values.
The particle properties can always be modified after the new particles are added to the `ParticleGroup` by using a `ParticleLoop` therefore here we discuss methods to define particle properties before the particles are added.


Particles are created via the `DescendantProducts` data structure which provides space for each of the new particle properties.
These particle properties are accessed from each of the N parent particles from a `ParticleLoop` kernel.
Once a `DescendantProducts` instance is populated with values those values can be used to create new particles in the parent `ParticleGroup` by calling `add_particles_local` with the new properties.

Properties of New Particles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a new `DescendantProducts` instance is created the number of output products M per existing particle is specified along with the REAL and INT particle properties which will be explicitly set in the `ParticleLoop` kernel.
This kernel must also set the parent particle of the new product particle by calling `set_parent` for the product particle to be included when `add_particles_local` is called.

The `set_parent` mechanism provides two functions, firstly it defines the product as a product which should be included when `add_particles_local` is called.
This `set_parent` call can be intentionally omitted to mask off any number of the M products which ultimately were not required.
Secondly for particle properties which are not defined in the `DescendantProducts` constructor the `set_parent` call defines the parent particle from which these properties should be copied.

The destination `ParticleGroup` for the new particle products has a set of particle properties for each of the particles in the `ParticleGroup`.
When the `DescendantProducts` are are added via `add_particles_local` there are two options.
If the property is explicitly defined in the `DescendantProducts` instance then the component values, for all particles and all components of that property, are copied from the `DescendantProducts` into the corresponding `ParticleDats` in the `ParticleGroup`.
If a property is defined in the `ParticleGroup` and not in the `DescendantProducts` then for all new particle entries in the `DescendantProducts` the component values for that property are copied from the parents specified in the `DescendantProducts`.

Note that the decision to copy property values from a parent particle or from a `DescendantProducts` entry is taken property wise for all particles for all components of the property.
If finer control is required, e.g. to inherit particle properties on a per particle or per component level, then the user should specify the property in the `DescendantProducts` instance and populate the new properties for all particles and components in the `ParticleLoop` kernel which is responsible for populating the entries in the `DescendantProducts` instance.

Property Ordering
~~~~~~~~~~~~~~~~~

In the `ParticleLoop` kernel which sets the properties of the new products the property, and component, are specified by an integer index instead of the `Sym` objects used in host code.
The ordering of the properties is defined as the order in which particle properties are specified for the `DescendantProducts` specification.
This ordering is contiguous within all properties of the same data type, e.g. if two INT properties are specified then the integer properties are indexed with 0 and 1 and if three REAL properties are specified then the real valued properties are indexed 0,1 and 2.
Interlacing of INT and REAL properties in the specification is ignored, the properties are indexed within the set of properties that have the same type in the order they are specified.

Example
=======

.. literalinclude:: ../example_sources/example_particle_descendant_products.hpp
   :language: cpp
   :caption: Example creating particles in a ParticleLoop and adding the new particle to a ParticleGroup. 


