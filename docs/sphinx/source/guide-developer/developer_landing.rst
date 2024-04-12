***********
Style Guide
***********

This is a style guide for NESO-Particles. It is almost certainly incomplete.
Please follow the existing source files where possible for style questions not answered by this section.

Files
=====

* Give files names which are lower case snake case with a ``.hpp`` file extension.
* Include guards should define a variable of the form ``_NESO_PARTICLES_<filename>_H_`` where ``<filename>`` is the filename in screaming snake case.
* If a file is created to contain a class then please name the file to follow the class name, e.g. class ``FooBar`` is contained in file ``foo_bar.hpp``.

Class and Method Naming
=======================

* Format class names in the PascalCase style and method names in the snake_case style.
* A typedef for a shared pointer type for a class should append the suffix ``SharedPtr``, e.g. for a class ``FooBar`` the shared pointer type would be ``FooBarSharedPtr``.


Example
=======



