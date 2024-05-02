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

Prefixes
========

This is a heterogeneous code where memory is allocated on host and device a variable may exist on both the host and device at the same time and hence it is useful to use prefixes to differentiate between the instances of the variable.
We use variable prefixes to indicate which device a variable corresponds to as follows:

* ``h_`` Variable exists on the host. In the case of a pointer the memory region is host allocated.
* ``d_`` Variable exists on the device. In the case of a pointer the memory region is device allocated.
* ``s_`` Variable exists on the host and device. In the case of a pointer the memory region is allocated as shared.
* ``b_`` Variable is a SYCL buffer.
* ``a_`` Variable is a SYCL accessor.
* ``k_`` Variable is a local copy of a variable to pass into a kernel by copy.

Example
=======
 .. code-block:: cpp

    // Capitialised macro names preferred
    #define ABS(x) ((x) < 0 ? (-(x)) : (x))
    
    // "g_" If a global really must be used.
    extern const int g_please_avoid;
    
    /**
     * PascalCase class name.
     * Description of class in Doxygen format.
     */
    class MyAmazingClass {
    
    protected:
      
      /// Description of protected variable.
      int protected_int;
    
    public:
    
      /**
       * Doxygen compatible description of class and arguments.
       *   
       * @param arg0 Description of arg0.
       * @param arg1 Description of arg1.
       */
      MyAmazingClass(Type0 arg0, Type1 arg1){

      }
    
      /**
       * Snake Case method name. With Doxygen compatible description. Please
       * use [in, out] if an argument is also a return variable.
       *
       * @param[in] arg0 Description of arg0.
       * @param[inm out] arg1 Description of arg1.
       */
      inline double get_some_value(Type0 arg0, Type1 &arg1){
        // Implementation
      }
    
      // Snake Case attribute where a descriptive name is appropriate.
      int scalar_int_value;
    
      // Attribute where the name mirroring the maths with a capital may
      // significantly help connecting maths to code.
      double kB;
    
      // "d_" for device allocated memory (CUDA inspired).
      double * d_x;
    
      // "h_" for host allocated memory (i.e. for when there is a h_foo and a
      // d_foo).
      double * h_x;
    
      // "s_" for allocated memory that is shared between host and device.
      double * s_x;
    
      // "b_": SYCL buffers
      sycl::buffer b_x;
    
      // "k_" for local temporaries used in kernels.
      // Note the "this->" for variables with type scope.
      inline void foo(){
    
        const int k_scalar_int_value = this->scalar_int_value;
        const double k_some_value = this->get_some_value();
    
        q.submit([&](auto &h) {
          h.parallel_for(range<1>(size), [=](id<1> idx) {
            // use k_scalar_int_value, k_some_value
        });
       }).wait();
    
      }
    }
    
    // Shared pointers to types should be named as follows
    typedef std::shared_ptr<MyAmazingClass> MyAmazingClassSharedPtr;


