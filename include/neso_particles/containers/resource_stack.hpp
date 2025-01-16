#ifndef _NESO_PARTICLES_CONTAINERS_RESOURCE_STACK_HPP_
#define _NESO_PARTICLES_CONTAINERS_RESOURCE_STACK_HPP_
#include <memory>
#include <stack>
#include <type_traits>

#include "../typedefs.hpp"
#include "resource_stack_interface.hpp"

namespace NESO::Particles {

/**
 * Manages a pool of objects in a stack. These objects are temporarily used by
 * external code then returned to the stack. The aim is to maintain this pool
 * instead of creating and destroying objects.
 *
 * Exactly how objects are created/destroyed/cleaned is described by an instance
 * of the ResourceStackInterface class.
 */
template <typename T> class ResourceStack {
protected:
  std::shared_ptr<ResourceStackInterface<T>> resource_stack_interface;
  std::size_t num_managed_objects;

public:
  ResourceStack(const ResourceStack &) = delete;
  ResourceStack &operator=(const ResourceStack &) = delete;
  ResourceStack(ResourceStack &&) = delete;
  ResourceStack &operator=(ResourceStack &&) = delete;

  /**
   * The destructor will attempt to call the free method for the contained
   * objects.
   */
  ~ResourceStack() { this->free(); }

  /// The underlying stack that represents the pool of objects.
  std::stack<std::shared_ptr<T>> stack;

  /**
   * Create new resource stack.
   *
   * @param resource_stack_interface Pointer to type which is dynamically
   * castable to ResourceStackInterface<T>.
   */
  template <typename U>
  ResourceStack(std::shared_ptr<U> resource_stack_interface)
      : resource_stack_interface(
            std::dynamic_pointer_cast<ResourceStackInterface<T>>(
                resource_stack_interface)),
        num_managed_objects(0) {
    static_assert(std::is_base_of<ResourceStackInterface<T>, U>::value);
    NESOASSERT(this->resource_stack_interface != nullptr,
               "Could not cast resource_stack_interface to "
               "ResourceStackInterface<T>.");
  }

  /**
   * Retrieve an object from the stack or create a new one if the stack is
   * empty. The calling function has exclusive use of the returned object until
   * the object is returned to the stack with ``restore''. Restore must be
   * called for all objects returned from .get.
   *
   * @returns Resource from the stack.
   */
  inline std::shared_ptr<T> get() {
    if (this->stack.empty()) {
      auto ptr = this->resource_stack_interface->construct();
      NESOASSERT(ptr != nullptr, "Failed to construct new resource.");
      this->num_managed_objects++;
      return ptr;
    } else {
      auto ptr = this->stack.top();
      NESOASSERT(ptr != nullptr, "Failed to get valid object from the stack.");
      this->stack.pop();
      return ptr;
    }
  }

  /**
   * Return an object to the stack. Calling this method returns all ownership of
   * the object to the stack. The calling code should no longer access the
   * object. This method must be called for all resources the instance provides.
   *
   * @param[in, out] resource Shared pointer to the resource which this class
   * returned with .get. This pointer will be set to nullptr on return.
   */
  inline void restore(std::shared_ptr<T> &resource) {
    std::shared_ptr<T> ptr = resource;
    NESOASSERT(ptr != nullptr, "Returned object is a nullptr. Calling code "
                               "should not reset the pointer.");
    resource.reset();
    resource = nullptr;
    NESOASSERT(ptr.use_count() == 1,
               "The use count on the restored object is greater than 1 which "
               "indicates a reference still exists to the object in the "
               "calling function.");
    this->resource_stack_interface->clean(ptr);
    NESOASSERT(ptr != nullptr, "Returned object is a nullptr. Cleaning "
                               "implementation should not reset the pointer.");
    this->stack.push(ptr);
  }

  /**
   * Call the free method of the ResourceStackInterface on all objects then
   * discard them.
   */
  inline void free() {
    NESOASSERT(this->stack.size() == this->num_managed_objects,
               "There are managed objects which have been reserved with .get "
               "that have not been restored with .restore.");
    while (!this->stack.empty()) {
      auto ptr = this->stack.top();
      this->resource_stack_interface->free(ptr);
      this->stack.pop();
    }
    this->num_managed_objects = 0;
  }
};

} // namespace NESO::Particles

#endif
