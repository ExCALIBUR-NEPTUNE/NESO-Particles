#ifndef _NESO_PARTICLES_CONTAINERS_RESOURCE_STACK_INTERFACE_HPP_
#define _NESO_PARTICLES_CONTAINERS_RESOURCE_STACK_INTERFACE_HPP_

#include <memory>

namespace NESO::Particles {

/**
 * Interface definition for types that create/destroy/cleanup resources stored
 * in a ResourceStack.
 */
template <typename T> struct ResourceStackInterface {

  virtual ~ResourceStackInterface() = default;

  /**
   * Method which is called to create a new instance of the resource type. This
   * method is called when the stack is empty and a new instance must be
   * created.
   */
  virtual inline std::shared_ptr<T> construct() = 0;

  /**
   * Method which is called when a resource is no longer required and is about
   * to be destroyed. This interface allows any cleanup methods to be called on
   * the resource before it is destroyed.
   *
   * @param resource Resource prior to destruction.
   */
  virtual inline void free(std::shared_ptr<T> &resource) = 0;

  /**
   * Method which is called on each resource when the resource is released back
   * to the ResourceStack. This method prepares the resource for the next usage.
   */
  virtual inline void clean(std::shared_ptr<T> &resource) = 0;
};

} // namespace NESO::Particles

#endif
