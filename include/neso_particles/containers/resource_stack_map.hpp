#ifndef _NESO_PARTICLES_CONTAINERS_RESOURCE_STACK_MAP_HPP_
#define _NESO_PARTICLES_CONTAINERS_RESOURCE_STACK_MAP_HPP_

#include "../typedefs.hpp"
#include "resource_stack.hpp"
#include "resource_stack_interface.hpp"
#include <typeindex>
#include <unordered_map>

namespace NESO::Particles {

/**
 * Holds a map from type info to ResourceStackBase pointers.
 */
class ResourceStackMap {
protected:
  std::unordered_map<std::type_index, std::shared_ptr<ResourceStackBase>>
      map_to_resources;

public:
  ~ResourceStackMap() = default;
  ResourceStackMap() = default;

  /**
   * Free all of the held resources.
   */
  inline void free() {
    for (auto ix : this->map_to_resources) {
      ix.second->free();
    }
    this->map_to_resources.clear();
  }

  /**
   * Does a ResourceStack exist for a key.
   *
   * @param key Key to test if a resource stack exists for.
   * @returns True if the resource stack exists in the map.
   */
  inline bool exists(const std::type_index key) {
    return this->map_to_resources.count(key);
  }

  /**
   * Set a ResourceStack for a key.
   *
   * @param key Key to set the resource stack for.
   * @param resource_stack New resource stack to set for the key.
   */
  template <typename T>
  inline void set(const std::type_index &key,
                  std::shared_ptr<ResourceStack<T>> resource_stack) {
    NESOASSERT(!this->exists(key),
               "Passed resource_stack already exists in this datastructure.");
    auto ptr = std::dynamic_pointer_cast<ResourceStackBase>(resource_stack);
    NESOASSERT(ptr != nullptr,
               "Cannot cast the passed resource_stack to ResourceStackBase.");
    this->map_to_resources[key] = ptr;
  }

  /**
   * Get the resource stack object for a key and a type.
   *
   * @param key Type index key to retrieve object for.
   * @returns ResourceStack for the key cast to type std::shared_ptr<T>.
   */
  template <typename T>
  inline std::shared_ptr<T> get(const std::type_index key) {
    NESOASSERT(this->exists(key),
               "Attempt to retrieve a ResourceStack that does not exist.");
    return this->map_to_resources.at(key);
  }

  /**
   * Does a ResourceStack exist for a key.
   *
   * @param key Key to test if a resource stack exists for.
   * @returns True if the resource stack exists in the map.
   */
  template <typename U> inline bool exists(U) {
    return this->map_to_resources.count(typeid(U));
  }

  /**
   * Set a ResourceStack for a key.
   *
   * @param key Key to set the resource stack for.
   * @param resource_stack New resource stack to set for the key.
   */
  template <typename T, typename U>
  inline void set(const U u, std::shared_ptr<ResourceStack<T>> resource_stack) {
    NESOASSERT(!this->exists(u),
               "Passed resource_stack already exists in this datastructure.");
    auto ptr = std::dynamic_pointer_cast<ResourceStackBase>(resource_stack);
    NESOASSERT(ptr != nullptr,
               "Cannot cast the passed resource_stack to ResourceStackBase.");
    this->map_to_resources[typeid(U)] = ptr;
  }

  /**
   * Get the resource stack object for a key and a type.
   *
   * @param key Type index key to retrieve object for.
   * @returns ResourceStack for the key cast to type std::shared_ptr<T>.
   */
  template <typename T, typename U> inline std::shared_ptr<T> get(const U u) {
    NESOASSERT(this->exists(u),
               "Attempt to retrieve a ResourceStack that does not exist.");

    auto ptr_map = this->map_to_resources.at(typeid(u));
    NESOASSERT(ptr_map != nullptr, "Map pointer is already NULL.");
    auto ptr = std::dynamic_pointer_cast<T>(ptr_map);
    NESOASSERT(ptr != nullptr, "Could not cast to ResourceStack.");
    return ptr;
  }
};

/**
 * Helper function to create a ResourceStack.
 * T is the type of the underlying resource, i.e. ResourceStack<T>.
 * U is the type of the interface.
 *
 * @param args Arguments to pass to resource stack interface constructor.
 * @returns New resource stack.
 */
template <typename T, typename U, typename... ARGS>
inline std::shared_ptr<ResourceStack<T>> create_resource_stack(ARGS &&...args) {
  auto tmp_interface = std::make_shared<U>(std::forward<ARGS...>(args)...);
  return std::make_shared<ResourceStack<T>>(tmp_interface);
}

/**
 * Helper function to get a ResourceStack from a ResourceStackMap or create one
 * if needed with the supplied arguments.
 * T is the type of the underlying resource, i.e. ResourceStack<T>.
 * U is the type of the interface.
 * V is the ResourceStackMap key type.
 *
 * @param resource_stack_map ResourceStackMap to index into.
 * @param key Type key for resource stack.
 * @param args Arguments to pass to resource stack interface constructor.
 * @returns A ResourceStack.
 */
template <typename T, typename U, typename V, typename... ARGS>
inline std::shared_ptr<ResourceStack<T>>
get_resource_stack(std::shared_ptr<ResourceStackMap> resource_stack_map, V key,
                   ARGS &&...args) {
  if (!resource_stack_map->exists(key)) {
    resource_stack_map->set(key, create_resource_stack<T, U>(args...));
  }
  return resource_stack_map->get<ResourceStack<T>>(key);
}

/**
 * Helper function to get a resource from a ResourceStackMap.
 * T is the type of the underlying resource, i.e. ResourceStack<T>.
 * U is the type of the interface.
 * V is the ResourceStackMap key type.
 *
 * @param resource_stack_map ResourceStackMap to index into.
 * @param key Type key for resource stack.
 * @param args Arguments to pass to resource stack interface constructor.
 * @returns A std::shared_ptr to a resource.
 */
template <typename T, typename U, typename V, typename... ARGS>
inline std::shared_ptr<T>
get_resource(std::shared_ptr<ResourceStackMap> resource_stack_map, V key,
             ARGS &&...args) {
  return get_resource_stack<T, U>(resource_stack_map, key, args...)->get();
}

/**
 * Helper function to restore a resource.
 * T is the type of the underlying resource, i.e. ResourceStack<T>.
 * U is the type of the interface.
 *
 * @param resource_stack_map ResourceStackMap to index into.
 * @param key Type key for resource stack.
 * @param resource Resource to restore.
 */
template <typename V, typename T>
inline void
restore_resource(std::shared_ptr<ResourceStackMap> resource_stack_map, V key,
                 std::shared_ptr<T> &resource) {
  resource_stack_map->get<ResourceStack<T>>(key)->restore(resource);
}

} // namespace NESO::Particles

#endif
