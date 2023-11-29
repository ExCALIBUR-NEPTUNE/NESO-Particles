#ifndef _NESO_PARTICLES_ACCESS_DESCRIPTORS_H_
#define _NESO_PARTICLES_ACCESS_DESCRIPTORS_H_
/**
 *  Types and functions relating to access descriptors for loops.
 */
namespace NESO::Particles::Access {

/**
 * Generic base type for an access descriptor around an object of type T.
 */
template <typename T> struct AccessGeneric { T obj; };

/**
 *  Read access descriptor.
 */
template <typename T> struct Read : AccessGeneric<T> {};

/**
 *  Write access descriptor.
 */
template <typename T> struct Write : AccessGeneric<T> {};

/**
 *  Atomic add access descriptor.
 */
template <typename T> struct Add : AccessGeneric<T> {};

/**
 *  Helper function that allows a loop to be constructed with a read-only
 *  parameter passed like:
 *
 *    Access::read(object)
 *
 * @param t Object to pass with read-only access.
 * @returns Access::Read object that wraps passed object.
 */
template <typename T> inline Read<T> read(T t) { return Read<T>{t}; }

/**
 *  Helper function that allows a loop to be constructed with a write
 *  parameter passed like:
 *
 *    Access::write(object)
 *
 * @param t Object to pass with write access.
 * @returns Access::Write object that wraps passed object.
 */
template <typename T> inline Write<T> write(T t) { return Write<T>{t}; }

/**
 *  Helper function that allows a loop to be constructed with a write
 *  parameter which is atomic addition passed like:
 *
 *    Access::add(object)
 *
 * @param t Object to pass with atomic add access.
 * @returns Access::Add object that wraps passed object.
 */
template <typename T> inline Add<T> add(T t) { return Add<T>{t}; }

} // namespace NESO::Particles::Access

#endif
