#ifndef _NESO_PARTICLES_COMMUNICATION_TYPEDEFS_HPP
#define _NESO_PARTICLES_COMMUNICATION_TYPEDEFS_HPP

#include "../typedefs.hpp"
#include <mpi.h>

#define _MACRO_STRING(x) #x
#define STR(x) _MACRO_STRING(x)
#define MPICHK(cmd) NESOASSERT(cmd == MPI_SUCCESS, "MPI ERROR");

namespace NESO::Particles {

namespace {
template <typename T> inline MPI_Datatype map_ctype_mpi_type_inner() {
  return MPI_DATATYPE_NULL;
}
template <> inline MPI_Datatype map_ctype_mpi_type_inner<int>() {
  return MPI_INT;
}
template <> inline MPI_Datatype map_ctype_mpi_type_inner<std::int64_t>() {
  return MPI_INT64_T;
}
template <> inline MPI_Datatype map_ctype_mpi_type_inner<std::uint64_t>() {
  return MPI_UINT64_T;
}
template <> inline MPI_Datatype map_ctype_mpi_type_inner<double>() {
  return MPI_DOUBLE;
}
template <> inline MPI_Datatype map_ctype_mpi_type_inner<float>() {
  return MPI_FLOAT;
}
} // namespace

/**
 *  For an input data type T get the matching MPI Datatype.
 *
 *  @returns MPI_Datatype that matches the input type T.
 */
template <typename T> inline MPI_Datatype map_ctype_mpi_type() {
  const MPI_Datatype t = map_ctype_mpi_type_inner<T>();
  NESOASSERT(t != MPI_DATATYPE_NULL,
             "Could not find MPI_Datatype matching passed type.");
  return t;
}

/**
 * @returns Unsigned integer MPI datatype that matches std::size_t.
 */
inline MPI_Datatype get_size_t_mpi_type() {

  static_assert((sizeof(std::size_t) == sizeof(unsigned int)) ||
                    (sizeof(std::size_t) == sizeof(unsigned long)) ||
                    (sizeof(std::size_t) == sizeof(unsigned long long)),
                "Could not find MPI datatype that matches std::size_t.");

  if constexpr (sizeof(std::size_t) == sizeof(unsigned int)) {
    return MPI_UNSIGNED;
  } else if constexpr (sizeof(std::size_t) == sizeof(unsigned long)) {
    return MPI_UNSIGNED_LONG;
  } else {
    return MPI_UNSIGNED_LONG_LONG;
  }
}

} // namespace NESO::Particles

#endif
