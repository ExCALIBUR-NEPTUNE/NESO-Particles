#ifndef _NESO_PARTICLES_ND_INDEX_H_
#define _NESO_PARTICLES_ND_INDEX_H_

#include "../typedefs.hpp"

namespace NESO::Particles {

/**
 * Generic helper type for indexing into containers with multiple dimensions.
 * If the N-dimensional object is indexed with indices (i0, i1, ..., iN-1) then
 * iN-1 is the fastest running dimension and i0 is the slowest.
 */
template <std::size_t N> struct NDIndex {
protected:
  template <std::size_t DIM, typename U>
  inline void linearise_inner(INT &index, const U &arg0) const {
    static_assert(DIM + 1 == N, "Unexpected number of args");
    index *= this->shape[DIM];
    index += arg0;
  }

  template <std::size_t DIM, typename U, typename... T>
  inline void linearise_inner(INT &index, const U &arg0, T... args) const {
    index *= this->shape[DIM];
    index += arg0;
    linearise_inner<DIM + 1>(index, args...);
  }

public:
  /// The size of each dimension.
  int shape[N];

  /**
   * Get convert a tuple index into a linear index. Indices are ordered from
   * slowest to fastest.
   *
   * @param args Parameter pack of size N representing the index as a tuple.
   * @returns Linear index.
   */
  template <typename... T> inline INT get_linear_index(T... args) const {
    static_assert(sizeof...(T) == N, "Incorrect number of arguments.");
    INT index = 0;
    this->linearise_inner<0>(index, args...);
    return index;
  }

  /**
   * @returns The total size of the index set.
   */
  inline INT size() const {
    INT s = this->shape[0];
    for (int ix = 1; ix < N; ix++) {
      s *= this->shape[ix];
    }
    return s;
  }
};

} // namespace NESO::Particles

#endif
