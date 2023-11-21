#ifndef _NESO_PARTICLES_TUPLE_H_
#define _NESO_PARTICLES_TUPLE_H_
#include <cstdlib>

namespace NESO::Particles::Tuple {

template <std::size_t INDEX, typename U> struct TupleImpl {
  U value;
  TupleImpl() = default;
  U &get() { return value; }
  const U &get_const() const { return value; }
};

template <size_t INDEX, typename... V> struct TupleBaseRec {
  TupleBaseRec() = default;
};

template <size_t INDEX, typename U, typename... V>
struct TupleBaseRec<INDEX, U, V...> : TupleImpl<INDEX, U>,
                                      TupleBaseRec<INDEX + 1, V...> {
  TupleBaseRec() = default;
};

template <typename U, typename... V> struct Tuple : TupleBaseRec<0, U, V...> {
  Tuple() = default;
};

template <size_t INDEX, typename T, typename... U> struct GetIndexType {
  using type = typename GetIndexType<INDEX - 1, U...>::type;
};

template <typename T, typename... U> struct GetIndexType<0, T, U...> {
  using type = T;
};

template <size_t INDEX, typename... U> auto &get(Tuple<U...> &u) {
  return static_cast<
             TupleImpl<INDEX, typename GetIndexType<INDEX, U...>::type> &>(u)
      .get();
}

template <size_t INDEX, typename... U> const auto &get(const Tuple<U...> &u) {
  return static_cast<const TupleImpl<
      INDEX, typename GetIndexType<INDEX, U...>::type> &>(u)
      .get_const();
}

template <size_t...> struct IntSequence {};

template <size_t N, size_t... S> struct GenerateIntSequence {
  using type = typename GenerateIntSequence<N - 1, N - 1, S...>::type;
};

template <size_t... S> struct GenerateIntSequence<0, S...> {
  using type = IntSequence<S...>;
};

template <typename KERNEL, size_t... S, typename... ARGS>
auto apply_inner(KERNEL &kernel, IntSequence<S...>, Tuple<ARGS...> &args) {
  return kernel(get<S>(args)...);
}

template <typename KERNEL, typename... ARGS>
auto apply(KERNEL kernel, Tuple<ARGS...> &args) {
  return apply_inner(
      kernel, typename GenerateIntSequence<sizeof...(ARGS)>::type(), args);
}

} // namespace NESO::Particles::Tuple

#endif
