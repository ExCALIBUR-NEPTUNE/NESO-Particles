#ifndef _NESO_PARTICLES_ACCESS
#define _NESO_PARTICLES_ACCESS

#include "typedefs.hpp"

namespace NESO::Particles {

/**
 * One of a pair of classes that enable indexing like foo[row][col] for an
 * object foo which stores data in a column major format.
 */
template <template <typename...> class T, typename U>
class ColumnMajorColumnAccessor {
private:
  T<U> &base;
  const int &stride;
  const int &rowx;

public:
  /**
   *  Create new instance to underlying data structure of type `T<U>` which
   * stores 2D data in a column major format using a given stride.
   *
   *  @param base Object to index into.
   *  @param stride Number of rows in the 2D object.
   *  @param rowx Row to access.
   */
  ColumnMajorColumnAccessor(T<U> &base, const int &stride, const int &rowx)
      : base(base), stride(stride), rowx(rowx){};

  /**
   * Access element at column, row is provided in the constructor.
   *
   * @param colx Column to access.
   */
  inline U &operator[](const int &colx) {
    return this->base[colx * this->stride + this->rowx];
  };
};

/**
 * One of a pair of classes that enable indexing like foo[row][col] for an
 * object foo which stores data in a column major format.
 */
template <template <typename...> class T, typename U>
class ColumnMajorRowAccessor {
private:
  T<U> &base;
  const int &stride;

public:
  /**
   *  Create new instance to underlying data structure of type `T<U>` which
   * stores 2D data in a column major format using a given stride.
   *
   *  @param base Object to index into.
   *  @param stride Number of rows in the 2D object.
   */
  ColumnMajorRowAccessor(T<U> &base, const int &stride)
      : base(base), stride(stride){};

  /**
   * Returns a ColumnMajorColumnAccessor instance which defines a subscript
   * operator that can be used to access any column of the specified row.
   *
   * @param rowx Row to provide access to.
   */
  inline ColumnMajorColumnAccessor<T, U> operator[](const int &rowx) {
    return ColumnMajorColumnAccessor<T, U>{this->base, this->stride, rowx};
  };
};

/**
 * One of a pair of classes that enable indexing like foo[row][col] for an
 * object foo which stores data in a column major format.
 */
template <typename T> class RawPointerColumnMajorColumnAccessor {
private:
  T *d_ptr;
  const int stride;
  const int rowx;

public:
  /**
   * Create new instance to access data through a pointer which
   * stores 2D data in a column major format using a given stride.
   *
   *  @param base pointer to index into.
   *  @param stride Number of rows in the 2D object.
   *  @param rowx Row to access.
   */
  RawPointerColumnMajorColumnAccessor(T *d_ptr, const int stride,
                                      const int rowx)
      : d_ptr(d_ptr), stride(stride), rowx(rowx){};

  /**
   * Access element at column, row is provided in the constructor.
   *
   * @param colx Column to access.
   */
  inline T &operator[](const int &colx) {
    return d_ptr[colx * this->stride + this->rowx];
  };
};

/**
 * One of a pair of classes that enable indexing like foo[row][col] for an
 * object foo which stores data in a column major format.
 */
template <typename T> class RawPointerColumnMajorRowAccessor {
private:
  T *d_ptr;
  const int stride;

public:
  /**
   * Create new instance to access data through a pointer which
   * stores 2D data in a column major format using a given stride.
   *
   *  @param base Object to index into.
   *  @param stride Number of rows in the 2D object.
   */
  RawPointerColumnMajorRowAccessor(T *d_ptr, const int stride)
      : d_ptr(d_ptr), stride(stride){};

  /**
   * Returns a RawPointerColumnMajorColumnAccessor instance which defines a
   * subscript operator that can be used to access any column of the specified
   * row.
   *
   * @param rowx Row to provide access to.
   */
  inline RawPointerColumnMajorColumnAccessor<T> operator[](const int rowx) {
    return RawPointerColumnMajorColumnAccessor<T>{this->d_ptr, this->stride,
                                                  rowx};
  };
};

} // namespace NESO::Particles

#endif
