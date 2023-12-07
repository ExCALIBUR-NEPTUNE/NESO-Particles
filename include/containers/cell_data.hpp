#ifndef _NESO_PARTICLES_CELL_DATA
#define _NESO_PARTICLES_CELL_DATA

#include "../compute_target.hpp"

namespace NESO::Particles {

// Forward declaration of ParticleLoop such that these classes can define
// ParticleLoop as a friend class.
template <typename KERNEL, typename... ARGS> class ParticleLoop;
template <typename T> class ParticleDatT;

/**
 * Container for the data within a single cell stored on the host. Data is
 * store column wise. The data is typically particle data for a single
 * ParticleDat for a single cell and exists as a 2D data structure.
 */
template <typename T> class CellDataT {
private:
  // std::format is C++20.......
  std::string format(INT value) {
    char buffer[128];
    const int err = snprintf(buffer, 128, "%ld", value);
    return std::string(buffer);
  }
  std::string format(REAL value) {
    char buffer[128];
    const int err = snprintf(buffer, 128, "%f", value);
    return std::string(buffer);
  }
  std::string format(char value) { return std::string(1, value); }
  SYCLTargetSharedPtr sycl_target;

public:
  /// Disable (implicit) copies.
  CellDataT(const CellDataT &st) = delete;
  /// Disable (implicit) copies.
  CellDataT &operator=(CellDataT const &a) = delete;

  /// Number of rows in the 2D data structure.
  const int nrow;
  /// Number of columns in the 2D data structure.
  const int ncol;
  /// 2D data.
  std::vector<std::vector<T>> data;

  /**
   * Create a new, empty and uninitialised container for 2D data of the
   * specified shape.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param nrow Number of rows.
   * @param ncol Number of columns.
   */
  inline CellDataT(SYCLTargetSharedPtr sycl_target, const int nrow,
                   const int ncol)
      : sycl_target(sycl_target), nrow(nrow), ncol(ncol) {
    this->data = std::vector<std::vector<T>>(ncol);
    for (int colx = 0; colx < ncol; colx++) {
      this->data[colx] = std::vector<T>(nrow);
    }
  }

  /**
   *  Subscript operator for cell data. Data should be indexed by column then
   * row. e.g. CellData cell_data; T value = *cell_data[col][row];
   */
  inline std::vector<T> &operator[](int col) { return this->data[col]; }

  /**
   *  Access data with more standard (row, column) indexing.
   *
   *  @param row Row to access.
   *  @param col Column to access.
   *  @returns reference to accessed element.
   */
  inline T &at(const int row, const int col) { return data[col][row]; }

  /**
   *  Print the contents of the CellDataT instance.
   */
  inline void print() {

    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        std::cout << this->format(this->data[colx][rowx]) << " ";
      }
      std::cout << std::endl;
    }
  }
};

template <typename T> using CellData = std::shared_ptr<CellDataT<T>>;

} // namespace NESO::Particles
#endif
