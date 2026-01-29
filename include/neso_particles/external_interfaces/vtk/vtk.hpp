#ifndef _NESO_PARTICLES_EXTERNAL_VTK_VTK_HPP_
#define _NESO_PARTICLES_EXTERNAL_VTK_VTK_HPP_

#include "../../communication/communication_typedefs.hpp"
#include "../../typedefs.hpp"
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace NESO::Particles::VTK {

enum CellType {
  point = 1,
  line = 3,
  triangle = 5,
  quadrilateral = 9,
  tetrahedron = 10,
  pyramid = 14,
  wedge = 13,
  hex = 12
};

/**
 * Datatype for representing the data for a single cell.
 */
struct UnstructuredCell {
  /// Number of points for the cell.
  int num_points;
  /// Coordinates of the points. This vector should be 3 doubles per point
  /// linearised into a single vector.
  std::vector<double> points;
  /// Enum describing the shape type.
  CellType cell_type;
  /// Map from a quantity name to a single value for the cell.
  std::map<std::string, double> cell_data;
  /// Map from a quantity name to a vector of size num_points containing the
  /// values for each point in the order the vertices are described in the
  /// points array.
  std::map<std::string, std::vector<double>> point_data;
};

#ifdef NESO_PARTICLES_HDF5

/**
 * The VTKHDF class facilitates writing VTKHDF files by using HDF5.
 */
class VTKHDF {
protected:
  MPI_Comm comm;
  bool is_closed;
  int step, rank, size;
  hid_t plist_id, file_id, root;
  std::string dataset_type;

  template <typename T>
  inline void
  write_dataset_2d(hid_t group, const int global_size, const int offset,
                   const int local_size, const int ncomp, hid_t out_datatype,
                   hid_t datatype, std::string name, std::vector<T> &data) {

    NESOASSERT(static_cast<std::size_t>(local_size) * ncomp == data.size(),
               "data size miss-match");
    // Create the memspace
    hsize_t dims_memspace[2] = {static_cast<hsize_t>(local_size),
                                static_cast<hsize_t>(ncomp)};
    hid_t memspace;
    H5CHK(memspace = H5Screate_simple(2, dims_memspace, NULL));

    // Create the filespace
    hsize_t dims_filespace[2] = {static_cast<hsize_t>(global_size),
                                 static_cast<hsize_t>(ncomp)};
    hid_t filespace;
    H5CHK(filespace = H5Screate_simple(2, dims_filespace, NULL));

    // Select this ranks region of the filespace
    hsize_t slab_offsets[2] = {static_cast<hsize_t>(offset),
                               static_cast<hsize_t>(0)};
    hsize_t slab_counts[2] = {static_cast<hsize_t>(local_size),
                              static_cast<hsize_t>(ncomp)};
    H5CHK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, slab_offsets, NULL,
                              slab_counts, NULL));
    // Create the partition
    hid_t partition;
    H5CHK(partition = H5Pcreate(H5P_DATASET_XFER));
    H5CHK(H5Pset_dxpl_mpio(partition, H5FD_MPIO_COLLECTIVE));

    hid_t dset;
    H5CHK(dset = H5Dcreate2(group, name.c_str(), out_datatype, filespace,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    H5CHK(
        H5Dwrite(dset, datatype, memspace, filespace, partition, data.data()));

    H5CHK(H5Dclose(dset));
    H5CHK(H5Pclose(partition));
    H5CHK(H5Sclose(filespace));
    H5CHK(H5Sclose(memspace));
  }

  template <typename T>
  inline void write_dataset(hid_t group, const int global_size,
                            const int offset, hid_t out_datatype,
                            hid_t datatype, std::string name,
                            std::vector<T> &data) {
    const int local_size = data.size();
    // Create the memspace
    hsize_t dims_memspace[1] = {static_cast<hsize_t>(local_size)};
    hid_t memspace;
    H5CHK(memspace = H5Screate_simple(1, dims_memspace, NULL));

    // Create the filespace
    hsize_t dims_filespace[1] = {static_cast<hsize_t>(global_size)};
    hid_t filespace;
    H5CHK(filespace = H5Screate_simple(1, dims_filespace, NULL));

    // Select this ranks region of the filespace
    hsize_t slab_offsets[1] = {static_cast<hsize_t>(offset)};
    hsize_t slab_counts[1] = {static_cast<hsize_t>(local_size)};
    H5CHK(H5Sselect_hyperslab(filespace, H5S_SELECT_SET, slab_offsets, NULL,
                              slab_counts, NULL));
    // Create the partition
    hid_t partition;
    H5CHK(partition = H5Pcreate(H5P_DATASET_XFER));
    H5CHK(H5Pset_dxpl_mpio(partition, H5FD_MPIO_COLLECTIVE));

    hid_t dset;
    H5CHK(dset = H5Dcreate2(group, name.c_str(), out_datatype, filespace,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    H5CHK(
        H5Dwrite(dset, datatype, memspace, filespace, partition, data.data()));

    H5CHK(H5Dclose(dset));
    H5CHK(H5Pclose(partition));
    H5CHK(H5Sclose(filespace));
    H5CHK(H5Sclose(memspace));
  }

public:
  ~VTKHDF() {
    if ((!this->is_closed) && (!this->rank)) {
      nprint("VTKHDF file was not closed correctly.");
    }
  };

  /**
   * Close the HDF5 file. This must be called collectively on the communicator.
   */
  void close();

  /**
   * Create a new VTKHDF file. Must be called collectively on the communicator.
   *
   * @param filename Output filename. This filename should end with the
   * extension .vtkhdf.
   * @param comm MPI communicator to use.
   * @param dataset_type The type of data the user will write to the file.
   */
  VTKHDF(std::string filename, MPI_Comm comm,
         std::string dataset_type = "UnstructuredGrid");

  /**
   * Write unstructured data to the file. Must be called collectively on the
   * communicator.
   *
   * @param data Data to write to the file.
   * @param point_data_keys Pass the point data keys for the UnstructuredCell
   * point data. This is optional if all ranks have data as all the data must
   * have the same keys. If some ranks may have no data then this must be set to
   * the keys that the ranks that do have data have as keys.
   * @param cell_data_keys Pass the point data keys for the UnstructuredCell
   * cell data. This is optional if all ranks have data as all the data must
   * have the same keys. If some ranks may have no data then this must be set to
   * the keys that the ranks that do have data have as keys.
   */
  void write(std::vector<UnstructuredCell> &data,
             std::set<std::string> point_data_keys = {},
             std::set<std::string> cell_data_keys = {});
};

#else

/**
 * The VTKHDF class facilitates writing VTKHDF files by using HDF5.
 */
class VTKHDF {
protected:
public:
  ~VTKHDF(){};

  /**
   * Close the HDF5 file. This must be called collectively on the communicator.
   */
  inline void close() {};

  /**
   * Create a new VTKHDF file. Must be called collectively on the communicator.
   *
   * @param filename Output filename. This filename should end with the
   * extension .vtkhdf.
   * @param comm MPI communicator to use.
   * @param dataset_type The type of data the user will write to the file.
   */
  VTKHDF([[maybe_unused]] std::string filename, [[maybe_unused]] MPI_Comm comm,
         [[maybe_unused]] std::string dataset_type = "UnstructuredGrid") {}

  /**
   * Write unstructured data to the file. Must be called collectively on the
   * communicator.
   *
   * @param data Data to write to the file.
   */
  inline void
  write([[maybe_unused]] std::vector<UnstructuredCell> &data,
        [[maybe_unused]] std::set<std::string> point_data_keys = {},
        [[maybe_unused]] std::set<std::string> cell_data_keys = {}) {}
};

#endif
} // namespace NESO::Particles::VTK

#endif
