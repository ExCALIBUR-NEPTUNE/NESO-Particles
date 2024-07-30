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
#include <sstream>
#include <string>
#include <vector>

namespace NESO::Particles::VTK {

/**
 * Datatype for representing the data for a single cell.
 */
struct UnstructuredCell {
  /// Number of points for the cell.
  int num_points;
  /// Coordinates of the points. This vector should be 3 doubles per point
  /// linearised into a single vector.
  std::vector<double> points;
  /// Number of spatial dimensions for the space the cell is embedded in, e.g. 2
  /// for Triangles and Quadrilaterals. num_dimensions=2 and num_points=4 is a
  /// Quadrilateral, num_dimensions=3 and num_points=4 is a Tetrahedron.
  int num_dimensions;
  /// Map from a quantity name to a vector of size num_points containing the
  /// values for each point in the order the vertices are described in the
  /// points array.
  std::map<std::string, double> cell_data;
  /// Map from a quantity name to a single value for the cell.
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

    NESOASSERT(local_size * ncomp == data.size(), "data size miss-match");
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

  inline int get_cell_type(const int num_dimensions, const int num_points) {
    NESOASSERT((-1 < num_dimensions) && (num_dimensions < 4),
               "Bad number of dimensions specified.");
    if (num_dimensions < 2) {
      NESOASSERT((0 < num_points) && (num_points < 3),
                 "Bad number of points specified.");
      if (num_points == 1) {
        return 1; // point
      } else {
        return 3; // line
      }
    } else if (num_dimensions == 2) {
      NESOASSERT((2 < num_points) && (num_points < 5),
                 "Bad number of points specified.");
      if (num_points == 3) {
        return 5; // triangle
      } else {
        return 9; // quad
      }
    } else {
      NESOASSERT((3 < num_points) && (num_points < 9) && (num_points != 7),
                 "Bad number of points specified.");
      if (num_points == 4) {
        return 10; // tet
      } else if (num_points == 5) {
        return 14; // pyr
      } else if (num_points == 6) {
        return 13; // prism (vtk wedge)
      } else {
        return 12; // hex
      }
    }
  }

public:
  ~VTKHDF() {
    NESOASSERT(this->is_closed, "VTKHDF file was not closed correctly.");
  };

  /**
   * Close the HDF5 file. This must be called collectively on the communicator.
   */
  inline void close() {
    H5CHK(H5Gclose(this->root));
    H5CHK(H5Fclose(this->file_id));
    H5CHK(H5Pclose(this->plist_id));
    this->is_closed = true;
  };

  /**
   * Create a new VTKHDF file. Must be called collectively on the communicator.
   *
   * @param filename Output filename. This filename should end with the
   * extension .vtkhdf.
   * @param comm MPI communicator to use.
   * @param dataset_type The type of data the user will write to the file.
   */
  VTKHDF(std::string filename, MPI_Comm comm,
         std::string dataset_type = "UnstructuredGrid")
      : comm(comm), is_closed(true), step(0), dataset_type(dataset_type) {
    MPICHK(MPI_Comm_rank(this->comm, &this->rank));
    MPICHK(MPI_Comm_size(this->comm, &this->size));

    this->plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5CHK(H5Pset_fapl_mpio(this->plist_id, this->comm, MPI_INFO_NULL));
    H5CHK(this->file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC,
                                    H5P_DEFAULT, this->plist_id));
    this->is_closed = false;

    // Create the root group
    H5CHK(this->root = H5Gcreate(file_id, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT));

    // Create the global attributes
    // Version
    {
      hid_t property;
      H5CHK(property = H5Pcreate(H5P_ATTRIBUTE_CREATE));
      hid_t dataspace;
      hsize_t dims[1] = {2};
      H5CHK(dataspace = H5Screate_simple(1, dims, dims));
      hid_t version;
      H5CHK(version = H5Acreate(this->root, "Version", H5T_STD_I64LE, dataspace,
                                property, H5P_DEFAULT));
      int value[2] = {2, 0};
      H5CHK(H5Awrite(version, H5T_NATIVE_INT, &value));
      H5CHK(H5Aclose(version));
      H5CHK(H5Sclose(dataspace));
      H5CHK(H5Pclose(property));
    }
    // Type
    {
      hid_t property;
      H5CHK(property = H5Pcreate(H5P_ATTRIBUTE_CREATE));
      hid_t dataspace;
      H5CHK(dataspace = H5Screate(H5S_SCALAR));
      hid_t version;

      std::string value = dataset_type;
      hid_t string_type;
      H5CHK(string_type = H5Tcreate(H5T_STRING, value.size()));
      H5CHK(version = H5Acreate(this->root, "Type", string_type, dataspace,
                                property, H5P_DEFAULT));
      H5CHK(H5Awrite(version, string_type, value.c_str()));

      H5CHK(H5Tclose(string_type));
      H5CHK(H5Aclose(version));
      H5CHK(H5Sclose(dataspace));
      H5CHK(H5Pclose(property));
    }
  }

  /**
   * Write unstructured data to the file. Must be called collectively on the
   * communicator.
   *
   * @param data Data to write to the file.
   */
  inline void write(std::vector<UnstructuredCell> &data) {
    NESOASSERT(this->dataset_type == "UnstructuredGrid",
               "Attempting to write unstructured grid data to a file which is "
               "not set up for unstructured grid data.");

    int npoint_local = 0;
    int ncell_local = 0;
    std::vector<double> points;
    std::vector<int> types;
    types.reserve(data.size());
    std::vector<int> connectivity;
    std::vector<int> offsets;
    std::map<std::string, std::vector<double>> point_data;
    std::map<std::string, std::vector<double>> cell_data;

    int point_index = 0;
    offsets.push_back(point_index);
    for (const auto &ex : data) {
      const int num_points = ex.num_points;
      npoint_local += num_points;
      ncell_local++;
      NESOASSERT(ex.points.size() == num_points * 3,
                 "Incorrect number of coordinates, expected " +
                     std::to_string(num_points * 3) + " but found " +
                     std::to_string(ex.points.size()) + ".");

      points.insert(points.end(), ex.points.begin(), ex.points.end());

      for (auto &name_data : ex.point_data) {
        auto &n = name_data.first;
        auto &d = name_data.second;
        NESOASSERT(d.size() == num_points,
                   "Points data for " + n + " has wrong array length.");
        point_data[n].insert(point_data[n].end(), d.begin(), d.end());
      }

      for (int vx = 0; vx < num_points; vx++) {
        connectivity.push_back(point_index++);
      }
      const int cell_type = this->get_cell_type(ex.num_dimensions, num_points);
      types.push_back(cell_type);
      offsets.push_back(point_index);
      for (auto &name_data : ex.cell_data) {
        cell_data[name_data.first].push_back(name_data.second);
      }
    }

    int counts_local[4] = {npoint_local, ncell_local,
                           static_cast<int>(connectivity.size()),
                           ncell_local + 1};
    int offsets_array[4] = {0, 0, 0, 0};
    int counts_global[4];
    MPICHK(MPI_Exscan(counts_local, offsets_array, 4, MPI_INT, MPI_SUM,
                      this->comm));
    for (int ix = 0; ix < 4; ix++) {
      counts_global[ix] = offsets_array[ix] + counts_local[ix];
    }
    MPICHK(MPI_Bcast(counts_global, 4, MPI_INT, this->size - 1, this->comm));

    const int nconnectivity_local = counts_local[2];
    const int noffset_local = counts_local[3];

    const int point_offset = offsets_array[0];
    const int cell_offset = offsets_array[1];
    const int connectivity_offset = offsets_array[2];
    const int offset_offset = offsets_array[3];

    const int npoint_global = counts_global[0];
    const int ncell_global = counts_global[1];
    const int nconnectivity_global = counts_global[2];
    const int noffset_global = counts_global[3];

    NESOASSERT(offsets.size() == noffset_local, "offsets sizes miss-match");

    this->write_dataset_2d(this->root, npoint_global, point_offset,
                           npoint_local, 3, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE,
                           "Points", points);

    std::vector<int> number_of_points = {npoint_local};
    this->write_dataset(this->root, this->size, this->rank, H5T_STD_I64LE,
                        H5T_NATIVE_INT, "NumberOfPoints", number_of_points);

    std::vector<int> number_of_cells = {ncell_local};
    this->write_dataset(this->root, this->size, this->rank, H5T_STD_I64LE,
                        H5T_NATIVE_INT, "NumberOfCells", number_of_cells);

    this->write_dataset(this->root, ncell_global, cell_offset, H5T_STD_U8LE,
                        H5T_NATIVE_INT, "Types", types);

    this->write_dataset(this->root, nconnectivity_global, connectivity_offset,
                        H5T_STD_I64LE, H5T_NATIVE_INT, "Connectivity",
                        connectivity);

    std::vector<int> number_of_connectivity_ids = {
        static_cast<int>(connectivity.size())};
    this->write_dataset(this->root, this->size, this->rank, H5T_STD_I64LE,
                        H5T_NATIVE_INT, "NumberOfConnectivityIds",
                        number_of_connectivity_ids);

    this->write_dataset(this->root, noffset_global, offset_offset,
                        H5T_STD_I64LE, H5T_NATIVE_INT, "Offsets", offsets);

    hid_t point_data_group;
    H5CHK(point_data_group = H5Gcreate(this->root, "PointData", H5P_DEFAULT,
                                       H5P_DEFAULT, H5P_DEFAULT));
    hid_t cell_data_group;
    H5CHK(cell_data_group = H5Gcreate(this->root, "CellData", H5P_DEFAULT,
                                      H5P_DEFAULT, H5P_DEFAULT));

    for (auto &name_data : point_data) {
      this->write_dataset(point_data_group, npoint_global, point_offset,
                          H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, name_data.first,
                          name_data.second);
    }

    for (auto &name_data : cell_data) {
      this->write_dataset(cell_data_group, ncell_global, cell_offset,
                          H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, name_data.first,
                          name_data.second);
    }

    H5CHK(H5Gclose(point_data_group));
    H5CHK(H5Gclose(cell_data_group));
  }
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
  inline void close(){};

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
  inline void write([[maybe_unused]] std::vector<UnstructuredCell> &data) {}
};

#endif
} // namespace NESO::Particles::VTK

#endif
