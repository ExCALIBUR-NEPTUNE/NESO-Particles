#ifndef _NESO_PARTICLES_EXTERNAL_VTK_VTK_HPP_
#define _NESO_PARTICLES_EXTERNAL_VTK_VTK_HPP_

#include "../../communication/communication_typedefs.hpp"
#include "../../typedefs.hpp"
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace NESO::Particles::VTK {

/**
 * TODO
 */
struct UnstructuredCell {
  std::vector<double> points;
  int cell_type;
  std::vector<double> point_data;
};

/**
 * TODO REMOVE?
 */
inline void write_legacy_vtk(std::string filename_base,
                             std::vector<UnstructuredCell> data,
                             const int rank = -1, const int timestep = -1) {

  std::string filename = filename_base;
  if (rank > -1) {
    filename += "." + std::to_string(rank);
  }
  if (timestep > -1) {
    filename += "." + std::to_string(timestep);
  }
  filename += ".vtk";

  std::ofstream vtk_file;
  vtk_file.open(filename);
  vtk_file << "# vtk DataFile Version 2.0" << std::endl;
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  vtk_file << "NESO-Particles output from: "
           << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << std::endl;
  vtk_file << "ASCII" << std::endl;
  vtk_file << "DATASET UNSTRUCTURED_GRID" << std::endl;
  vtk_file << std::endl;

  int num_points = 0;
  std::vector<double> points;
  std::vector<double> point_data;
  int num_cells = 0;
  std::vector<int> cells;
  std::vector<int> cell_types;

  for (const auto &ex : data) {
    NESOASSERT(ex.points.size() == ex.point_data.size() * 3,
               "Unexpected data lengths.");
    const int ex_num_points = ex.point_data.size();
    cell_types.push_back(ex.cell_type);
    cells.push_back(ex_num_points);
    for (int pointx = 0; pointx < ex_num_points; pointx++) {
      const int point_index = num_points++;
      cells.push_back(point_index);
    }
    num_cells++;
    points.insert(points.end(), ex.points.begin(), ex.points.end());
    point_data.insert(point_data.end(), ex.point_data.begin(),
                      ex.point_data.end());
  }

  vtk_file << "POINTS " << point_data.size() << " float\n";
  for (auto &px : points) {
    vtk_file << std::to_string(px) << " ";
  }
  vtk_file << std::endl;
  vtk_file << std::endl;

  vtk_file << "CELLS " << num_cells << " " << cells.size() << std::endl;
  for (auto &px : cells) {
    vtk_file << std::to_string(px) << " ";
  }
  vtk_file << std::endl;
  vtk_file << std::endl;

  vtk_file << "CELL_TYPES " << num_cells << std::endl;
  for (auto &px : cell_types) {
    vtk_file << std::to_string(px) << " ";
  }
  vtk_file << std::endl;
  vtk_file << std::endl;

  vtk_file << "POINT_DATA " << point_data.size() << std::endl;
  vtk_file << "SCALARS scalars float 1" << std::endl;
  vtk_file << "LOOKUP_TABLE default" << std::endl;
  for (auto &px : point_data) {
    vtk_file << std::to_string(px) << " ";
  }
  vtk_file << std::endl;
  vtk_file << std::endl;

  vtk_file.close();
}

/**
 * TODO
 *
 * TODO CREATE MASKED OFF VERSION FOR NO HDF5
 */
class VTKHDF {
protected:
  MPI_Comm comm;
  bool is_closed;
  int step, rank, size;
  hid_t plist_id, file_id, root;

  template <typename T>
  inline void
  write_dataset_2d(hid_t group, const int global_size, const int offset,
                   const int local_size, const int ncomp, hid_t out_datatype,
                   hid_t datatype, std::string name, std::vector<T> &data) {
    NESOASSERT(local_size * ncomp == data.size(), "data size missmatch");
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
    NESOASSERT(this->is_closed, "VTKHDF file was not closed correctly.");
  };

  /**
   * TODO
   */
  inline void close() {
    H5CHK(H5Gclose(this->root));
    H5CHK(H5Fclose(this->file_id));
    H5CHK(H5Pclose(this->plist_id));
    this->is_closed = true;
  };

  /**
   * TODO
   */
  VTKHDF(std::string filename, MPI_Comm comm,
         std::string dataset_type = "UnstructuredGrid")
      : comm(comm), is_closed(true), step(0) {

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
   * TODO
   */
  inline void write(std::vector<UnstructuredCell> &data) {

    int npoint_local = 0;
    int ncell_local = 0;
    std::vector<double> points;
    std::vector<int> types;
    types.reserve(data.size());
    std::vector<int> connectivity;
    std::vector<int> offsets;
    std::vector<double> point_data;
    std::vector<double> cell_data;

    int point_index = 0;
    offsets.push_back(point_index);
    for (const auto &ex : data) {
      npoint_local += ex.point_data.size();
      ncell_local++;
      points.insert(points.end(), ex.points.begin(), ex.points.end());
      point_data.insert(point_data.end(), ex.point_data.begin(),
                        ex.point_data.end());
      const int num_vertices = ex.point_data.size();
      for (int vx = 0; vx < num_vertices; vx++) {
        connectivity.push_back(point_index++);
      }
      types.push_back(ex.cell_type);
      offsets.push_back(point_index);
      cell_data.push_back(1.0);
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

    NESOASSERT(offsets.size() == noffset_local, "offsets sizses missmatch");

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

    this->write_dataset(point_data_group, npoint_global, point_offset,
                        H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, "default",
                        point_data);

    this->write_dataset(cell_data_group, ncell_global, cell_offset,
                        H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE, "default",
                        cell_data);

    H5CHK(H5Gclose(point_data_group));
    H5CHK(H5Gclose(cell_data_group));
  }
};

} // namespace NESO::Particles::VTK

#endif
