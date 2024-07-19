#ifndef _NESO_PARTICLES_EXTERNAL_VTK_VTK_HPP_
#define _NESO_PARTICLES_EXTERNAL_VTK_VTK_HPP_

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

struct UnstructuredCell {
  std::vector<REAL> points;
  int cell_type;
  std::vector<REAL> point_data;
};

inline void write_legacy_vtk(std::string filename_base,
                             std::vector<UnstructuredCell> &&data,
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
  std::vector<REAL> points;
  std::vector<REAL> point_data;
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

} // namespace NESO::Particles::VTK

#endif
