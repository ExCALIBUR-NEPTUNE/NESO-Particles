#ifdef NESO_PARTICLES_HDF5

#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/vtk/vtk.hpp>

TEST(VTK, vtkhdf) {

  int rank = 0;
  int size = 0;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  std::string filename = get_test_root_file("test_all.vtkhdf");
  VTK::VTKHDF v(filename, MPI_COMM_WORLD);

  const REAL z_base = rank * 2.0;

  const int num_cells = 4;
  std::vector<VTK::UnstructuredCell> vtk_data(num_cells);

  {
    const VTK::CellType t = VTK::CellType::point;
    const int index = 0;
    const int num_vertices = get_num_vertices(t);
    vtk_data[index].num_points = num_vertices;
    vtk_data[index].points.push_back(0.0);
    vtk_data[index].points.push_back(0.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].cell_type = t;
    vtk_data[index].cell_data["u"] = rank;
    for (int vx = 0; vx < num_vertices; vx++) {
      vtk_data[index].point_data["v"].push_back(rank);
    }
  }

  {
    const VTK::CellType t = VTK::CellType::line;
    const int index = 1;
    const int num_vertices = get_num_vertices(t);
    vtk_data[index].num_points = num_vertices;
    vtk_data[index].points.push_back(1.0);
    vtk_data[index].points.push_back(0.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].points.push_back(1.0);
    vtk_data[index].points.push_back(1.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].cell_type = t;
    vtk_data[index].cell_data["u"] = rank;
    for (int vx = 0; vx < num_vertices; vx++) {
      vtk_data[index].point_data["v"].push_back(rank +
                                                (1.0 / num_vertices) * vx);
    }
  }

  {
    const VTK::CellType t = VTK::CellType::triangle;
    const int index = 2;
    const int num_vertices = get_num_vertices(t);
    vtk_data[index].num_points = num_vertices;
    vtk_data[index].points.push_back(2.0);
    vtk_data[index].points.push_back(0.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].points.push_back(3.0);
    vtk_data[index].points.push_back(0.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].points.push_back(3.0);
    vtk_data[index].points.push_back(2.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].cell_type = t;
    vtk_data[index].cell_data["u"] = rank;
    for (int vx = 0; vx < num_vertices; vx++) {
      vtk_data[index].point_data["v"].push_back(rank +
                                                (1.0 / num_vertices) * vx);
    }
  }

  {
    const VTK::CellType t = VTK::CellType::quadrilateral;
    const int index = 3;
    const int num_vertices = get_num_vertices(t);
    vtk_data[index].num_points = num_vertices;
    vtk_data[index].points.push_back(4.0);
    vtk_data[index].points.push_back(0.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].points.push_back(5.0);
    vtk_data[index].points.push_back(0.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].points.push_back(5.0);
    vtk_data[index].points.push_back(1.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].points.push_back(4.0);
    vtk_data[index].points.push_back(1.0);
    vtk_data[index].points.push_back(z_base);
    vtk_data[index].cell_type = t;
    vtk_data[index].cell_data["u"] = rank;
    for (int vx = 0; vx < num_vertices; vx++) {
      vtk_data[index].point_data["v"].push_back(rank +
                                                (1.0 / num_vertices) * vx);
    }
  }

  v.write(vtk_data);
  v.close();

  VTK::VTKHDF u(filename, MPI_COMM_WORLD);

  std::vector<VTK::UnstructuredCell> vtk_data_read;
  u.read(num_cells, vtk_data_read, {"v"}, {"u"});
  u.close();

  ASSERT_EQ(vtk_data_read.size(), num_cells);
  for (int cellx = 0; cellx < num_cells; cellx++) {
    ASSERT_EQ(vtk_data_read.at(cellx).cell_type, vtk_data.at(cellx).cell_type);
    ASSERT_EQ(vtk_data_read.at(cellx).points, vtk_data.at(cellx).points);
    ASSERT_EQ(vtk_data_read.at(cellx).point_data,
              vtk_data.at(cellx).point_data);
    ASSERT_EQ(vtk_data_read.at(cellx).cell_data, vtk_data.at(cellx).cell_data);
  }
}

#endif
