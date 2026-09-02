#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>
#include <string>

using namespace NESO::Particles;

TEST(PETSc, dmplex_interface_3d_base) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mixed_ref_cube_0.8.msh");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  int rank = -1;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  auto mesh_helper =
      std::make_shared<PetscInterface::DMPlexHelper>(MPI_COMM_WORLD, dm);

  std::vector<PetscScalar> point{0.1, 0.1, 0.1};
  mesh_helper->cell_contains_point_3d(0, point);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  ASSERT_TRUE(mesh->validate_halos(false));

  const double volume = mesh->dmh->get_volume();
  ASSERT_NEAR(volume, 8.0, 1.0e-10);

  mesh->free();
  mesh_helper->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSc, dmplex_3d_mapper) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mixed_ref_cube_0.8.msh");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  const int ndim = 3;
  int rank = -1;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  std::mt19937 rng_pos(52234234 + rank);

  REAL extents[3] = {2.0, 2.0, 2.0};

  const int cell_count = mesh->get_cell_count();
  const int N = 200 * cell_count;
  auto positions = uniform_within_extents(N, ndim, extents, rng_pos);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions[dimx][px] - 1.0;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }
  A->add_particles_local(initial_distribution);

  A->hybrid_move();
  A->cell_move();

  std::vector<PetscScalar> point(3);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto CELL_ID = A->get_cell(Sym<INT>("CELL_ID"), cellx);
    auto P = A->get_cell(Sym<REAL>("P"), cellx);
    const int nrow = CELL_ID->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      point[0] = P->at(rowx, 0);
      point[1] = P->at(rowx, 1);
      point[2] = P->at(rowx, 2);

      const bool host_bool = mesh->dmh->cell_contains_point(cellx, point);
      ASSERT_TRUE(host_bool);
    }
  }

  sycl_target->free();
  mesh->free();

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

namespace {

struct BoundaryInteraction3DTest : PetscInterface::BoundaryInteraction3D {

  template <typename... ARGS>
  BoundaryInteraction3DTest(ARGS... args) : BoundaryInteraction3D(args...) {}

  MAKE_GETTER_METHOD(required_mh_cells)
  MAKE_GETTER_METHOD(collected_mh_cells)
  MAKE_GETTER_METHOD(padding)
  MAKE_GETTER_METHOD(d_map_facet_discovery)
  MAKE_GETTER_METHOD(map_label_to_groups)
  MAKE_GETTER_METHOD(d_map_facet_normals)
  MAKE_WRAP_METHOD(collect_cells)
  MAKE_WRAP_METHOD(get_labels)
};

struct BoundaryTriangleTest {
  REAL vertices[3][3];
  REAL normal[3];
  int face_id;
  int group_id;
  int type;
};

std::vector<REAL> get_coords(DM dm, PetscInt petsc_index) {
  const PetscScalar *tmp;
  PetscScalar *vertices = nullptr;
  PetscInt num_coords;
  PetscBool is_dg;

  PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords, &tmp,
                                    &vertices));
  const int num_vertices = num_coords / 3;
  NESOASSERT(num_vertices == 1, "Expected a point.");
  std::vector<PetscScalar> h_vertices;
  h_vertices.reserve(num_coords);
  for (PetscInt ix = 0; ix < num_coords; ix++) {
    h_vertices.push_back(vertices[ix]);
  }
  PETSCCHK(DMPlexRestoreCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                        &tmp, &vertices));

  return h_vertices;
}

bool normal_points_towards_point(DM dm, const PetscInt p0, const PetscInt p1,
                                 const PetscInt p2, const PetscInt point) {

  auto v0 = get_coords(dm, p0);
  auto v1 = get_coords(dm, p1);
  auto v2 = get_coords(dm, p2);
  auto d = get_coords(dm, point);

  std::vector<REAL> n01(3);
  std::vector<REAL> n02(3);
  std::vector<REAL> vd(3);
  for (int dx = 0; dx < 3; dx++) {
    n01.at(dx) = v1.at(dx) - v0.at(dx);
    n02.at(dx) = v2.at(dx) - v0.at(dx);
    vd.at(dx) = d.at(dx) - v0.at(dx);
  }
  std::vector<REAL> n = {0.0, 0.0, 0.0};

  KERNEL_CROSS_PRODUCT_3D(n01[0], n01[1], n01[2], n02[0], n02[1], n02[2], n[0],
                          n[1], n[2]);
  const REAL dd = KERNEL_DOT_PRODUCT_3D(n[0], n[1], n[2], vd[0], vd[1], vd[2]);

  return dd >= 0.0;
}

DMPolytopeType get_point_type(DM dm, const PetscInt point_index) {
  DMPolytopeType cell_type;
  PETSCCHK(DMPlexGetCellType(dm, point_index, &cell_type));
  return cell_type;
}

void get_vertex_neighbours(DM dm, const PetscInt point_index,
                           std::vector<PetscInt> &neighbours) {

  neighbours.clear();
  PetscInt support_size = 0;
  PETSCCHK(DMPlexGetSupportSize(dm, point_index, &support_size));
  const PetscInt *support = nullptr;
  PETSCCHK(DMPlexGetSupport(dm, point_index, &support));

  for (PetscInt sx = 0; sx < support_size; sx++) {
    PetscInt cone_size = 0;
    const PetscInt support_point = support[sx];
    PETSCCHK(DMPlexGetConeSize(dm, support_point, &cone_size));
    NESOASSERT(cone_size == 2, "Expected support point to be an edge.");
    const PetscInt *support_cone = nullptr;
    PETSCCHK(DMPlexGetCone(dm, support_point, &support_cone));
    const PetscInt p0 = support_cone[0];
    const PetscInt p1 = support_cone[1];

    if (p0 == point_index) {
      neighbours.push_back(p1);
    } else {
      neighbours.push_back(p0);
    }
  }
}

std::vector<PetscInt> get_canonical_vertex_order(DM dm, const PetscInt point) {
  std::vector<PetscInt> order;

  PetscInt depth = -1;
  PETSCCHK(DMPlexGetPointDepth(dm, point, &depth));
  const PetscInt *cone = nullptr;
  PetscInt cone_size = 0;
  PETSCCHK(DMPlexGetConeSize(dm, point, &cone_size));
  if (cone_size > 0) {
    PETSCCHK(DMPlexGetCone(dm, point, &cone));
  }

  auto point_type = get_point_type(dm, point);

  std::vector<std::vector<PetscInt>> faces;
  std::set<PetscInt> vertex_points;
  for (PetscInt fx = 0; fx < cone_size; fx++) {
    auto t = get_canonical_vertex_order(dm, cone[fx]);
    for (auto tx : t) {
      vertex_points.insert(tx);
    }
    faces.push_back(t);
  }

  if (point_type == DM_POLYTOPE_POINT) {
    order.push_back(point);
  } else if (point_type == DM_POLYTOPE_SEGMENT) {
    order.push_back(cone[0]);
    order.push_back(cone[1]);
  } else if (point_type == DM_POLYTOPE_POINT_PRISM_TENSOR) {
    order.push_back(cone[0]);
    order.push_back(cone[1]);
  } else if ((point_type == DM_POLYTOPE_TRIANGLE) ||
             (point_type == DM_POLYTOPE_QUADRILATERAL) ||
             (point_type == DM_POLYTOPE_SEG_PRISM_TENSOR)) {

    std::map<PetscInt, std::set<PetscInt>> map_vertex_to_neighbours;

    PetscInt first_vertex = -1;
    for (PetscInt edgex = 0; edgex < cone_size; edgex++) {
      const PetscInt edge = cone[edgex];
      std::vector<PetscInt> edge_cone = get_canonical_vertex_order(dm, edge);

      const PetscInt v0 = edge_cone.at(0);
      const PetscInt v1 = edge_cone.at(1);
      if (edgex == 0) {
        first_vertex = v0;
      }

      map_vertex_to_neighbours[v0].insert(v1);
      map_vertex_to_neighbours[v1].insert(v0);
    }

    PetscInt current_vertex = first_vertex;
    for (int edgex = 0; edgex < cone_size; edgex++) {

      order.push_back(current_vertex);
      // get a neighbour vertex
      const PetscInt next_vertex =
          *map_vertex_to_neighbours.at(current_vertex).begin();
      // Remove the current point from the neighbours of the next point such
      // that the loop never travels backwards.
      map_vertex_to_neighbours.at(next_vertex).erase(current_vertex);

      current_vertex = next_vertex;
    }

    if (point_type == DM_POLYTOPE_SEG_PRISM_TENSOR) {
      const PetscInt t2 = order.at(2);
      const PetscInt t3 = order.at(3);
      order.at(2) = t3;
      order.at(3) = t2;
    }

  } else if (point_type == DM_POLYTOPE_TETRAHEDRON) {

    auto bottom_face = faces.at(0);
    for (auto tx : bottom_face) {
      vertex_points.erase(tx);
    }
    NESOASSERT(vertex_points.size() == 1, "Expected one remaining point.");
    const PetscInt point3 = *vertex_points.begin();

    const bool correct_order = normal_points_towards_point(
        dm, bottom_face.at(0), bottom_face.at(2), bottom_face.at(1), point3);

    if (correct_order) {
      order.push_back(bottom_face.at(0));
      order.push_back(bottom_face.at(1));
      order.push_back(bottom_face.at(2));
      order.push_back(point3);
    } else {
      order.push_back(bottom_face.at(2));
      order.push_back(bottom_face.at(1));
      order.push_back(bottom_face.at(0));
      order.push_back(point3);
    }
  } else if (point_type == DM_POLYTOPE_PYRAMID) {
    // There is one quad and this is the base.

    std::vector<PetscInt> bottom_face;
    for (auto &fx : faces) {
      if (fx.size() == 4) {
        bottom_face = fx;
      }
    }
    NESOASSERT(bottom_face.size() == 4, "Failed to find Pyramid base.");
    for (auto tx : bottom_face) {
      vertex_points.erase(tx);
    }
    NESOASSERT(vertex_points.size() == 1, "Expected one remaining point.");
    const PetscInt point4 = *vertex_points.begin();

    const bool correct_order = normal_points_towards_point(
        dm, bottom_face.at(0), bottom_face.at(3), bottom_face.at(1), point4);

    if (correct_order) {
      order.push_back(bottom_face.at(0));
      order.push_back(bottom_face.at(1));
      order.push_back(bottom_face.at(2));
      order.push_back(bottom_face.at(3));
      order.push_back(point4);
    } else {
      order.push_back(bottom_face.at(3));
      order.push_back(bottom_face.at(2));
      order.push_back(bottom_face.at(1));
      order.push_back(bottom_face.at(0));
      order.push_back(point4);
    }
  } else if ((point_type == DM_POLYTOPE_TRI_PRISM) ||
             (point_type == DM_POLYTOPE_TRI_PRISM_TENSOR)) {

    std::vector<PetscInt> top_face;
    std::vector<PetscInt> bottom_face;
    for (auto &fx : faces) {
      // Only consider the triangles.
      if (fx.size() == 3) {
        if (top_face.size() == 0) {
          top_face = fx;
        } else if (bottom_face.size() == 0) {
          bottom_face = fx;
        }
      } else {
        NESOASSERT(fx.size() == 4, "Remaining faces should be quads.");
      }
    }
    NESOASSERT(top_face.size() == 3, "Failed to find top face.");
    NESOASSERT(bottom_face.size() == 3, "Failed to find bottom face.");

    const bool bottom_points_inwards =
        normal_points_towards_point(dm, bottom_face.at(0), bottom_face.at(2),
                                    bottom_face.at(1), top_face.at(0));

    // tri prism bottom face is clockwise for prism and anticlockwise for tensor
    // prism.
    if ((!bottom_points_inwards) && (point_type == DM_POLYTOPE_TRI_PRISM)) {
      std::reverse(bottom_face.begin(), bottom_face.end());
    }
    if ((bottom_points_inwards) &&
        (point_type == DM_POLYTOPE_TRI_PRISM_TENSOR)) {
      std::reverse(bottom_face.begin(), bottom_face.end());
    }

    const bool top_normal_upwards = !normal_points_towards_point(
        dm, top_face.at(0), top_face.at(1), top_face.at(2), bottom_face.at(0));
    // tri prism and the tensor version have the same top face ordering.
    if ((!top_normal_upwards)) {
      std::reverse(top_face.begin(), top_face.end());
    }

    const PetscInt p0 = bottom_face.at(0);
    NESOASSERT(get_point_type(dm, p0) == DM_POLYTOPE_POINT,
               "Expected p0 to be a point.");
    std::vector<PetscInt> neighbours;
    get_vertex_neighbours(dm, p0, neighbours);

    PetscInt p3;
    for (auto &nx : neighbours) {
      if (std::find(top_face.begin(), top_face.end(), nx) != top_face.end()) {
        p3 = nx;
        break;
      }
    }

    auto iterator_start_top = std::find(top_face.begin(), top_face.end(), p3);
    const std::size_t index_start_top = iterator_start_top - top_face.begin();
    NESOASSERT(index_start_top < 3, "Failed to find starting top index");

    const PetscInt p4 = top_face.at((index_start_top + 1) % 3);
    const PetscInt p5 = top_face.at((index_start_top + 2) % 3);

    PetscInt to_test;
    get_vertex_neighbours(dm, bottom_face.at(1), neighbours);
    for (auto nx : neighbours) {
      if (std::find(top_face.begin(), top_face.end(), nx) != top_face.end()) {
        to_test = nx;
        break;
      }
    }

    const PetscInt correct = (point_type == DM_POLYTOPE_TRI_PRISM) ? p5 : p4;
    NESOASSERT(to_test == correct,
               "Failed to find consistent loop for top and bottom faces.");

    order.push_back(bottom_face.at(0));
    order.push_back(bottom_face.at(1));
    order.push_back(bottom_face.at(2));
    order.push_back(p3);
    order.push_back(p4);
    order.push_back(p5);
  } else if ((point_type == DM_POLYTOPE_HEXAHEDRON) ||
             (point_type == DM_POLYTOPE_QUAD_PRISM_TENSOR)) {

    auto bottom_face = faces.at(0);
    std::set<PetscInt> bottom_face_set;
    for (auto &fx : bottom_face) {
      bottom_face_set.insert(fx);
    }

    std::vector<PetscInt> top_face;

    for (auto &fx : faces) {
      NESOASSERT(fx.size() == 4, "Expected all faces to be quads.");

      bool top_face_candidate = true;
      for (auto px : fx) {
        if (bottom_face_set.count(px)) {
          top_face_candidate = false;
          break;
        }
      }
      if (top_face_candidate) {
        top_face = fx;
        break;
      }
    }

    NESOASSERT(top_face.size() == 4, "Failed to find a top face.");
    for (auto &px : top_face) {
      NESOASSERT(bottom_face_set.count(px) == 0,
                 "Top face candidate has a point from the bottom face.");
    }

    const bool bottom_points_inwards =
        normal_points_towards_point(dm, bottom_face.at(0), bottom_face.at(1),
                                    bottom_face.at(3), top_face.at(0));

    if (bottom_points_inwards && (point_type == DM_POLYTOPE_HEXAHEDRON)) {
      std::reverse(bottom_face.begin(), bottom_face.end());
    }
    if (!bottom_points_inwards &&
        (point_type == DM_POLYTOPE_QUAD_PRISM_TENSOR)) {
      std::reverse(bottom_face.begin(), bottom_face.end());
    }

    const bool top_points_inwards = normal_points_towards_point(
        dm, top_face.at(0), top_face.at(1), top_face.at(3), bottom_face.at(0));

    if (top_points_inwards) {
      std::reverse(top_face.begin(), top_face.end());
    }

    if (point_type == DM_POLYTOPE_HEXAHEDRON) {
      const bool bottom0 =
          normal_points_towards_point(dm, bottom_face.at(0), bottom_face.at(3),
                                      bottom_face.at(1), top_face.at(0));
      NESOASSERT(bottom0, "Bottom normal check failed.");
    }

    if (point_type == DM_POLYTOPE_QUAD_PRISM_TENSOR) {
      const bool bottom0 =
          normal_points_towards_point(dm, bottom_face.at(0), bottom_face.at(1),
                                      bottom_face.at(3), top_face.at(0));
      NESOASSERT(bottom0, "Bottom normal check failed.");
    }

    const bool top0 = !normal_points_towards_point(
        dm, top_face.at(0), top_face.at(1), top_face.at(3), bottom_face.at(0));

    NESOASSERT(top0, "Top normal check failed.");

    const PetscInt p0 = bottom_face.at(0);
    std::vector<PetscInt> neighbours;
    get_vertex_neighbours(dm, p0, neighbours);
    PetscInt p4;
    bool p4_found = false;
    for (auto &nx : neighbours) {
      if (std::find(top_face.begin(), top_face.end(), nx) != top_face.end()) {
        NESOASSERT(!p4_found, "p4 was already found.");
        p4_found = true;
        p4 = nx;
      }
    }

    auto iterator_start_top = std::find(top_face.begin(), top_face.end(), p4);
    const std::size_t index_start_top = iterator_start_top - top_face.begin();
    NESOASSERT(index_start_top < 4, "Failed to find starting top index");
    NESOASSERT(top_face.at(index_start_top) == p4,
               "p4 consistency check failed.");
    const PetscInt p5 = top_face.at((index_start_top + 1) % 4);
    const PetscInt p6 = top_face.at((index_start_top + 2) % 4);
    const PetscInt p7 = top_face.at((index_start_top + 3) % 4);

    PetscInt to_test;
    get_vertex_neighbours(dm, bottom_face.at(1), neighbours);
    for (auto nx : neighbours) {
      if (std::find(top_face.begin(), top_face.end(), nx) != top_face.end()) {
        to_test = nx;
        break;
      }
    }

    const PetscInt correct = (point_type == DM_POLYTOPE_HEXAHEDRON) ? p7 : p5;
    NESOASSERT(to_test == correct,
               "Failed to find consistent loop for top and bottom faces.");

    order.push_back(bottom_face.at(0));
    order.push_back(bottom_face.at(1));
    order.push_back(bottom_face.at(2));
    order.push_back(bottom_face.at(3));
    order.push_back(p4);
    order.push_back(p5);
    order.push_back(p6);
    order.push_back(p7);

  } else {
    NESOASSERT(false, "Unknown point type.");
  }

  return order;
}

} // namespace

TEST(PETScBoundary3D, foo) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mixed_ref_cube_0.8.msh");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  int rank = -1;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  PetscInt point_start = 0;
  PetscInt point_end = 0;
  PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));

  for (PetscInt px = point_start; px < point_end; px++) {

    auto o = get_canonical_vertex_order(dm, px);
    auto point_type = get_point_type(dm, px);

    if (point_type == DM_POLYTOPE_PYRAMID) {
      nprint("point:", px);
      for (auto &vx : o) {
        nprint("\t", vx);
      }
    }
  }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary3D, setup) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mixed_ref_cube_0.8.msh");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  int rank = -1;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  auto mesh_hierarchy = mesh->get_mesh_hierarchy();

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[0] = {100, 200, 300, 400, 500, 600};

  auto boundary_interaction = std::make_shared<BoundaryInteraction3DTest>(
      sycl_target, mesh, boundary_groups, 1.0e-14);

  auto &required_mh_cells = boundary_interaction->get_required_mh_cells();

  auto labels = boundary_interaction->wrap_get_labels();
  auto &map_label_to_groups = boundary_interaction->get_map_label_to_groups();

  // map from label to petsc point indices in the dm for the facets
  auto face_sets = mesh->dmh->get_face_sets();

  std::deque<std::pair<INT, double>> cells;
  auto padding = boundary_interaction->get_padding();

  std::vector<std::vector<REAL>> coords;
  std::vector<REAL> normal_vector;
  int triangle_index = 0;

  std::vector<BoundaryTriangleTest> h_triangles;
  std::vector<int> h_map_to_test;

  for (auto &item : face_sets) {
    if (labels.count(item.first)) {

      nprint("============================================================");
      std::vector<BoundaryTriangleTest> h_triangles_edge;

      {
        std::vector<VTK::UnstructuredCell> vtk_data;

        for (auto &point_id : item.second) {
          auto d = mesh->dmh->get_vtk_point_data(point_id);
          d.cell_data["u"] = point_id;
          vtk_data.push_back(d);

          if (point_id == 1848) {
            nprint("START");
            auto order = get_canonical_vertex_order(dm, point_id);
            nprint_variable(order);

            for (auto px : order) {
              std::vector<std::vector<REAL>> vertices;
              mesh->dmh->get_point_vertices(px, vertices);
              nprint(vertices.at(0));
            }

            nprint("END");
          }
        }

        VTK::VTKHDF w("foo_" + std::to_string(item.first) + ".vtkhdf",
                      mesh->get_comm());
        w.write(vtk_data);
        w.close();
      }

      nprint("------------------------------------------------------------");

      for (auto &point_id : item.second) {
        auto label_id = item.first;
        // If the facet is a quad then we will split that quad into two
        // triangles.
        const auto cell_type = mesh->dmh->get_point_type(point_id);
        const bool is_triangle = cell_type == DM_POLYTOPE_TRIANGLE;

        mesh->dmh->get_linear_normal_vector(point_id, normal_vector);
        auto bounding_box = mesh->dmh->get_point_bounding_box(point_id);
        bounding_box->expand({padding, padding, padding});

        const PetscInt facet_global_id =
            mesh->dmh->get_point_global_index(point_id);
        const int group_id = map_label_to_groups.at(label_id);

        cells.clear();
        ExternalCommon::bounding_box_map(bounding_box, mesh_hierarchy, cells);

        for (auto &cell_weight : cells) {
          required_mh_cells.insert(cell_weight.first);
        }

        auto lambda_push_triangle = [&](auto &t) {
          h_triangles.push_back(t);
          h_triangles_edge.push_back(t);
          for (auto &cell_weight : cells) {
            h_map_to_test.push_back(cell_weight.first);
            h_map_to_test.push_back(triangle_index);
          }
        };

        if (is_triangle) {
          mesh->dmh->get_point_vertices(point_id, coords);
          ASSERT_EQ(coords.size(), 3);

          BoundaryTriangleTest triangle;

          for (int dx = 0; dx < 3; dx++) {
            for (int cx = 0; cx < 3; cx++) {
              triangle.vertices[dx][cx] = coords.at(dx).at(cx);
            }
            triangle.normal[dx] = normal_vector.at(dx);
          }
          triangle.face_id = facet_global_id;
          triangle.group_id = group_id;
          triangle.type = 1;
          lambda_push_triangle(triangle);
          triangle_index++;

        } else {

          std::array<std::array<PetscInt, 3>, 2> triangle_indices;
          PetscInterface::split_quadrilateral_into_two_triangles(
              mesh->dmh->dm, point_id, triangle_indices);

          for (int tx : {0, 1}) {
            BoundaryTriangleTest triangle;
            for (int vx : {0, 1, 2}) {
              const PetscInt inner_point_id = triangle_indices.at(tx).at(vx);
              mesh->dmh->get_point_vertices(inner_point_id, coords);
              ASSERT_EQ(coords.size(), 1);
              for (int cx : {0, 1, 2}) {
                triangle.vertices[vx][cx] = coords.at(0).at(cx);
              }
              triangle.normal[vx] = normal_vector.at(vx);
            }
            triangle.face_id = facet_global_id;
            triangle.group_id = group_id;
            triangle.type = 2 + tx;
            lambda_push_triangle(triangle);
            triangle_index++;
          }
        }
      }

      {

        nprint("............................................................");
        std::vector<VTK::UnstructuredCell> vtk_data;

        for (auto &triangle : h_triangles_edge) {

          VTK::UnstructuredCell t;
          t.num_points = 3;
          t.cell_type = VTK::CellType::triangle;
          for (int vx = 0; vx < 3; vx++) {
            for (int cx = 0; cx < 3; cx++) {
              t.points.push_back(triangle.vertices[vx][cx]);
            }
          }

          vtk_data.push_back(t);
        }

        VTK::VTKHDF w("bar_" + std::to_string(item.first) + ".vtkhdf",
                      mesh->get_comm());

        w.write(vtk_data);
        w.close();
      }
    }
  }

  boundary_interaction->wrap_collect_cells();

  BufferDevice<BoundaryTriangleTest> d_triangles(sycl_target, h_triangles);
  auto k_triangles = d_triangles.ptr;
  BufferDevice<int> d_map_to_test(sycl_target, h_map_to_test);
  auto k_map_to_test = d_map_to_test.ptr;

  auto k_intersect_object_root =
      boundary_interaction->get_d_map_facet_discovery()->root;
  auto k_map_facet_normals =
      boundary_interaction->get_d_map_facet_normals()->root;

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  sycl_target->queue
      .parallel_for(
          sycl::range<1>(h_map_to_test.size() / 2),
          [=](auto idx) {
            const std::size_t index = idx.get_id(0);
            const int mh_cell = k_map_to_test[2 * index + 0];
            const int triangle_index = k_map_to_test[2 * index + 1];
            const auto &triangle = k_triangles[triangle_index];

            bool *exists = nullptr;
            PetscInterface::BoundaryInteractionCellData3D *data = nullptr;
            bool found = false;
            if (k_intersect_object_root->get_location(mh_cell, &exists,
                                                      &data)) {

              NESO_KERNEL_ASSERT(*exists, k_ep);
              // naively find the triangle in this MH cell
              const int num_facets = data->num_facets;

              for (int fx = 0; (fx < num_facets) && (!found); fx++) {
                bool all_close = true;
                for (int vx = 0; vx < 3; vx++) {
                  for (int cx = 0; cx < 3; cx++) {
                    const REAL coord_to_find = triangle.vertices[vx][cx];
                    const REAL coord_of_cand = data->d_real[fx * 3 + vx][cx];
                    const REAL err = Kernel::abs(coord_of_cand - coord_to_find);

                    if (err > 1.0e-15) {
                      all_close = false;
                    }
                  }
                }
                found = all_close;
                if (all_close) {
                  NESO_KERNEL_ASSERT(
                      triangle.group_id == data->d_int[fx * 2 + 0], k_ep);
                  NESO_KERNEL_ASSERT(
                      triangle.face_id == data->d_int[fx * 2 + 1], k_ep);

                  PetscInterface::BoundaryInteractionNormalData3D *normal_data =
                      nullptr;
                  if (k_map_facet_normals->get_location(
                          triangle.face_id, &exists, &normal_data)) {

                    const REAL err0 = Kernel::abs(triangle.normal[0] -
                                                  normal_data->d_normal[0]);
                    const REAL err1 = Kernel::abs(triangle.normal[1] -
                                                  normal_data->d_normal[1]);
                    const REAL err2 = Kernel::abs(triangle.normal[2] -
                                                  normal_data->d_normal[2]);

                    NESO_KERNEL_ASSERT(err0 < 1.0e-15, k_ep);
                    NESO_KERNEL_ASSERT(err1 < 1.0e-15, k_ep);
                    NESO_KERNEL_ASSERT(err2 < 1.0e-15, k_ep);
                  }
                }
              }
            }
            NESO_KERNEL_ASSERT(found, k_ep);
          })
      .wait_and_throw();

  ASSERT_FALSE(ep.get_flag());

  boundary_interaction->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary3D, detection) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mixed_ref_cube_0.8.msh");

  const int ndim = 3;

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  int rank = -1;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<REAL>("IP"), ndim),
                             ParticleProp(Sym<REAL>("IN"), ndim),
                             ParticleProp(Sym<INT>("IM"), 2)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  std::mt19937 rng(52234234 + rank);
  REAL extents[3] = {2.0, 2.0, 2.0};

  const int cell_count = mesh->get_cell_count();
  const int N = 10 * cell_count;
  auto positions = uniform_within_extents(N, ndim, extents, rng);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions[dimx][px] - 1.0;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }
  A->add_particles_local(initial_distribution);

  A->hybrid_move();
  A->cell_move();

  std::uniform_real_distribution<> rng_dist(-1.0, 1.0);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng); };
  const int rng_ncomp = 3;
  auto rng_device_kernel =
      host_per_particle_block_rng<REAL>(rng_lambda, rng_ncomp);

  particle_loop(
      A,
      [=](auto INDEX, auto V, auto RNG) {
        REAL v0 = RNG.at(INDEX, 0);
        REAL v1 = RNG.at(INDEX, 1);
        REAL v2 = RNG.at(INDEX, 2);

        const REAL l2 = v0 * v0 + v1 * v1 + v2 * v2;
        const REAL l = l2 != 0.0 ? 1.0 / Kernel::sqrt(l2) : 1.0;
        if (l2 == 0.0) {
          v0 = 1.0;
        }
        V.at(0) = l * v0;
        V.at(1) = l * v1;
        V.at(2) = l * v2;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("V")),
      Access::read(rng_device_kernel))
      ->execute();

  std::vector<int> faces = {100, 200, 300, 400, 500, 600};
  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  for (int ix : faces) {
    boundary_groups[ix] = {ix};
  }

  std::map<PetscInt, std::array<REAL, 3>> map_face_normal;
  map_face_normal[100] = {0.0, -1.0, 0.0};
  map_face_normal[200] = {0.0, 1.0, 0.0};
  map_face_normal[300] = {1.0, 0.0, 0.0};
  map_face_normal[400] = {-1.0, 0.0, 0.0};
  map_face_normal[500] = {0.0, 0.0, -1.0};
  map_face_normal[600] = {0.0, 0.0, 1.0};

  auto boundary_interaction = std::make_shared<BoundaryInteraction3DTest>(
      sycl_target, mesh, boundary_groups, 1.0e-14);

  boundary_interaction->pre_integration(A);

  particle_loop(
      A,
      [=](auto P, auto V) {
        for (int dx = 0; dx < 3; dx++) {
          P.at(dx) += 100.0 * V.at(dx);
        }
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
      ->execute();

  auto groups = boundary_interaction->post_integration(A);

  int count = 0;
  for (auto &gx : groups) {
    count += gx.second->get_npart_local();

    copy_ephemeral_dat_to_particle_dat(
        gx.second, BoundaryInteractionSpecification::intersection_point,
        Sym<REAL>("IP"));
    copy_ephemeral_dat_to_particle_dat(
        gx.second, BoundaryInteractionSpecification::intersection_normal,
        Sym<REAL>("IN"));
    copy_ephemeral_dat_to_particle_dat(
        gx.second, BoundaryInteractionSpecification::intersection_metadata,
        Sym<INT>("IM"));
  }
  ASSERT_EQ(count, A->get_npart_local());

  std::vector<REAL> h_normal(601 * 3);

  for (auto fx : faces) {
    for (int dx = 0; dx < 3; dx++) {
      h_normal.at(fx * 3 + dx) = map_face_normal.at(fx).at(dx);
    }
  }

  BufferDevice<REAL> d_normal(sycl_target, h_normal);
  REAL *k_normal = d_normal.ptr;

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      A,
      [=](auto IP, auto IN, auto IM) {
        const INT group_id = IM.at(0);
        const REAL n[3] = {k_normal[group_id * 3 + 0],
                           k_normal[group_id * 3 + 1],
                           k_normal[group_id * 3 + 2]};

        const REAL errn0 = Kernel::abs(IN.at(0) - n[0]);
        const REAL errn1 = Kernel::abs(IN.at(1) - n[1]);
        const REAL errn2 = Kernel::abs(IN.at(2) - n[2]);

        const bool all_closen =
            errn0 < 1.0e-15 && errn1 < 1.0e-15 && errn2 < 1.0e-15;

        NESO_KERNEL_ASSERT(all_closen, k_ep);

        for (int dx = 0; dx < 3; dx++) {
          const bool in_bounds =
              (IP.at(dx) >= (-1.0 - 1.0e-14)) && (IP.at(dx) <= (1.0 + 1.0e-14));
          NESO_KERNEL_ASSERT(in_bounds, k_ep);
        }
      },
      Access::read(Sym<REAL>("IP")), Access::read(Sym<REAL>("IN")),
      Access::read(Sym<INT>("IM")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  boundary_interaction->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary3D, reflection) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mixed_ref_cube_0.8.msh");

  const int ndim = 3;
  const int Nsteps = 100;
  const REAL dt = 0.05;

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  int rank = -1;
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<REAL>("U"), ndim),
                             ParticleProp(Sym<REAL>("TSP"), 2)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  std::mt19937 rng(52234234 + rank);
  REAL extents[3] = {2.0, 2.0, 2.0};

  const int cell_count = mesh->get_cell_count();
  const int N = 10 * cell_count;
  auto positions = uniform_within_extents(N, ndim, extents, rng);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions[dimx][px] - 1.0;
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }
  A->add_particles_local(initial_distribution);

  A->hybrid_move();
  A->cell_move();

  std::uniform_real_distribution<> rng_dist(-1.0, 1.0);
  auto rng_lambda = [&]() -> REAL { return rng_dist(rng); };
  const int rng_ncomp = 3;
  auto rng_device_kernel =
      host_per_particle_block_rng<REAL>(rng_lambda, rng_ncomp);

  particle_loop(
      A,
      [=](auto INDEX, auto V, auto RNG) {
        REAL v0 = RNG.at(INDEX, 0);
        REAL v1 = RNG.at(INDEX, 1);
        REAL v2 = RNG.at(INDEX, 2);

        const REAL l2 = v0 * v0 + v1 * v1 + v2 * v2;
        const REAL l = l2 != 0.0 ? 1.0 / Kernel::sqrt(l2) : 1.0;
        if (l2 == 0.0) {
          v0 = 1.0;
        }
        V.at(0) = l * v0;
        V.at(1) = l * v1;
        V.at(2) = l * v2;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("V")),
      Access::read(rng_device_kernel))
      ->execute();

  std::vector<int> faces = {100, 200, 300, 400, 500, 600};
  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[0] = faces;

  auto boundary_interaction = std::make_shared<BoundaryInteraction3DTest>(
      sycl_target, mesh, boundary_groups, 1.0e-14);

  auto ep = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = ep->device_ptr();

  auto reflection = std::make_shared<BoundaryReflection>(3, 1.0e-10);

  auto lambda_apply_boundary_conditions = [&](auto aa) {
    auto sub_groups = boundary_interaction->post_integration(aa);

    for (auto &gx : sub_groups) {

      particle_loop(
          gx.second,
          [=](auto V, auto U) {
            for (int dx = 0; dx < 3; dx++) {
              U.at(dx) = V.at(dx);
            }
          },
          Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("U")))
          ->execute();

      reflection->execute(gx.second, Sym<REAL>("P"), Sym<REAL>("V"),
                          Sym<REAL>("TSP"),
                          boundary_interaction->previous_position_sym);

      particle_loop(
          gx.second,
          [=](auto V, auto U) {
            const REAL V_mag =
                V.at(0) * V.at(0) + V.at(1) * V.at(1) + V.at(2) * V.at(2);
            const REAL U_mag =
                U.at(0) * U.at(0) + U.at(1) * U.at(1) + U.at(2) * U.at(2);

            NESO_KERNEL_ASSERT(Kernel::abs(V_mag - U_mag) < 1.0e-14, k_ep);

            if ((Kernel::abs(V.at(0)) > 1.0e-14) &&
                (Kernel::abs(V.at(1)) > 1.0e-14) &&
                (Kernel::abs(V.at(2)) > 1.0e-14)) {
              const bool x_flipped = Kernel::abs(V.at(0) + U.at(0)) < 1.0e-15;
              const bool y_flipped = Kernel::abs(V.at(1) + U.at(1)) < 1.0e-15;
              const bool z_flipped = Kernel::abs(V.at(2) + U.at(2)) < 1.0e-15;
              if (x_flipped) {
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(1) - U.at(1)) < 1.0e-15,
                                   k_ep);
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(2) - U.at(2)) < 1.0e-15,
                                   k_ep);
              }
              if (y_flipped) {
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(0) - U.at(0)) < 1.0e-15,
                                   k_ep);
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(2) - U.at(2)) < 1.0e-15,
                                   k_ep);
              }
              if (z_flipped) {
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(0) - U.at(0)) < 1.0e-15,
                                   k_ep);
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(1) - U.at(1)) < 1.0e-15,
                                   k_ep);
              }
            }
          },
          Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("U")))
          ->execute();
      ASSERT_EQ(ep->get_flag(), 0);
    }
  };

  auto lambda_apply_timestep_reset = [&](auto aa) {
    particle_loop(
        aa,
        [=](auto TSP) {
          TSP.at(0) = 0.0;
          TSP.at(1) = 0.0;
        },
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };
  auto lambda_apply_advection_step =
      [=](ParticleSubGroupSharedPtr iteration_set) -> void {
    particle_loop(
        "euler_advection", iteration_set,
        [=](auto V, auto P, auto TSP) {
          const REAL dt_left = dt - TSP.at(0);
          if (dt_left > 0.0) {
            for (int dx = 0; dx < 3; dx++) {
              P.at(dx) += dt_left * V.at(dx);
            }
            TSP.at(0) = dt;
            TSP.at(1) = dt_left;
          }
        },
        Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("P")),
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };
  auto lambda_pre_advection = [&](auto aa) {
    boundary_interaction->pre_integration(aa);
  };
  auto lambda_find_partial_moves = [&](auto aa) {
    return static_particle_sub_group(
        aa, [=](auto TSP) { return TSP.at(0) < dt; },
        Access::read(Sym<REAL>("TSP")));
  };
  auto lambda_partial_moves_remaining = [&](auto aa) -> bool {
    const int size = get_npart_global(aa);
    return size > 0;
  };
  auto lambda_apply_timestep = [&](auto aa) {
    lambda_apply_timestep_reset(aa);
    lambda_pre_advection(aa);
    lambda_apply_advection_step(aa);
    lambda_apply_boundary_conditions(aa);
    aa = lambda_find_partial_moves(aa);
    while (lambda_partial_moves_remaining(aa)) {
      lambda_pre_advection(aa);
      lambda_apply_advection_step(aa);
      lambda_apply_boundary_conditions(aa);
      aa = lambda_find_partial_moves(aa);
    }
  };

  // H5Part h5part("trajectory.h5part", A, Sym<REAL>("V"));
  for (int stepx = 0; stepx < Nsteps; stepx++) {
    lambda_apply_timestep(static_particle_sub_group(A));
    A->hybrid_move();
    A->cell_move();
    // h5part.write();
  }
  // h5part.close();

  boundary_interaction->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
