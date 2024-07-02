#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"

namespace {
class TestBoundaryInteraction2D : public PetscInterface::BoundaryInteraction2D {
public:
  template <typename... T>
  TestBoundaryInteraction2D(T... args) : BoundaryInteraction2D(args...) {}

  using BoundaryInteractionCellData2D =
      BoundaryInteraction2D::BoundaryInteractionCellData2D;
  using BoundaryInteractionNormalData2D =
      BoundaryInteraction2D::BoundaryInteractionNormalData2D;

  MAKE_WRAP_METHOD(get_labels)
  MAKE_WRAP_METHOD(get_bounding_box)
  MAKE_WRAP_METHOD(collect_cells)
  MAKE_GETTER_METHOD(facets_real)
  MAKE_GETTER_METHOD(facets_int)
  MAKE_GETTER_METHOD(num_facets_global)
  MAKE_GETTER_METHOD(ncomp_int)
  MAKE_GETTER_METHOD(ncomp_real)
  MAKE_GETTER_METHOD(required_mh_cells)
  MAKE_GETTER_METHOD(collected_mh_cells)
  MAKE_GETTER_METHOD(d_map_edge_discovery)
  MAKE_GETTER_METHOD(d_map_edge_normals)
};

TEST(PETScBoundary, constructor_2d) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = 32;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               lower, upper,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2, 3, 4};

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);

  auto labels = b2d->wrap_get_labels();
  ASSERT_TRUE(labels.count(1));
  ASSERT_TRUE(labels.count(2));
  ASSERT_TRUE(labels.count(3));
  ASSERT_TRUE(labels.count(4));

  const int num_facets_global = b2d->get_num_facets_global();
  const int ncomp_int = b2d->get_ncomp_int();
  const int ncomp_real = b2d->get_ncomp_real();
  REAL *ptr_real = b2d->get_facets_real();
  int *ptr_int = b2d->get_facets_int();

  auto lambda_find_int = [&](const int value) -> int {
    for (int ix = 0; ix < num_facets_global; ix++) {
      if (ptr_int[ix * ncomp_int + 1] == value) {
        return ix;
      }
    }
    return -1;
  };

  auto face_sets = mesh->dmh->get_face_sets();
  std::vector<std::vector<REAL>> coords;

  std::vector<REAL> test_real;
  std::vector<int> test_int;

  for (auto &item : face_sets) {
    const PetscInt face_id = item.first;
    if (labels.count(face_id)) {
      for (auto &point_index : item.second) {
        const PetscInt facet_global_id =
            mesh->dmh->get_point_global_index(point_index);

        auto index = lambda_find_int(facet_global_id);
        ASSERT_TRUE(index > -1);
        ASSERT_EQ(ptr_int[index * ncomp_int + 0], face_id);
        ASSERT_EQ(ptr_int[index * ncomp_int + 1], facet_global_id);

        test_int.push_back(face_id);
        test_int.push_back(facet_global_id);

        mesh->dmh->get_generic_vertices(point_index, coords);
        const REAL x0 = coords.at(0).at(0);
        const REAL y0 = coords.at(0).at(1);
        const REAL x1 = coords.at(1).at(0);
        const REAL y1 = coords.at(1).at(1);

        ASSERT_EQ(ptr_real[index * ncomp_real + 0], x0);
        ASSERT_EQ(ptr_real[index * ncomp_real + 1], y0);
        ASSERT_EQ(ptr_real[index * ncomp_real + 2], x1);
        ASSERT_EQ(ptr_real[index * ncomp_real + 3], y1);

        test_real.push_back(x0);
        test_real.push_back(y0);
        test_real.push_back(x1);
        test_real.push_back(y1);

        const REAL n0 = ptr_real[index * ncomp_real + 4];
        const REAL n1 = ptr_real[index * ncomp_real + 5];
        test_real.push_back(n0);
        test_real.push_back(n1);

        // Is the normal vector a unit normal vector
        ASSERT_NEAR(n0 * n0 + n1 * n1, 1.0, 1.0e-15);
        const REAL d0 = x1 - x0;
        const REAL d1 = y1 - y0;
        ASSERT_NEAR(d0 * n0 + d1 * n1, 0.0, 1.0e-15);
      }
    }
  }

  std::vector<REAL> test_global_real;
  std::vector<int> test_global_int;

  all_gather_v(test_real, MPI_COMM_WORLD, test_global_real);
  all_gather_v(test_int, MPI_COMM_WORLD, test_global_int);

  ASSERT_EQ(test_global_real.size(), ncomp_real * num_facets_global);
  ASSERT_EQ(test_global_int.size(), ncomp_int * num_facets_global);

  for (int fx = 0; fx < num_facets_global; fx++) {
    for (int rx = 0; rx < ncomp_real; rx++) {
      ASSERT_EQ(ptr_real[fx * ncomp_real + rx],
                test_global_real.at(fx * ncomp_real + rx));
    }
    for (int ix = 0; ix < ncomp_int; ix++) {
      ASSERT_EQ(ptr_int[fx * ncomp_int + ix],
                test_global_int.at(fx * ncomp_int + ix));
    }
  }

  b2d->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary, collect_2d) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = 32;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               lower, upper,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2, 3, 4};

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);

  const int num_facets_global = b2d->get_num_facets_global();
  REAL *ptr_real = b2d->get_facets_real();
  int *ptr_int = b2d->get_facets_int();

  auto mesh_hierarchy = mesh->get_mesh_hierarchy();
  std::deque<std::pair<INT, double>> cells;

  std::map<INT, std::set<int>> map_mh_index_to_index;
  for (int ex = 0; ex < num_facets_global; ex++) {
    cells.clear();
    auto bb = b2d->wrap_get_bounding_box(ex);
    ExternalCommon::bounding_box_map(bb, mesh_hierarchy, cells);
    for (auto &cx_w : cells) {
      map_mh_index_to_index[cx_w.first].insert(ex);
    }
  }

  auto &required_mh_cells = b2d->get_required_mh_cells();
  auto &collected_mh_cells = b2d->get_collected_mh_cells();

  const INT mh_cell = map_mh_index_to_index.begin()->first;

  ASSERT_EQ(required_mh_cells.size(), 0);
  ASSERT_EQ(collected_mh_cells.size(), 0);

  required_mh_cells.insert(mh_cell);
  b2d->wrap_collect_cells();
  ASSERT_EQ(required_mh_cells.size(), 1);
  ASSERT_EQ(collected_mh_cells.size(), 1);

  auto d_map_edge_discovery = b2d->get_d_map_edge_discovery();
  auto d_map_edge_normals = b2d->get_d_map_edge_normals();

  TestBoundaryInteraction2D::BoundaryInteractionCellData2D cell_data;
  TestBoundaryInteraction2D::BoundaryInteractionNormalData2D normal_data;

  ASSERT_TRUE(d_map_edge_discovery->host_get(mh_cell, &cell_data));

  ASSERT_EQ(cell_data.num_edges, map_mh_index_to_index.at(mh_cell).size());

  const int num_edges = cell_data.num_edges;
  const int ncomp_real = b2d->get_ncomp_real();
  const int ncomp_int = b2d->get_ncomp_int();
  std::vector<REAL> h_real(num_edges * 4);
  std::vector<int> h_int(num_edges * ncomp_int);

  sycl_target->queue
      .memcpy(h_real.data(), cell_data.d_real, num_edges * 4 * sizeof(REAL))
      .wait_and_throw();
  sycl_target->queue
      .memcpy(h_int.data(), cell_data.d_int,
              num_edges * ncomp_int * sizeof(int))
      .wait_and_throw();

  auto lambda_find_int = [&](const int value) -> int {
    for (int ix = 0; ix < num_facets_global; ix++) {
      if (ptr_int[ix * ncomp_int + 1] == value) {
        return ix;
      }
    }
    return -1;
  };

  for (int edgex = 0; edgex < num_edges; edgex++) {
    const int group_id = h_int.at(edgex * ncomp_int + 0);
    const int facet_global_id = h_int.at(edgex * ncomp_int + 1);
    auto index = lambda_find_int(facet_global_id);
    ASSERT_TRUE(index > -1);
    ASSERT_EQ(ptr_real[index * ncomp_real + 0], h_real.at(edgex * 4 + 0));
    ASSERT_EQ(ptr_real[index * ncomp_real + 1], h_real.at(edgex * 4 + 1));
    ASSERT_EQ(ptr_real[index * ncomp_real + 2], h_real.at(edgex * 4 + 2));
    ASSERT_EQ(ptr_real[index * ncomp_real + 3], h_real.at(edgex * 4 + 3));
    ASSERT_EQ(1, h_int.at(edgex * ncomp_int + 0));
    ASSERT_EQ(ptr_int[index * ncomp_int + 1], h_int.at(edgex * ncomp_int + 1));

    ASSERT_TRUE(d_map_edge_normals->host_get(facet_global_id, &normal_data));
    REAL h_normal[2];
    sycl_target->queue.memcpy(h_normal, normal_data.d_normal, 2 * sizeof(REAL))
        .wait_and_throw();

    ASSERT_EQ(ptr_real[index * ncomp_real + 4], h_normal[0]);
    ASSERT_EQ(ptr_real[index * ncomp_real + 5], h_normal[1]);
  }

  b2d->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

} // namespace

#endif
