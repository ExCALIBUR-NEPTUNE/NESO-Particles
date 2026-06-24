#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>
#include <string>

using namespace NESO::Particles;

TEST(PETSc, dmplex_interface_3d_base) {
  std::filesystem::path gmsh_filepath;
  // GET_TEST_RESOURCE(gmsh_filepath,
  // "gmsh/reference_all_types_square_0.2.msh");

  nprint("TODO commit a mesh");
  gmsh_filepath = get_env_string("GMSH_TMP", "");

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

  nprint("TODO fix ordering");
  // auto vtk_data = mesh_helper->get_vtk_cell_data();
  // VTK::VTKHDF vtkhdf("foo.vtkhdf", MPI_COMM_WORLD);
  // vtkhdf.write(vtk_data);
  // vtkhdf.close();

  std::vector<PetscScalar> point{0.1, 0.1, 0.1};
  mesh_helper->cell_contains_point_3d(0, point);

  nprint("TODO cleanup");
  for (int cellx = 0; cellx < mesh_helper->get_cell_count(); cellx++) {
    const bool contained = mesh_helper->cell_contains_point_3d(cellx, point);
    if (contained) {
      nprint("FOUND", cellx);
    }
  }

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  // ASSERT_TRUE(mesh->validate_halos(false));
  nprint("TODO revert to false");
  ASSERT_TRUE(mesh->validate_halos(true));

  const double volume = mesh->dmh->get_volume();
  ASSERT_NEAR(volume, 8.0, 1.0e-10);

  mesh->free();

  mesh_helper->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSc, dmplex_3d_mapper) {
  std::filesystem::path gmsh_filepath;
  // GET_TEST_RESOURCE(gmsh_filepath,
  // "gmsh/reference_all_types_square_0.2.msh");

  nprint("TODO commit a mesh");
  gmsh_filepath = get_env_string("GMSH_TMP", "");

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

  // auto vtk_data = mesh->dmh->get_vtk_cell_data();
  // VTK::VTKHDF vtkhdf("foo.vtkhdf", MPI_COMM_WORLD);
  // for(int cellx=0 ; cellx<cell_count ; cellx++){
  //   vtk_data.at(cellx).cell_data["CELL_ID"] = cellx;
  // }
  // vtkhdf.write(vtk_data);
  // vtkhdf.close();
  // H5Part h5part("bar.h5part", A, Sym<INT>("CELL_ID"));
  // const int Nsteps = 1;
  // for(int stepx=0 ; stepx<Nsteps ; stepx++){
  //   h5part.write();
  //   h5part.close();
  //   A->hybrid_move();
  //   A->cell_move();
  // }

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

  MAKE_GETTER_METHOD(required_mh_cells);
  MAKE_GETTER_METHOD(collected_mh_cells);
  MAKE_GETTER_METHOD(padding);
  MAKE_GETTER_METHOD(d_map_facet_discovery);
  MAKE_GETTER_METHOD(map_label_to_groups);
  MAKE_GETTER_METHOD(d_map_facet_normals);
  MAKE_WRAP_METHOD(collect_cells);
  MAKE_WRAP_METHOD(get_labels);
};

struct BoundaryTriangleTest {
  REAL vertices[3][3];
  REAL normal[3];
  int face_id;
  int group_id;
  int type;
};

} // namespace

TEST(PETScBoundary3D, setup) {
  std::filesystem::path gmsh_filepath;
  // GET_TEST_RESOURCE(gmsh_filepath,
  // "gmsh/reference_all_types_square_0.2.msh");

  nprint("TODO commit a mesh");
  gmsh_filepath = get_env_string("GMSH_TMP", "");

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
          for (auto &cell_weight : cells) {
            h_map_to_test.push_back(cell_weight.first);
            h_map_to_test.push_back(triangle_index);
          }
        };

        if (is_triangle) {
          mesh->dmh->get_generic_vertices(point_id, coords);
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
              mesh->dmh->get_generic_vertices(inner_point_id, coords);
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

TEST(PETSc, foo) {
  std::filesystem::path gmsh_filepath;
  // GET_TEST_RESOURCE(gmsh_filepath,
  // "gmsh/reference_all_types_square_0.2.msh");

  nprint("TODO commit a mesh");
  gmsh_filepath = get_env_string("GMSH_TMP", "");

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

  {
    auto vtk_data = mesh->dmh->get_vtk_cell_data();

    for (auto &element : vtk_data) {
      const int num_vertices = element.points.size() / 3;
      for (int vx = 0; vx < num_vertices; vx++) {
        const REAL x = element.points.at(3 * vx + 0);
        const REAL y = element.points.at(3 * vx + 1);
        const REAL z = element.points.at(3 * vx + 2);
        element.point_data["x"].push_back(x);
        element.point_data["y"].push_back(y);
        element.point_data["z"].push_back(z);
      }
      element.cell_data["rank"] = rank;
    }

    VTK::VTKHDF vtkhdf("foo.vtkhdf", MPI_COMM_WORLD);
    vtkhdf.write(vtk_data);
    vtkhdf.close();
  }

  nprint("BEFORE HALO VTK");

  {
    auto vtk_data = mesh->dmh_halo->get_vtk_cell_data();

    for (auto &element : vtk_data) {
      element.cell_data["rank"] = rank;
    }

    VTK::VTKHDF vtkhdf("foo_halo_" + std::to_string(rank) + ".vtkhdf",
                       MPI_COMM_SELF);
    vtkhdf.write(vtk_data);
    vtkhdf.close();
  }

  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
