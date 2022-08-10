#include <CL/sycl.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>
#include <string>

using namespace NESO::Particles;

TEST(ParticleIO, H5Part) {

#ifdef NESO_PARTICLES_HDF5
  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 8;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 0;
  CartesianHMesh mesh(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order);

  SYCLTarget sycl_target{GPU_SELECTOR, mesh.get_comm()};

  // create object to map local cells + stencil to ranks
  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  Domain domain(mesh, cart_local_mapper);

  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true),
      ParticleProp(Sym<REAL>("V"), 3),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("ID"), 1),
      ParticleProp(Sym<INT>("ID2"), 1),
  };

  ParticleGroup A(domain, particle_spec, sycl_target);

  CartesianPeriodic pbc(sycl_target, mesh, A.position_dat);
  CartesianCellBin ccb(sycl_target, mesh, A.position_dat, A.cell_id_dat);

  A.add_particle_dat(ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3),
                                 domain.mesh.get_cell_count()));

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);
  std::mt19937 rng_rank(18241);

  const int N = 1024;

  auto positions =
      uniform_within_extents(N, ndim, mesh.global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  std::uniform_int_distribution<int> uniform_dist(
      0, sycl_target.comm_pair.size_parent - 1);

  ParticleSet initial_distribution(N, A.get_particle_spec());

  // determine which particles should end up on which rank
  std::map<int, std::vector<int>> mapping;
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px;
    initial_distribution[Sym<INT>("ID2")][px][0] = px * 10;
    const auto px_rank = uniform_dist(rng_rank);
    initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
    mapping[px_rank].push_back(px);
  }

  if (sycl_target.comm_pair.rank_parent == 0) {
    A.add_particles_local(initial_distribution);
  }

  A.hybrid_move();
  ccb.execute();
  A.cell_move();

  H5Part h5part("test_dump.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"),
                Sym<INT>("ID"), Sym<INT>("ID2"), Sym<INT>("NESO_MPI_RANK"));

  h5part.write();
  h5part.close();

  if (sycl_target.comm_pair.rank_parent == 0) {
    std::vector<long long> data_ll(N);
    std::vector<double> data_real(N);
    std::vector<long long> ordering(N);

    hid_t file_id = H5Fopen("test_dump.h5part", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t group_step = H5Gopen(file_id, "Step#0", H5P_DEFAULT);

    hid_t dataset = H5Dopen(group_step, "ID_0", H5P_DEFAULT);
    hid_t filespace = H5Dget_space(dataset);
    hsize_t d_rank = H5Sget_simple_extent_ndims(filespace);
    ASSERT_EQ(d_rank, 1);

    hsize_t dims[1];
    H5CHK(H5Sget_simple_extent_dims(filespace, dims, NULL));
    ASSERT_EQ(dims[0], N);

    hid_t memspace = H5Screate_simple(1, dims, NULL);

    H5CHK(H5Dread(dataset, H5T_NATIVE_LLONG, memspace, filespace, H5P_DEFAULT,
                  data_ll.data()));

    for (int px = 0; px < N; px++) {
      const long long tmp_id = data_ll[px];
      ordering[px] = tmp_id;
    }

    H5CHK(H5Sclose(memspace));
    H5CHK(H5Sclose(filespace));
    H5CHK(H5Dclose(dataset));

    auto get_real = [&](std::string attr_name) {
      hid_t dataset = H5Dopen(group_step, attr_name.c_str(), H5P_DEFAULT);
      hid_t filespace = H5Dget_space(dataset);
      hsize_t d_rank = H5Sget_simple_extent_ndims(filespace);
      ASSERT_EQ(d_rank, 1);
      hsize_t dims[1];
      H5CHK(H5Sget_simple_extent_dims(filespace, dims, NULL));
      ASSERT_EQ(dims[0], N);
      hid_t memspace = H5Screate_simple(1, dims, NULL);
      H5CHK(H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, filespace,
                    H5P_DEFAULT, data_real.data()));
      H5CHK(H5Sclose(memspace));
      H5CHK(H5Sclose(filespace));
      H5CHK(H5Dclose(dataset));
    };

    for (int dimx = 0; dimx < 2; dimx++) {
      get_real("P_" + std::to_string(dimx));
      for (int px = 0; px < N; px++) {
        const int orig_px = ordering[px];
        ASSERT_EQ(data_real[px],
                  initial_distribution[Sym<REAL>("P")][orig_px][dimx]);
      }
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      get_real("V_" + std::to_string(dimx));
      for (int px = 0; px < N; px++) {
        const int orig_px = ordering[px];
        ASSERT_EQ(data_real[px],
                  initial_distribution[Sym<REAL>("V")][orig_px][dimx]);
      }
    }

    get_real("x");
    for (int px = 0; px < N; px++) {
      const int orig_px = ordering[px];
      ASSERT_EQ(data_real[px],
                initial_distribution[Sym<REAL>("P")][orig_px][0]);
    }
    get_real("y");
    for (int px = 0; px < N; px++) {
      const int orig_px = ordering[px];
      ASSERT_EQ(data_real[px],
                initial_distribution[Sym<REAL>("P")][orig_px][1]);
    }

    auto get_int = [&](std::string attr_name) {
      hid_t dataset = H5Dopen(group_step, attr_name.c_str(), H5P_DEFAULT);
      hid_t filespace = H5Dget_space(dataset);
      hsize_t d_rank = H5Sget_simple_extent_ndims(filespace);
      ASSERT_EQ(d_rank, 1);
      hsize_t dims[1];
      H5CHK(H5Sget_simple_extent_dims(filespace, dims, NULL));
      ASSERT_EQ(dims[0], N);
      hid_t memspace = H5Screate_simple(1, dims, NULL);
      H5CHK(H5Dread(dataset, H5T_NATIVE_LLONG, memspace, filespace, H5P_DEFAULT,
                    data_ll.data()));
      H5CHK(H5Sclose(memspace));
      H5CHK(H5Sclose(filespace));
      H5CHK(H5Dclose(dataset));
    };

    get_int("ID2_0");
    for (int px = 0; px < N; px++) {
      const int orig_px = ordering[px];
      ASSERT_EQ(data_ll[px], initial_distribution[Sym<INT>("ID2")][orig_px][0]);
    }

    H5CHK(H5Gclose(group_step));
    H5CHK(H5Fclose(file_id));
  }

  mesh.free();

#endif
}
