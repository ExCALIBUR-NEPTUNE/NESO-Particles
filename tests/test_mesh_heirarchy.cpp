#include <CL/sycl.hpp>
#include <catch2/catch.hpp>
#include <neso_particles.hpp>
using namespace NESO::Particles;

TEST_CASE("test_mesh_hierarchy_1") {

    SYCLTarget sycl_target{GPU_SELECTOR, MPI_COMM_WORLD};

    std::vector<int> dims(2);
    dims[0] = 2;
    dims[1] = 4;

    MeshHierarchy mh(sycl_target, 2, dims, 2.0, 2);

    REQUIRE(mh.ndim == 2);
    REQUIRE(mh.dims[0] == 2);
    REQUIRE(mh.dims[1] == 4);
    REQUIRE(mh.subdivision_order == 2);
    REQUIRE(mh.cell_width_coarse == 2.0);
    REQUIRE(std::abs(mh.cell_width_fine - 2.0 / std::pow(2, 2)) < 1E-14);
    REQUIRE(mh.ncells_coarse == 8);
    REQUIRE(mh.ncells_fine == 16);
}
