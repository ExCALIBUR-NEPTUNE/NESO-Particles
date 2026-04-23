#include "include/test_neso_particles.hpp"

TEST(IndexMap, device) {

  {
    IndexMapDevice<2, 1> d0;
    ASSERT_EQ(d0.key_dim, 2);
    ASSERT_EQ(d0.value_dim, 1);
  }

  {
    const int N0 = 7;
    const int N1 = 5;
    const int N2 = 3;
    const int max_num_values = 13;

    const int N = N0 * N1 * N2;
    std::vector<INT> h_offsets(N + 1);
    std::vector<int> h_values(N * max_num_values);

    IndexMapDevice<3, 1> d0 = {
        h_offsets.data(), {h_values.data()}, {N0, N1, N2}};

    ASSERT_EQ(d0.key_strides[0], N0);
    ASSERT_EQ(d0.key_strides[1], N1);
    ASSERT_EQ(d0.key_strides[2], N2);

    std::map<INT, std::vector<int>> map_key_values;

    {
      INT linear_index = 0;
      INT offset = 0;
      int values = 0;
      for (int i0 = 0; i0 < N0; i0++) {
        for (int i1 = 0; i1 < N1; i1++) {
          for (int i2 = 0; i2 < N2; i2++) {
            const int key[3] = {i0, i1, i2};
            const INT to_test = d0.get_linear_index(key);
            ASSERT_EQ(to_test, linear_index);
            const INT Nvalues = linear_index % max_num_values;
            h_offsets.at(linear_index) = offset;

            for (int vx = 0; vx < Nvalues; vx++) {
              h_values.at(offset + vx) = values;
              map_key_values[linear_index].push_back(values);
              values++;
            }

            linear_index++;
            offset += Nvalues;
          }
        }
      }
      h_offsets.at(linear_index) = offset;
    }

    {
      INT linear_index = 0;
      INT offset = 0;
      for (int i0 = 0; i0 < N0; i0++) {
        for (int i1 = 0; i1 < N1; i1++) {
          for (int i2 = 0; i2 < N2; i2++) {
            const int key[3] = {i0, i1, i2};
            const INT to_test = d0.get_num_values(key);
            const INT Nvalues = linear_index % max_num_values;
            ASSERT_EQ(to_test, Nvalues);

            const INT to_test_offset = d0.get_offset(key);
            ASSERT_EQ(to_test_offset, offset);

            const auto &c_values = map_key_values[linear_index];
            std::vector<int> t_values;
            for (int vx = 0; vx < Nvalues; vx++) {
              const int tx = d0.at(key, vx, 0);
              t_values.push_back(tx);
            }
            ASSERT_EQ(t_values, c_values);

            linear_index++;
            offset += Nvalues;
          }
        }
      }
    }
  }
}

TEST(IndexMap, host) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  {
    auto im = get_index_map<3, 2>(sycl_target);

    const int N0 = 7;
    const int N1 = 5;
    const int N2 = 3;
    const int max_num_values = 13;

    int key_strides[3] = {N0, N1, N2};
    im->set_key_strides(key_strides);

    auto d_tmp_num_values = im->get_tmp_buffer_num_values();
    std::vector<INT> h_tmp_num_values(d_tmp_num_values->size);
    std::vector<INT> h_correct_offsets(d_tmp_num_values->size);
    std::vector<INT> h_to_test_offsets(d_tmp_num_values->size);

    INT offset = 0;
    INT linear_index = 0;
    for (int i0 = 0; i0 < N0; i0++) {
      for (int i1 = 0; i1 < N1; i1++) {
        for (int i2 = 0; i2 < N2; i2++) {
          const INT Nvalues = linear_index % max_num_values;
          h_tmp_num_values.at(linear_index) = Nvalues;
          h_correct_offsets.at(linear_index) = offset;
          offset += Nvalues;
          linear_index++;
        }
      }
    }
    h_correct_offsets.at(linear_index) = offset;

    sycl_target->queue
        .memcpy(d_tmp_num_values->ptr, h_tmp_num_values.data(),
                d_tmp_num_values->size_bytes())
        .wait_and_throw();

    im->populate_offsets_buffer(d_tmp_num_values->ptr);
    im->restore_tmp_buffer_num_values(d_tmp_num_values);

    auto d_im = im->get_device();

    sycl_target->queue
        .memcpy(h_to_test_offsets.data(), d_im.d_offsets,
                h_to_test_offsets.size() * sizeof(INT))
        .wait_and_throw();

    ASSERT_EQ(h_correct_offsets, h_to_test_offsets);

    ErrorPropagate ep(sycl_target);
    auto k_ep = ep.device_ptr();

    sycl_target->queue
        .parallel_for(sycl::range<3>(N0, N1, N2),
                      [=](sycl::item<3> idx) {
                        const int key[3] = {static_cast<int>(idx.get_id(0)),
                                            static_cast<int>(idx.get_id(1)),
                                            static_cast<int>(idx.get_id(2))};
                        const INT num_values_to_test = d_im.get_num_values(key);
                        const INT linear_index = d_im.get_linear_index(key);
                        const INT num_values_correct =
                            linear_index % max_num_values;
                        NESO_KERNEL_ASSERT(
                            num_values_correct == num_values_to_test, k_ep);
                        for (INT ix = 0; ix < num_values_to_test; ix++) {
                          d_im.at(key, ix, 0) = 12391 + linear_index + ix;
                          d_im.at(key, ix, 1) = 107 + linear_index + 2 * ix;
                        }
                      })
        .wait_and_throw();

    ASSERT_FALSE(ep.get_flag());

    sycl_target->queue
        .parallel_for(
            sycl::range<3>(N0, N1, N2),
            [=](sycl::item<3> idx) {
              const int key[3] = {static_cast<int>(idx.get_id(0)),
                                  static_cast<int>(idx.get_id(1)),
                                  static_cast<int>(idx.get_id(2))};
              const INT num_values_to_test = d_im.get_num_values(key);
              const INT linear_index = d_im.get_linear_index(key);
              for (INT ix = 0; ix < num_values_to_test; ix++) {
                NESO_KERNEL_ASSERT(
                    d_im.at(key, ix, 0) = 12391 + linear_index + ix, k_ep);
                NESO_KERNEL_ASSERT(
                    d_im.at(key, ix, 1) = 107 + linear_index + 2 * ix, k_ep);
              }
            })
        .wait_and_throw();

    ASSERT_FALSE(ep.get_flag());

    restore_index_map(sycl_target, im);
  }

  sycl_target->free();
}
