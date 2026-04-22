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
              const int tx = d0.get_value(key, vx, 0);
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
