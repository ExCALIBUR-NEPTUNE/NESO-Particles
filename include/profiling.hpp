#ifndef _NESO_PARTICLES_PROFILING
#define _NESO_PARTICLES_PROFILING

#include <CL/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <map>
#include <mpi.h>
#include <string>

#include "typedefs.hpp"

using namespace cl;

namespace NESO::Particles {

struct ProfileEntry {
  int64_t value_integral = 0;
  double value_real = 0.0;
};

class ProfileMap {
private:
public:
  std::map<std::string, std::map<std::string, ProfileEntry>> profile;
  ~ProfileMap(){};
  ProfileMap(){};

  inline void set(const std::string key1, const std::string key2,
                  const int64_t value_integral, const double value_real = 0.0) {
    this->profile[key1][key2].value_integral = value_integral;
    this->profile[key1][key2].value_real = value_real;
  };

  inline void inc(const std::string key1, const std::string key2,
                  const int64_t value_integral, const double value_real = 0.0) {
    this->profile[key1][key2].value_integral += value_integral;
    this->profile[key1][key2].value_real += value_real;
  };

  inline void reset() { this->profile.clear(); };

  inline void print() {
    for (auto &key1x : this->profile) {
      std::cout << key1x.first << ":" << std::endl;
      for (auto &key2x : key1x.second) {
        std::cout << "  " << key2x.first << ": " << key2x.second.value_integral
                  << " | " << key2x.second.value_real << std::endl;
      }
    }
  };
};

inline std::chrono::high_resolution_clock::time_point profile_timestamp() {
  return std::chrono::high_resolution_clock::now();
}

inline double
profile_elapsed(std::chrono::high_resolution_clock::time_point time_start,
                std::chrono::high_resolution_clock::time_point time_end) {
  std::chrono::duration<double> time_taken = time_end - time_start;
  const double time_taken_double = (double)time_taken.count();
  return time_taken_double;
}

} // namespace NESO::Particles

#endif
