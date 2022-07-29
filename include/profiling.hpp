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

/**
 * Structure to describe an entry in the profiling data structure.
 */
struct ProfileEntry {
  /// Integral value stored on the entry.
  int64_t value_integral = 0;
  /// Floating point value stored on the entry.
  double value_real = 0.0;
};

/**
 *  Data structure for profiling data. Fundamentally is a map from key to key
 *  to value.
 */
class ProfileMap {
private:
public:
  /// Main data structure storing profiling data.
  std::map<std::string, std::map<std::string, ProfileEntry>> profile;
  ~ProfileMap(){};

  /**
   * Construct a new empty instance.
   */
  ProfileMap(){};

  /**
   * Set or create a new profiling entry in the data structure.
   *
   * @param key1 First key for the entry.
   * @param key2 Second key for the entry.
   * @param value_integral Integral value for the entry.
   * @param value_real Floating point value for the entry.
   */
  inline void set(const std::string key1, const std::string key2,
                  const int64_t value_integral, const double value_real = 0.0) {
    this->profile[key1][key2].value_integral = value_integral;
    this->profile[key1][key2].value_real = value_real;
  };

  /**
   * Increment a profiling entry in the data structure.
   *
   * @param key1 First key for the entry.
   * @param key2 Second key for the entry.
   * @param value_integral Integral value for the entry to add to current value.
   * @param value_real Floating point value for the entry to add to current
   * value.
   */
  inline void inc(const std::string key1, const std::string key2,
                  const int64_t value_integral, const double value_real = 0.0) {
    this->profile[key1][key2].value_integral += value_integral;
    this->profile[key1][key2].value_real += value_real;
  };

  /**
   * Reset the profiling data by emptying the current set of keys and values.
   */
  inline void reset() { this->profile.clear(); };

  /**
   * Print the profiling data.
   */
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

/**
 * Get a time stamp that can be used with profile_elapsed.
 *
 * @returns Time stamp.
 */
inline std::chrono::high_resolution_clock::time_point profile_timestamp() {
  return std::chrono::high_resolution_clock::now();
}

/**
 *  Compute and return the time in seconds between two time stamps created with
 *  profile_timestamp.
 *
 *  @param time_start Start time stamp.
 *  @param time_end End time stamp.
 *  @return Elapsed time in seconds between time stamps.
 */
inline double
profile_elapsed(std::chrono::high_resolution_clock::time_point time_start,
                std::chrono::high_resolution_clock::time_point time_end) {
  std::chrono::duration<double> time_taken = time_end - time_start;
  const double time_taken_double = (double)time_taken.count();
  return time_taken_double;
}

} // namespace NESO::Particles

#endif
