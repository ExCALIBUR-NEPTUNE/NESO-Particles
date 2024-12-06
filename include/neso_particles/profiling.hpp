#ifndef _NESO_PARTICLES_PROFILING
#define _NESO_PARTICLES_PROFILING

#include <chrono>
#include <cstdint>
#include <fstream>
#include <list>
#include <map>
#include <mpi.h>
#include <string>

#include "typedefs.hpp"

namespace NESO::Particles {

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
 * Struct to hold a ProfileRegion for profiling.
 */
struct ProfileRegion {
  std::chrono::high_resolution_clock::time_point time_start;
  std::chrono::high_resolution_clock::time_point time_end;
  std::string key1;
  std::string key2;
  int level;
  std::size_t num_bytes{0};
  std::size_t num_flops{0};

  ProfileRegion() = default;

#ifdef NESO_PARTICLES_PROFILING_REGION
  /**
   * Create new ProfileRegion.
   *
   * @param key1 First key for ProfileRegion.
   * @param key2 Second key for ProfileRegion.
   */
  ProfileRegion(const std::string key1, const std::string key2,
                const int level = 0)
      : time_start(profile_timestamp()), key1(key1), key2(key2), level(level) {}

#else

  /**
   * Create new ProfileRegion.
   *
   * @param key1 First key for ProfileRegion.
   * @param key2 Second key for ProfileRegion.
   */
  ProfileRegion([[maybe_unused]] const std::string key1,
                [[maybe_unused]] const std::string key2,
                [[maybe_unused]] const int level = 0) {}
#endif

  /**
   * End the ProfileRegion.
   */
  inline void end() {
#ifdef NESO_PARTICLES_PROFILING_REGION
    this->time_end = profile_timestamp();
#endif
  }
};

/**
 *  Data structure for profiling data. Fundamentally is a map from key to key
 *  to value.
 */
class ProfileMap {
private:
protected:
public:
  /// Reference time for start of profiling.
  std::chrono::high_resolution_clock::time_point time_start;

  /// Main data structure storing profiling data.
  std::map<std::string, std::map<std::string, ProfileEntry>> profile;

  /// Main data structure for storing events.
  std::list<std::tuple<std::string, std::string,
                       std::chrono::high_resolution_clock::time_point>>
      events;

  // Main data structure for storing regions.
  std::list<ProfileRegion> regions;

  // Is recording of events enabled?
  bool enabled{false};

  /**
   * Enable recording of events and regions.
   */
  inline void enable() { this->enabled = true; }

  /**
   * Disable recording of events and regions.
   */
  inline void disable() { this->enabled = false; }

  ~ProfileMap(){};

  /**
   * Construct a new empty instance.
   */
  ProfileMap() { this->reset(); };

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
    if (this->enabled) {
      this->profile[key1][key2].value_integral = value_integral;
      this->profile[key1][key2].value_real = value_real;
    }
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
    if (this->enabled) {
      this->profile[key1][key2].value_integral += value_integral;
      this->profile[key1][key2].value_real += value_real;
    }
  };

  /**
   * Reset the profiling data by emptying the current set of keys and values.
   */
  inline void reset() {
    this->profile.clear();
    this->time_start = profile_timestamp();
  };

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

  /**
   * Record a profiling event in the data structure.
   *
   * @param key1 First key for the entry.
   * @param key2 Second key for the entry.
   */
  inline void event(const std::string key1, const std::string key2) {
    if (this->enabled) {
      auto t = profile_timestamp();
      this->events.emplace_back(key1, key2, t);
    }
  }

  /**
   * Add a region to this profiling.
   *
   * @param profile_region ProfileRegion to add.
   */
  inline void add_region([[maybe_unused]] ProfileRegion &profile_region) {
#ifdef NESO_PARTICLES_PROFILING_REGION
    if (this->enabled) {
      this->regions.push_back(profile_region);
    }
#endif
  }

  /**
   * Print the event data to stdout.
   */
  inline void print_events() {
    for (auto &ex : this->events) {
      nprint(std::get<0>(ex), std::get<1>(ex),
             profile_elapsed(this->time_start, std::get<2>(ex)));
    }
  }

  /**
   * Write events and regions to JSON file.
   */
  inline void write_events_json([[maybe_unused]] std::string basename,
                                [[maybe_unused]] const int rank) {
#ifdef NESO_PARTICLES_PROFILING_REGION
    basename += "." + std::to_string(rank) + ".json";
    std::ofstream fh;
    fh.open(basename);
    const int num_events = this->events.size();
    fh << "{\n";
    fh << "\"rank\":" << rank << ",\n";
    fh << "\"events\":" << "[\n";
    int ei = 0;
    for (const auto &ex : this->events) {
      const auto e0 = std::get<0>(ex);
      const auto e1 = std::get<1>(ex);
      const auto e2 = profile_elapsed(this->time_start, std::get<2>(ex));
      fh << "[\"" << e0 << "\",\"" << e1 << "\"," << e2 << "]";
      ei++;
      fh << ((ei < num_events) ? ",\n" : "\n");
    }
    fh << "],\n";
    fh << "\"regions\":";
    fh << "[\n";
    const int num_regions = this->regions.size();
    int ri = 0;
    for (const auto &rx : this->regions) {
      const auto e0 = rx.key1;
      const auto e1 = rx.key2;
      const auto e2 = profile_elapsed(this->time_start, rx.time_start);
      const auto e3 = profile_elapsed(this->time_start, rx.time_end);
      const auto e4 = rx.level;
      const auto e5 = rx.num_bytes;
      const auto e6 = rx.num_flops;
      fh << "[\"" << e0 << "\",\"" << e1 << "\"," << e2 << "," << e3 << ","
         << e4 << "," << e5 << "," << e6 << "]";
      ri++;
      fh << ((ri < num_regions) ? ",\n" : "\n");
    }

    fh << "]\n";
    fh << "}\n";
    fh.close();
#endif
  }
};

} // namespace NESO::Particles

#endif
