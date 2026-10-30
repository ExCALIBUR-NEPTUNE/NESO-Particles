#ifndef _NESO_PARTICLES_PROFILING
#define _NESO_PARTICLES_PROFILING

#include <chrono>
#include <cstdint>
#include <fstream>
#include <list>
#include <map>
#include <mpi.h>
#include <optional>
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
 * Enum to specify a particular type of profile region, e.g. per particle
 * timing.
 */
enum ProfileRegionType { STANDARD_REGION, PER_PARTICLE_TIMING };

/**
 * Structure to describe an entry in the profiling data structure.
 */
struct ProfileEntry {
  /// Integral value stored on the entry.
  std::int64_t value_integral = 0;
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
  ProfileRegionType profile_region_type{STANDARD_REGION};

  std::size_t profile_region_key1;
  std::size_t profile_region_key2;
  double profile_region_type_time = 0.0;
  double profile_region_type_coeff = 0.0;
  std::size_t num_bytes{0};
  std::size_t num_flops{0};

  ProfileRegion() = default;

  /**
   * Create new ProfileRegion.
   *
   * @param key1 First key for ProfileRegion.
   * @param key2 Second key for ProfileRegion.
   * @param level Optional level for region, default 0.
   */
  ProfileRegion(const std::string key1, const std::string key2,
                const int level = 0)
      : time_start(profile_timestamp()), key1(key1), key2(key2), level(level),
        profile_region_type(STANDARD_REGION) {}

  /**
   * Create new ProfileRegion.
   *
   * @param key1 First key for ProfileRegion.
   * @param key2 Second key for ProfileRegion.
   * @param level Optional level for region, default 0.
   */
  ProfileRegion(const std::string key1, const std::string key2,
                const ProfileRegionType profile_region_type)
      : time_start(profile_timestamp()), key1(key1), key2(key2), level(0),
        profile_region_type(profile_region_type) {}

  /**
   * End the ProfileRegion.
   */
  inline void end() { this->time_end = profile_timestamp(); }

  /**
   * End the ProfileRegion and provide a total number of particles.
   *
   * @param key1 First key identifying the calling code, e.g. the type ID of the
   * calling code.
   * @param key2 Second key for the calling code.
   * @param num_particles Number of particles processed in the region.
   */
  inline void end(const std::size_t key1, const std::size_t key2,
                  const std::size_t num_particles) {
    this->end();
    this->profile_region_key1 = key1;
    this->profile_region_key2 = key2;

    // Here we store the current average time per particle as the time of
    // interest.
    const double current_total_time =
        this->profile_region_type_coeff > 0.0
            ? this->profile_region_type_time * this->profile_region_type_coeff
            : 0.0;

    const double new_total_time =
        current_total_time + profile_elapsed(this->time_start, this->time_end);

    this->profile_region_type_coeff += static_cast<double>(num_particles);
    this->profile_region_type_time =
        new_total_time / this->profile_region_type_coeff;
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

  // Map from keys to per particle timings.
  std::map<std::pair<std::size_t, std::size_t>, double>
      per_particle_timings_map;

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
                  const std::int64_t value_integral,
                  const double value_real = 0.0) {
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
                  const std::int64_t value_integral,
                  const double value_real = 0.0) {
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
    this->regions.clear();
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

protected:
  inline void
  push_per_particle_timing([[maybe_unused]] ProfileRegion &profile_region) {
    if (profile_region.profile_region_type ==
        ProfileRegionType::PER_PARTICLE_TIMING) {
      this->per_particle_timings_map[{profile_region.profile_region_key1,
                                      profile_region.profile_region_key2}] =
          profile_region.profile_region_type_time;
    }
  }

public:
  /**
   * Add a region to this profiling.
   *
   * @param profile_region ProfileRegion to add.
   */
  inline void add_region([[maybe_unused]] ProfileRegion &profile_region) {
    if (this->enabled) {
      this->regions.push_back(profile_region);
      push_per_particle_timing(profile_region);
    }
  }

  /**
   * Start a region if profiling is enabled.
   *
   * @param key1 First key for ProfileRegion.
   * @param key2 Second key for ProfileRegion.
   * @param level Optional level for region, default 0.
   */
  [[nodiscard]] inline std::optional<ProfileRegion>
  start_region(const std::string key1, const std::string key2,
               const int level = 0) {
    if (this->enabled) {
      return ProfileRegion(key1, key2, level);
    } else {
      return {};
    }
  }

  /**
   * Start a region if profiling is enabled for per particle costs.
   *
   * @param key1 First key for ProfileRegion.
   * @param key2 Second key for ProfileRegion.
   * @param profile_region_type Specify a ProfileRegionType for the new region.
   */
  [[nodiscard]] inline std::optional<ProfileRegion>
  start_region(const std::string key1, const std::string key2,
               const ProfileRegionType profile_region_type) {
    if (this->enabled) {
      return ProfileRegion(key1, key2, profile_region_type);
    } else {
      return {};
    }
  }

  /**
   * End a region and add it to the profiling if profiling is enabled.
   *
   * @param profile_region ProfileRegion to end and add to internal container.
   */
  inline void end_region(std::optional<ProfileRegion> &profile_region) {
    if (this->enabled && profile_region) {
      profile_region.value().end();
      this->regions.push_back(profile_region.value());
      push_per_particle_timing(profile_region.value());
    }
  }

  /**
   * End the ProfileRegion and add it to the internal container.
   *
   * @param key1 First key identifying the calling code, e.g. the type ID of the
   * calling code.
   * @param key2 Second key for the calling code.
   * @param num_particles Number of particles processed in the region.
   */
  inline void end_region(std::optional<ProfileRegion> &profile_region,
                         const std::size_t key1, const std::size_t key2,
                         const std::size_t num_particles) {
    if (this->enabled && profile_region) {
      profile_region.value().end(key1, key2, num_particles);
      this->regions.push_back(profile_region.value());
      push_per_particle_timing(profile_region.value());
    }
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
  }
};

} // namespace NESO::Particles

#endif
