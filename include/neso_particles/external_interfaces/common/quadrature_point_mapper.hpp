#ifndef _NESO_PARTICLES_QUADRATURE_POINT_MAPPER_H_
#define _NESO_PARTICLES_QUADRATURE_POINT_MAPPER_H_
#include "../../compute_target.hpp"
#include "../../particle_group.hpp"
#include "../../particle_group_impl.hpp"
#include "../../containers/cell_dat_const.hpp"
#include "../../typedefs.hpp"
#include <vector>
#include <stack>

namespace NESO::Particles::ExternalCommon {

class  QuadraturePointMapper {
protected:
  int ndim;
  int rank;
  int size;

  struct Point {
    REAL coords[3];
    int adding_rank;
    int adding_index;
  };
  std::stack<Point> added_points;

public:
  SYCLTargetSharedPtr sycl_target;
  DomainSharedPtr domain;
  ParticleGroupSharedPtr particle_group;

  QuadraturePointMapper() = default;
  
  /**
   * TODO
   */
  QuadraturePointMapper(
    SYCLTargetSharedPtr sycl_target,
    DomainSharedPtr domain
  ):
    sycl_target(sycl_target), ndim(domain->mesh->get_ndim()),
    rank(sycl_target->comm_pair.rank_parent),
    size(sycl_target->comm_pair.size_parent)
  {
    NESOASSERT((0<ndim) && (ndim<4), "Bad number of dimensions.");


  }

  /**
   * TODO
   */
  inline void add_points_initialise(){
    NESOASSERT(this->added_points.empty(),
      "Point creation cannot occur more than once.");
  }

  /**
   * TODO
   */
  inline void add_point(const REAL * point){
    Point p;
    p.adding_rank = this->rank;
    p.adding_index = this->added_points.size();
    for(int dx=0 ; dx<this->ndim ; dx++){
      p.coords[dx] = point[dx];
    }
    this->added_points.push(p);
  }

  /**
   * TODO
   */
  inline void add_points_finalise(){
    ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("ADDING_RANK_INDEX"), 2)
    };

    this->particle_group = std::make_shared<ParticleGroup>(
      this->domain, 
      particle_spec, this->sycl_target);
    
    // Add the quadrature points as particles into the particle group.
    ParticleSet initial_distribution(this->added_points.size(), particle_spec);
    int index = 0;
    while(!this->added_points.empty()){
      const auto t = this->added_points.top();
      initial_distribution[Sym<INT>("ADDING_RANK_INDEX")][index][0] = t.adding_rank;
      initial_distribution[Sym<INT>("ADDING_RANK_INDEX")][index][1] = t.adding_index;
      for(int dx=0 ; dx<this->ndim ; dx++){
        initial_distribution[Sym<REAL>("P")][index][dx] = t.coords[dx];
      }
      this->added_points.pop();
      index++;
    }
    this->particle_group->hybrid_move();
    this->particle_group->cell_move();
    
    // Communicate to the rank which added a quadrature point which rank now
    // holds the point.
    struct RemotePoint {
      int adding_rank;
      int adding_index;
      int cell;
      int layer;
    };

    auto remote_indices = std::make_shared<LocalArray<RemotePoint>>(
      this->sycl_target, this->particle_group->get_npart_local());
    auto remote_indices_counter = std::make_shared<LocalArray<int>>(
      this->sycl_target, 1);
    remote_indices_counter->fill(0);
    
    const int k_rank = this->rank;
    particle_loop(
      "QuadraturePointMapper::add_points_finalise_0",
      this->particle_group,
      [=](
        auto INDEX,
        auto ADDING_RANK_INDEX,
        auto REMOTE_INDICES,
        auto REMOTE_INDICES_COUNTER
      ){
        // Was this point added by a remote rank?
        if (ADDING_RANK_INDEX.at(0) != k_rank) {
          const int tmp_index = REMOTE_INDICES_COUNTER.fetch_add(0, 1);
          RemotePoint p;
          p.adding_rank = ADDING_RANK_INDEX.at(0);
          p.adding_index = ADDING_RANK_INDEX.at(1);
          p.cell = INDEX.cell;
          p.layer = INDEX.layer;

        }
      },
      Access::read(ParticleLoopIndex{}),
      Access::read(Sym<INT>("ADDING_RANK_INDEX")),
      Access::write(remote_indices),
      Access::add(remote_indices_counter)
    )->execute();


  }

  /**
   * TODO
   */
  inline void free(){
    if(this->particle_group){
      this->particle_group->free();
      this->particle_group = nullptr;
    }
  }

};


}

#endif
