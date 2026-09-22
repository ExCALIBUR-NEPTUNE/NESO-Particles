// clang-format off
inline void pair_loop_example_simple(
    CellwisePairListAbsolute<ParticleGroup, CellwisePairList> pair_list
) {
  particle_pair_loop(
    pair_list,
    [=](auto P_A, auto P_B, auto Q_A, auto Q_B) {
      // Distance between the two particles.
      const REAL r0 = P_B.at(0) - P_A.at(0);
      const REAL r1 = P_B.at(1) - P_A.at(1);
      const REAL diff_squared = r0*r0 + r1*r1;

      // If the two particles are close swap the propery Q.
      if (diff_squared < 0.1){
        const REAL Q_tmp = Q_A.at(0);
        Q_A.at(0) = Q_B.at(0);
        Q_B.at(0) = Q_tmp;
      }
    },
    Access::A(Access::read(Sym<REAL>("P"))),
    Access::B(Access::read(Sym<REAL>("P"))),
    Access::A(Access::write(Sym<REAL>("Q"))),
    Access::B(Access::write(Sym<REAL>("Q")))
  )->execute();
}
// clang-format on
