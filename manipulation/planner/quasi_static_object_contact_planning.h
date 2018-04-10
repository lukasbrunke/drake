#pragma once

#include "drake/manipulation/planner/object_contact_planning.h"

namespace drake {
namespace manipulation {
namespace planner {
class QuasiStaticObjectContactPlanning : public ObjectContactPlanning {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuasiStaticObjectContactPlanning)

  QuasiStaticObjectContactPlanning(
      int nT, double mass, const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV, int num_pushers,
      const std::vector<BodyContactPoint>& Q,
      bool add_second_order_cone_for_R = false);

  /**
   * Adds static equilibrium constraint for each knot
   */
  void AddStaticEquilibriumConstraint();

  ~QuasiStaticObjectContactPlanning() = default;
};
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
