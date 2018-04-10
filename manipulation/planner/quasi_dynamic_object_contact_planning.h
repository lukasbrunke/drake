#pragma once

#include "drake/manipulation/planner/object_contact_planning.h"

namespace drake {
namespace manipulation {
namespace planner {
class QuasiDynamicObjectContactPlanning : public ObjectContactPlanning {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuasiDynamicObjectContactPlanning)

  QuasiDynamicObjectContactPlanning(
      int nT, double mass, const Eigen::Ref<const Eigen::Matrix3d>& I_B,
      const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV, int num_pushers,
      const std::vector<BodyContactPoint>& Q,
      bool add_second_order_cone_for_R = false);

  ~QuasiDynamicObjectContactPlanning() = default;

 private:
  Eigen::Matrix3d I_B_;
};
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
