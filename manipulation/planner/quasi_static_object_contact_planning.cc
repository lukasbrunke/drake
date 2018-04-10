#include "drake/manipulation/planner/quasi_static_object_contact_planning.h"

namespace drake {
namespace manipulation {
namespace planner {
QuasiStaticObjectContactPlanning::QuasiStaticObjectContactPlanning(
      int nT, double mass, const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV, int num_pushers,
      const std::vector<BodyContactPoint>& Q,
      bool add_second_order_cone_for_R)
  : ObjectContactPlanning(nT, mass, p_BC, p_BV, num_pushers, Q, add_second_order_cone_for_R) {}

void QuasiStaticObjectContactPlanning::AddStaticEquilibriumConstraint() {
  for (int knot = 0; knot < nT(); ++knot) {
    AddStaticEquilibriumConstraintAtKnot(knot);
  }
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
