#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/planner/multicontact_time_optimal_planner.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
MultiContactTimeOptimalPlanner::MultiContactTimeOptimalPlanner(
    double mass, const Eigen::Ref<const Eigen::Matrix3d>& inertia,
    const std::vector<ContactFacet>& contact_facets, int nT)
    : solvers::MathematicalProgram(),
      m_(mass),
      I_B_(inertia),
      contact_facets_(contact_facets),
      gravity_(0, 0, -9.81),
      nT_(nT) {
  theta_ = NewContinuousVariables(nT_, "theta");
}
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
