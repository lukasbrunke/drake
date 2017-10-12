#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/planner/multicontact_time_optimal_planner.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
ContactFacet::ContactFacet(
    const Eigen::Ref<const Eigen::Matrix3Xd>& vertices,
    const Eigen::Ref<const Eigen::Matrix3Xd>& friction_cone_edges)
    : vertices_{vertices}, friction_cone_edges_{friction_cone_edges} {}

std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>>
ContactFacet::WrenchConeEdges() const {
  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> wrench_edges(
      num_vertices());
  for (int i = 0; i < num_vertices(); ++i) {
    wrench_edges[i].resize(6, num_friction_cone_edges());
    wrench_edges[i].topRows<3>() = friction_cone_edges_;
    // vertex_tilde represents the cross product with vertex, namely
    // vertex_tilde * a = vertex.cross(a) for any vector a.
    Eigen::Matrix3d vertex_tilde;
    vertex_tilde << 0, -vertices_(2, i), vertices_(1, i), vertices_(2, i), 0,
        -vertices_(0, i), -vertices_(1, i), vertices_(0, i), 0;
    wrench_edges[i].bottomRows<3>() = vertex_tilde * friction_cone_edges_;
  }
  return wrench_edges;
}

MultiContactTimeOptimalPlanner::MultiContactTimeOptimalPlanner(
    double mass, const Eigen::Ref<const Eigen::Matrix3d>& inertia,
    const std::vector<ContactFacet>& contact_facets, int nT, int num_arms)
    : solvers::MathematicalProgram(),
      m_(mass),
      I_B_(inertia),
      contact_facets_(contact_facets),
      gravity_(0, 0, -9.81),
      nT_(nT),
      num_arms_(num_arms),
      s_(Eigen::VectorXd::LinSpaced(nT - 1, 0, 1)) {
  theta_ = NewContinuousVariables(nT_, "theta");
  // θ = ṡ², so θ >= 0.
  AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), theta_);

  int num_facets = contact_facets_.size();
  lambda_.reserve(num_facets);
  for (int i = 0; i < num_facets; ++i) {
    lambda_.push_back(
        NewContinuousVariables(contact_facets_[i].num_vertices() *
                                   contact_facets_[i].num_friction_cone_edges(),
                               nT_, "lambda[" + std::to_string(i) + "]"));
    // This is the contact wrench cone constraint, that the weight has to be
    // non-negative.
    AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(),
                             lambda_[i]);
  }
  B_ = NewBinaryVariables(num_facets, nT_, "B");
  // The contact wrench is active, only when the contact facet is active. Use
  // the big-M approach.
  for (int i = 0; i < num_facets; ++i) {
    // sum_k λ[i](k, j) <= 3 * mg * B(i, j)
    // Here we choose 3 * mg as the big M number.
    AddLinearConstraint(
        lambda_[i].cast<symbolic::Expression>().colwise().sum().transpose() <=
        3 * m_ * 9.81 * B_.row(i).cast<symbolic::Expression>().transpose());
  }
  // The number of active facets at each time point, is no larger than the
  // number of arms.
  AddLinearConstraint(
      B_.cast<symbolic::Expression>().colwise().sum().transpose() <=
      num_arms_ * Eigen::VectorXd::Ones(nT_));
}

Eigen::Matrix<symbolic::Expression, 3, 1>
MultiContactTimeOptimalPlanner::x_accel(
    int i, const Eigen::Ref<const Eigen::Vector3d>& x_prime,
    const Eigen::Ref<const Eigen::Vector3d>& x_double_prime) const {
  return x_double_prime * theta_(i) + x_prime * s_ddot(i);
}

std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>
MultiContactTimeOptimalPlanner::com_path_prime(
    const Eigen::Ref<const Eigen::Matrix3Xd>& com_path) const {
  // We assume that
}

void MultiContactTimeOptimalPlanner::SetObjectPoseSequence(
    const std::vector<Eigen::Isometry3d>& object_pose) {
  // First compute dr_ds and ddr_dds
  // Suppose that CoM position r is a piecewise linear function of s,
  // and s is evenly spaced between 0 and 1.
}
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
