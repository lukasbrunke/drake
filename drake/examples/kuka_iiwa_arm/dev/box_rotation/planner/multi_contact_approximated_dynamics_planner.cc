#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/planner/multi_contact_approximated_dynamics_planner.h"

namespace drake {
using solvers::MatrixXDecisionVariable;
using solvers::MatrixDecisionVariable;
using solvers::VectorXDecisionVariable;
using solvers::VectorDecisionVariable;
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
ContactFacet::ContactFacet(
    const Eigen::Ref<const Eigen::Matrix3Xd>& vertices,
    const Eigen::Ref<const Eigen::Matrix3Xd>& friction_cone_edges)
    : vertices_{vertices}, friction_cone_edges_{friction_cone_edges} {}

std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>>
ContactFacet::CalcWrenchConeEdges() const {
  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> wrench_edges(
      NumVertices());
  for (int i = 0; i < NumVertices(); ++i) {
    wrench_edges[i].resize(6, NumFrictionConeEdges());
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

MultiContactApproximatedDynamicsPlanner::
    MultiContactApproximatedDynamicsPlanner(
        double mass, const Eigen::Ref<const Eigen::Matrix3d>& inertia,
        const std::vector<ContactFacet>& contact_facets, int nT,
        int num_arm_patches)
    : solvers::MathematicalProgram(),
      m_{mass},
      I_B_{inertia},
      gravity_{0, 0, -9.81},
      contact_facets_{contact_facets},
      nT_{nT},
      num_arms_patches_{num_arm_patches} {
  if (m_ <= 0) {
    throw std::runtime_error("mass should be positive.");
  }
  if (Eigen::LLT<Eigen::Matrix3d>(I_B_).info() != Eigen::Success) {
    throw std::runtime_error("inertia should be positive definite.");
  }
  int num_facets = contact_facets.size();
  com_pos_ = NewContinuousVariables<3, Eigen::Dynamic>(3, nT_, "r");
  com_vel_ = NewContinuousVariables<3, Eigen::Dynamic>(3, nT_, "rdot");
  com_accel_ = NewContinuousVariables<3, Eigen::Dynamic>(3, nT_, "rddot");
  R_WB_.resize(nT_);
  for (int i = 0; i < nT_; ++i) {
    R_WB_[i] = NewContinuousVariables<3, 3>("R_WB_[" + std::to_string(i) + "]");
  }
  omega_BpB_ = NewContinuousVariables<3, Eigen::Dynamic>(3, nT_, "omega_BpB");
  omega_dot_BpB_ =
      NewContinuousVariables<3, Eigen::Dynamic>(3, nT_, "omega_dot_BpB");

  B_active_facet_ = NewBinaryVariables(num_facets, nT_, "B");
  // The number of active facets at each time sample, is no larger than
  // num_arms_patches_;
  AddLinearConstraint(
      B_active_facet_.cast<symbolic::Expression>().colwise().sum() <=
      Eigen::RowVectorXd::Constant(nT_, num_arms_patches_));
  contact_wrench_weight_.resize(num_facets);
  for (int i = 0; i < num_facets; ++i) {
    contact_wrench_weight_[i] =
        NewContinuousVariables(contact_facets_[i].NumVertices() *
                                   contact_facets_[i].NumFrictionConeEdges(),
                               nT_, "wrench_weight");
    // The weights are all non-negative.
    AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(),
                             contact_wrench_weight_[i]);
  }

  total_contact_wrench_ =
      NewContinuousVariables<6, Eigen::Dynamic>(6, nT_, "w_total");
  Vector6<symbolic::Expression> total_contact_wrench_expected;
  total_contact_wrench_expected << 0, 0, 0, 0, 0, 0;
  for (int i = 0; i < num_facets; ++i) {
    const auto friction_cone_edge_wrenches_i =
        contact_facets_[i].CalcWrenchConeEdges();
    total_contact_wrench_expected +=
        friction_cone_edge_wrenches_i * contact_wrench_weight_[i];
  }
  AddLinearConstraint(total_contact_wrench_.array() ==
                      total_contact_wrench_expected.array());
}

void MultiContactApproximatedDynamicsPlanner::AddLinearDynamicConstraint() {
  for (int i = 0; i < nT_; ++i) {
    VectorDecisionVariable<9, 1> R_WB_flat;
    R_WB_flat << R_WB_[i].col(0), R_WB_[i].col(1), R_WB_[i].col(2);
    // R_WB_plus_force_plus(i, j) should be equal to (R_WB_flat(i) + force(j))²
    auto R_WB_times_force_plus = NewContinuousVariables<9, 3>(
        "R_WB_times_force_+[" + std::to_string(i) + "]");
    auto R_WB_times_force_minus = NewContinuousVariables<9, 3>(
        "R_WB_times_force_-[" + std::to_string(i) + "]");
    Vector3<symbolic::Expression> R_WB_times_force =
        R_WB_[i] * total_contact_wrench_.block<3, 1>(0, i);
  }
}
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
