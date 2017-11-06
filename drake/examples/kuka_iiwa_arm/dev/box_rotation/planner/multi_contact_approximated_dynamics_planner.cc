#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/planner/multi_contact_approximated_dynamics_planner.h"

#include "drake/solvers/non_convex_optimization_util.h"
namespace drake {
using solvers::MatrixXDecisionVariable;
using solvers::MatrixDecisionVariable;
using solvers::VectorXDecisionVariable;
using solvers::VectorDecisionVariable;
using solvers::Binding;
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
  Eigen::Matrix<symbolic::Expression, 6, Eigen::Dynamic>
      total_contact_wrench_expected =
          Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, nT_);
  for (int i = 0; i < num_facets; ++i) {
    const auto friction_cone_edge_wrenches_i =
        contact_facets_[i].CalcWrenchConeEdges();
    for (int j = 0; j < contact_facets_[i].NumVertices(); ++j) {
      total_contact_wrench_expected +=
          friction_cone_edge_wrenches_i[j] *
          contact_wrench_weight_[i].block(
              j * contact_facets_[i].NumFrictionConeEdges(), 0,
              contact_facets_[i].NumFrictionConeEdges(), nT_);
    }
  }

  AddDynamicConstraint();
}

void MultiContactApproximatedDynamicsPlanner::AddDynamicConstraint() {
  for (int i = 0; i < nT_; ++i) {
    Vector6<symbolic::Expression> dynamics_linear;
    Vector6<symbolic::Expression> dynamics_quadratic;
    // We write the quadratic and linear part of the dynamics separately.
    // We want the expression
    // m * com_accel - R_WB * force - m * gravity
    // I * omega_dot + omega.cross(I * omega) - torque
    // being zero. This expression can be decomposed as the sum of a quadratic
    // part and a linear part.
    dynamics_linear << m_ * com_accel_.col(i) - m_ * gravity_,
        I_B_ * omega_dot_BpB_.col(i) - total_contact_wrench_.block<3, 1>(3, i);
    dynamics_quadratic << -R_WB_[i].cast<symbolic::Expression>() *
                              total_contact_wrench_.block<3, 1>(0, i),
        0, 0, 0;
    // It is possible that omega.cross(I * omega) is not a quadratic expression,
    // for example, when the inertia matrix I is the identity matrix (the box
    // is a cuboid with identical length along each edge.)
    const Vector3<symbolic::Expression> omega_cross_inertia_times_omega = omega_BpB_.col(i).cross(I_B_ * omega_BpB_.col(i));
    for (int j = 0; j < 3; ++j) {
      if (!is_zero(omega_cross_inertia_times_omega(j))) {
        dynamics_quadratic(3 + j) = omega_cross_inertia_times_omega(j);
      }
    }

    for (int j = 0; j < 6; ++j) {
      if (!is_zero(dynamics_quadratic(j))) {
        Binding<solvers::QuadraticCost> quadratic_expr =
            solvers::internal::ParseQuadraticCost(dynamics_quadratic(j));
        Binding<solvers::LinearCost> linear_expr =
            solvers::internal::ParseLinearCost(dynamics_linear(j));
        Eigen::MatrixXd Q1, Q2;
        std::tie(Q1, Q2) = solvers::DecomposeNonConvexQuadraticForm(
            quadratic_expr.constraint()->Q());
        NonConvexQuadraticConstraint non_convex_quadratic_constraint;
        non_convex_quadratic_constraint.lb = -linear_expr.constraint()->b();
        non_convex_quadratic_constraint.ub = -linear_expr.constraint()->b();
        non_convex_quadratic_constraint.p = linear_expr.constraint()->a();
        non_convex_quadratic_constraint.y = linear_expr.variables();
        non_convex_quadratic_constraint.Q1 = Q1;
        non_convex_quadratic_constraint.Q2 = Q2;
        non_convex_quadratic_constraint.x = quadratic_expr.variables();
        non_convex_quadratic_constraints_.push_back(
            non_convex_quadratic_constraint);
      } else {
        AddLinearConstraint(dynamics_linear(j) == 0);
      }
    }
  }
}
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
