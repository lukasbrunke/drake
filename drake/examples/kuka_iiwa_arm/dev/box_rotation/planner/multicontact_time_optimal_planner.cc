#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/planner/multicontact_time_optimal_planner.h"

#include "drake/common/eigen_types.h"
#include "drake/common/trajectories/qp_spline/spline_generation.h"

namespace drake {
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
      s_(Eigen::VectorXd::LinSpaced(nT, 0, 1)) {
  theta_ = NewContinuousVariables(nT_, "theta");
  // θ = ṡ², so θ >= 0.
  AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), theta_);

  int num_facets = contact_facets_.size();
  lambda_.reserve(num_facets);
  for (int i = 0; i < num_facets; ++i) {
    lambda_.push_back(
        NewContinuousVariables(contact_facets_[i].NumVertices() *
                                   contact_facets_[i].NumFrictionConeEdges(),
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

  // Add the time interval upper bound variable
  AddTimeIntervalBoundVariables();
}

void MultiContactTimeOptimalPlanner::AddTimeIntervalBoundVariables() {
  t_bar_ = NewContinuousVariables(nT_ - 1, "t_bar");
  solvers::VectorXDecisionVariable z = NewContinuousVariables(nT_, "z");
  // z is the slack variable here, θ(i) >= z(i)²
  for (int i = 0; i < nT_; ++i) {
    Vector3<symbolic::Expression> expr;
    expr << theta_(i), 1, z(i);
    AddRotatedLorentzConeConstraint(expr);
  }
  for (int i = 0; i < nT_ - 1; ++i) {
    Vector3<symbolic::Expression> expr1;
    expr1 << t_bar_(i), (z(i) + z(i + 1)), std::sqrt(2 * (s_(i + 1) - s_(i)));
    AddRotatedLorentzConeConstraint(expr1);
  }
}

Eigen::Matrix<symbolic::Expression, 3, 1>
MultiContactTimeOptimalPlanner::VecAccel(
    int i, const Eigen::Ref<const Eigen::Vector3d>& x_prime,
    const Eigen::Ref<const Eigen::Vector3d>& x_double_prime) const {
  return x_double_prime * theta_(i) + x_prime * s_ddot(i);
}

std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>
MultiContactTimeOptimalPlanner::ComPathPrime(
    const Eigen::Ref<const Eigen::Matrix3Xd>& com_path) const {
  // We assume that the CoM path is a cubic spline with continuous first and
  // second derivatives. We further assume that the first and second derivatives
  // at the final point are both 0.
  Eigen::Matrix3Xd r_prime, r_double_prime;
  r_prime.resize(3, nT_);
  r_double_prime.resize(3, nT_);
  std::vector<int> segment_polynomial_orders(nT_ - 1, 3);
  std::vector<double> breaks(s_.data(), s_.data() + nT_);
  for (int row = 0; row < 3; ++row) {
    SplineInformation spline_info(segment_polynomial_orders, breaks);
    spline_info.addValueConstraint(
        nT_ - 2, ValueConstraint(0, spline_info.getEndTime(nT_ - 2),
                                 com_path(row, nT_ - 1)));
    for (int i = 0; i < nT_ - 1; ++i) {
      spline_info.addValueConstraint(
          i, ValueConstraint(0, spline_info.getStartTime(i), com_path(row, i)));
    }
    for (int i = 0; i < nT_ - 2; ++i) {
      for (int derivative_order = 0; derivative_order < 3; ++derivative_order) {
        spline_info.addContinuityConstraint(
            ContinuityConstraint(derivative_order, i, i + 1));
      }
    }
    spline_info.addValueConstraint(
        nT_ - 2, ValueConstraint(1, spline_info.getEndTime(nT_ - 2), 0));
    spline_info.addValueConstraint(
        nT_ - 2, ValueConstraint(2, spline_info.getEndTime(nT_ - 2), 0));
    auto spline_traj = generateSpline(spline_info);
    auto spline_deriv = spline_traj.derivative(1);
    auto spline_dderiv = spline_deriv.derivative(1);
    for (int i = 0; i < nT_; ++i) {
      r_prime(row, i) = spline_deriv.value(s_(i))(0);
      r_double_prime(row, i) = spline_dderiv.value(s_(i))(0);
    }
  }
  return std::make_pair(r_prime, r_double_prime);
}

std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd>
MultiContactTimeOptimalPlanner::AngularPathPrime(
    const std::vector<Eigen::Matrix3d>& orient_path) const {
  // This is a naive hack. We compute the average angular velocity in each
  // segment, and treat it as the mean of the angular velocity in each segment.
  Eigen::Matrix3Xd omega_bar(3, nT_);
  Eigen::Matrix3Xd omega_bar_prime(3, nT_);
  Eigen::Matrix3Xd omega_bar_average(3, nT_ - 1);
  for (int i = 0; i < nT_ - 1; ++i) {
    Eigen::AngleAxisd angle_diff(orient_path[i].transpose() *
                                 orient_path[i + 1]);
    omega_bar_average.col(i) =
        angle_diff.axis() * angle_diff.angle() / (s_(i + 1) - s_(i));
  }

  // Assume that the final angular velocity is zero
  omega_bar.col(nT_ - 1) = Eigen::Vector3d::Zero();
  for (int i = nT_ - 2; i >= 0; --i) {
    omega_bar.col(i) = 2 * omega_bar_average.col(i) - omega_bar.col(i + 1);
  }

  // Assume that the starting and ending accelerations are 0.
  omega_bar_prime.col(0) = Eigen::Vector3d::Zero();
  omega_bar_prime.col(nT_ - 1) = Eigen::Vector3d::Zero();
  for (int i = 1; i < nT_ - 1; ++i) {
    omega_bar_prime.col(i) =
        (omega_bar_average.col(i) - omega_bar_average.col(i - 1)) /
        ((s_(i + 1) - s_(i)) / 2);
  }
  return std::make_pair(omega_bar, omega_bar_prime);
}

Eigen::Matrix<symbolic::Expression, 6, 1>
MultiContactTimeOptimalPlanner::ContactFacetWrench(int facet_index,
                                                   int time_index) const {
  const std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>>
      wrench_cone_edges = contact_facets_[facet_index].CalcWrenchConeEdges();
  Eigen::Matrix<symbolic::Expression, 6, 1> facet_wrench;
  facet_wrench << 0, 0, 0, 0, 0, 0;
  for (int i = 0; i < contact_facets_[facet_index].NumVertices(); ++i) {
    facet_wrench +=
        wrench_cone_edges[i] *
        lambda_[facet_index].block(
            contact_facets_[facet_index].NumFrictionConeEdges() * i, time_index,
            contact_facets_[facet_index].NumFrictionConeEdges(), 1);
  }
  return facet_wrench;
}

void MultiContactTimeOptimalPlanner::AddTimeIntervalLowerBound(
    int interval_index, double dt_lower_bound) {
  if (interval_index < 0 || interval_index >= nT_ - 1) {
    throw std::runtime_error("interval_index is out of range.");
  }
  if (dt_lower_bound < 0) {
    throw std::runtime_error("dt_lower_bound is out of range.");
  }
  // The time interval is computed as
  // dt = 2 * (s(i+1) - s(i)) / (sqrt(θ(i+1)) + sqrt(θ(i))
  // The constraint dt >= lower_bound is a non-convex constraint on
  // θ. We thus impose a convex constraint as a sufficient condition
  // θ(i) <= (s(i+1)-s(i)/lower_bound)²
  // θ(i+1) <= (s(i+1)-s(i)/lower_bound)²
  double theta_upper_bound = std::pow(
      (s_(interval_index + 1) - s_(interval_index)) / dt_lower_bound, 2);
  AddBoundingBoxConstraint(0, theta_upper_bound, theta_.block<2, 1>(interval_index, 0));
}

void MultiContactTimeOptimalPlanner::SetObjectPoseSequence(
    const std::vector<Eigen::Isometry3d>& object_pose) {
  Eigen::Matrix3Xd r_prime, r_double_prime, omega_bar, omega_bar_prime;
  Eigen::Matrix3Xd com_path(3, nT_);
  std::vector<Eigen::Matrix3d> orient_path(nT_);
  for (int i = 0; i < nT_; ++i) {
    com_path.col(i) = object_pose[i].translation();
    orient_path[i] = object_pose[i].linear();
  }
  std::tie(r_prime, r_double_prime) = ComPathPrime(com_path);
  std::tie(omega_bar, omega_bar_prime) = AngularPathPrime(orient_path);

  // Now add the Newton's law as constraint
  for (int i = 0; i < nT_; ++i) {
    Eigen::Matrix<symbolic::Expression, 6, 1> dynamics_constraint_lhs,
        dynamics_constraint_rhs;
    dynamics_constraint_lhs.topRows<3>() =
        m_ * VecAccel(i, r_prime.col(i), r_double_prime.col(i));
    dynamics_constraint_rhs.bottomRows<3>() =
        I_B_ * VecAccel(i, omega_bar.col(i), omega_bar_prime.col(i)) +
        omega_bar.col(i).cross(I_B_ * omega_bar.col(i)) * theta_(i);
    dynamics_constraint_rhs << 0, 0, 0, 0, 0, 0;
    dynamics_constraint_rhs.topRows<3>() = m_ * gravity_;
    for (int facet_index = 0;
         facet_index < static_cast<int>(contact_facets_.size());
         ++facet_index) {
      const Eigen::Matrix<symbolic::Expression, 6, 1> contact_facet_wrench =
          ContactFacetWrench(facet_index, i);
      dynamics_constraint_rhs.topRows<3>() +=
          orient_path[i] * contact_facet_wrench.topRows<3>();
      dynamics_constraint_rhs.bottomRows<3>() +=
          contact_facet_wrench.bottomRows<3>();
    }
    AddLinearConstraint(dynamics_constraint_lhs == dynamics_constraint_rhs);
  }
}
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
