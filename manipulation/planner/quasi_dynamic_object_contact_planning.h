#pragma once

#include "drake/manipulation/planner/object_contact_planning.h"

namespace drake {
namespace manipulation {
namespace planner {
class QuasiDynamicObjectContactPlanning : public ObjectContactPlanning {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuasiDynamicObjectContactPlanning)

  QuasiDynamicObjectContactPlanning(
      int nT, double dt, double mass,
      const Eigen::Ref<const Eigen::Matrix3d>& I_B,
      const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV, int num_pushers,
      const std::vector<BodyContactPoint>& Q,
      bool add_second_order_cone_for_R = false);

  ~QuasiDynamicObjectContactPlanning() = default;

  /** Getter for the object linear velocity, measured and expressed in the
   * object instantaneous body frame B. */
  solvers::MatrixDecisionVariable<3, Eigen::Dynamic> v_B() const {
    return v_B_;
  }

  /** Getter for the object angular velocity, measured and expressed in the
   * object instantaneous body frame B. */
  solvers::MatrixDecisionVariable<3, Eigen::Dynamic> omega_B() const {
    return omega_B_;
  }

 private:
  // Add the interpolation constraint
  void AddInterpolationConstraint();

  // Add the translation interpolation constraint
  // p_WB[knot+1] - p_WB[knot] = 0.5 * (R_WB[knot] * v_B_.col(knot) + R_WB[knot
  // + 1] * v_B_.col[knot + 1]) * dt
  void AddTranslationInterpolationConstraint();

  // Add the orientation interpolation constraint
  // R_WB[knot+1] - R_WB[knot] = 0.5 * (R_WB[knot] + R_WB[knot+1]) *
  // SkewSymmetric((omega_B_.col(knot) + omega_B_.col(knot + 1))  * dt / 2)
  void AddOrientationInterpolationConstraint();

  double dt_;
  Eigen::Matrix3d I_B_;
  solvers::MatrixDecisionVariable<3, Eigen::Dynamic> v_B_;
  solvers::MatrixDecisionVariable<3, Eigen::Dynamic> omega_B_;
  // phi_v_B_ are the ends of the intervals for v_B_(i, j).
  Eigen::Matrix<double, 5, 1> phi_v_B_;
  // phi_omega_B_ are the ends of the intervals for omega_B_(i, j).
  Eigen::Matrix<double, 5, 1> phi_omega_B_;
  // b_v_B[i][j] are the binary variables, indicating which interval v_B_(i, j)
  // is in.
  std::array<std::vector<solvers::VectorXDecisionVariable>, 3> b_v_B_;
  // b_omega_average[i][j] are the binary variables, indicating which interval
  // (omega_B_(i, j) + omega_B_(i, j+1)) / 2 is in.
  std::array<std::vector<solvers::VectorXDecisionVariable>, 3> b_omega_average_;
};
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
