#include "drake/multibody/global_inverse_kinematics.h"

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
class GlobalInverseKinematicsTester {
 public:
  GlobalInverseKinematicsTester() {}

  static double ReconstructJointAngleForRevoluteJoint(
      const Eigen::Matrix3d& R_WP, const Eigen::Matrix3d& R_WB,
      const Eigen::Vector3d& a_B, const Eigen::Vector3d& p_WB,
      const Eigen::Matrix3Xd& p_WC, const Eigen::Matrix3Xd& p_BC, double beta,
      double angle_lower, double angle_upper) {
    return GlobalInverseKinematics::ReconstructJointAngleForRevoluteJoint(
        R_WP, R_WB, a_B, p_WB, p_WC, p_BC, beta, angle_lower, angle_upper);
  }
};

void CheckReconstructJointAngleForRevoluteJoint(
    const Eigen::Matrix3d& R_WP, const Eigen::Matrix3d& R_WB,
    const Eigen::Vector3d& a_B, const Eigen::Vector3d& p_WB,
    const Eigen::Matrix3Xd& p_WC, const Eigen::Matrix3Xd& p_BC, double beta,
    double angle_lower, double angle_upper) {
  const double theta_optimal =
      GlobalInverseKinematicsTester::ReconstructJointAngleForRevoluteJoint(
          R_WP, R_WB, a_B, p_WB, p_WC, p_BC, beta, angle_lower, angle_upper);
  EXPECT_GE(theta_optimal, angle_lower);
  EXPECT_LE(theta_optimal, angle_upper);
  auto Cost = [&](double theta) {
    const Eigen::Matrix3d R_PB =
        Eigen::AngleAxisd(theta, a_B).toRotationMatrix();
    const Eigen::Matrix3d R_err = R_WB - R_WP * R_PB;
    const Eigen::Matrix3Xd pos_err =
        p_WB * Eigen::RowVectorXd::Ones(1, p_WC.cols()) + R_WP * R_PB * p_BC -
        p_WC;
    return (R_err.transpose() * R_err).trace() +
           beta * (pos_err.array() * pos_err.array()).rowwise().sum().sum();
  };
  const double cost_theta_optimal = Cost(theta_optimal);
  const Eigen::VectorXd theta_sample =
      Eigen::VectorXd::LinSpaced(100, angle_lower, angle_upper);
  for (int i = 0; i < theta_sample.size(); ++i) {
    EXPECT_GE(Cost(theta_sample(i)), cost_theta_optimal);
  }
}

GTEST_TEST(GlobalInverseKinematics, ReconstructJointAngleForRevoluteJoint) {
  const Eigen::Matrix3d R_WP =
      Eigen::AngleAxisd(0.1 * M_PI, Eigen::Vector3d(0.2, 0.5, 1.2).normalized())
          .toRotationMatrix();
  const Eigen::Vector3d a_B = Eigen::Vector3d(0.2, -3.1, -.5).normalized();
  Eigen::Matrix3d R_WB =
      R_WP * Eigen::AngleAxisd(-0.2 * M_PI, a_B).toRotationMatrix();
  // Add some small noise to R_WB;
  R_WB += (Eigen::Matrix3d() << 0.1, -0.05, 0.02, 0.05, -0.03, -0.02, -0.01,
           0.03, 0.04)
              .finished();
  const Eigen::Vector3d p_WB(0.1, 1.2, 0.5);
  Eigen::Matrix3Xd p_BC(3, 2);
  p_BC << 0.2, 0.8, 1.2, 0.4, 0.2, -0.3;
  Eigen::Matrix3Xd p_WC = p_WB * Eigen::RowVector2d::Ones() + R_WB * p_BC;
  // Add some small noise to p_WC
  p_WC +=
      (Eigen::Matrix<double, 3, 2>() << 0.03, 0.05, -0.01, -0.03, -0.12, 0.21)
          .finished();
  // First test beta = 0. The optimal is where the gradient vanishes.
  CheckReconstructJointAngleForRevoluteJoint(R_WP, R_WB, a_B, p_WB, p_WC, p_BC,
                                             0, -0.4 * M_PI, 0.2 * M_PI);
  CheckReconstructJointAngleForRevoluteJoint(R_WP, R_WB, a_B, p_WB, p_WC, p_BC,
                                             0, -0.4 * M_PI + 2 * M_PI,
                                             0.2 * M_PI + 2 * M_PI);

  // The optimal is at lower bound.
  CheckReconstructJointAngleForRevoluteJoint(R_WP, R_WB, a_B, p_WB, p_WC, p_BC,
                                             0, 0 * M_PI, 0.2 * M_PI);
  // The optimal is at upper bound.
  CheckReconstructJointAngleForRevoluteJoint(R_WP, R_WB, a_B, p_WB, p_WC, p_BC,
                                             0, -1.2 * M_PI, -0.4 * M_PI);

  // beta is non-zero. The optimal is where the gradient vanishes.
  CheckReconstructJointAngleForRevoluteJoint(R_WP, R_WB, a_B, p_WB, p_WC, p_BC,
                                             0.5, -0.4 * M_PI, 0.2 * M_PI);
  CheckReconstructJointAngleForRevoluteJoint(R_WP, R_WB, a_B, p_WB, p_WC, p_BC,
                                             5, -0.4 * M_PI, 0.2 * M_PI);

  // The optimal is at lower bound.
  CheckReconstructJointAngleForRevoluteJoint(R_WP, R_WB, a_B, p_WB, p_WC, p_BC,
                                             2, 0.2 * M_PI, 0.6 * M_PI);
  // The optimal is at upper bound.
  CheckReconstructJointAngleForRevoluteJoint(R_WP, R_WB, a_B, p_WB, p_WC, p_BC,
                                             2, -1.2 * M_PI, -0.5 * M_PI);
  
  
}
}  // namespace multibody
}  // namespace drake
