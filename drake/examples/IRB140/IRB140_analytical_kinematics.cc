#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"

namespace drake {
namespace examples {
namespace IRB140 {
IRB140AnalyticalKinematics::IRB140AnalyticalKinematics()
    : l0_(0.1095),
      l1_x_(0.07),
      l1_y_(0.2425),
      l2_(0.36),
      l3_(0.2185),
      l4_(0.1615),
      l5_(0.065) {}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_01(double theta) const {
  Eigen::Isometry3d X;
  Eigen::Matrix3d R_0J = Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d(1, 0, 0)).toRotationMatrix();
  X.linear() = R_0J * Eigen::AngleAxisd(theta, Eigen::Vector3d(0, -1, 0)).toRotationMatrix();
  X.translation() = Eigen::Vector3d(0, 0, l0_);
  return X;
}

Eigen::Isometry3d IRB140AnalyticalKinematics::X_12(double theta) const {
  Eigen::Isometry3d X;
  X.linear() = Eigen::AngleAxisd(theta, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
  X.translation() = Eigen::Vector3d(l1_x_, -l1_y_, 0);
  return X;
}
}
}
}