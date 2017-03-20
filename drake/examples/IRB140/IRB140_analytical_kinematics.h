#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "drake/common/drake_copyable.h"

namespace drake {
namespace examples {
namespace IRB140 {
/**
 * This class uses the formulation in [1] to compute the forward
 * and inverse kinematics of ABB 1RB 140 robot in the analytical form.
 * [1] Introduction to Robotics by J. J. Craig, 2003
 */
class IRB140AnalyticalKinematics {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IRB140AnalyticalKinematics)

  IRB140AnalyticalKinematics();

  // The pose of link 1 measured and expressed in link 0.
  Eigen::Isometry3d X_01(double theta) const;

  // The pose of link 2 measured and expressed in link 1.
  Eigen::Isometry3d X_12(double theta) const;

  // The pose of link 3 measured and expressed in link 2.
  Eigen::Isometry3d X_23(double theta) const;

  // The pose of link 4 measured and expressed in link 3.
  Eigen::Isometry3d X_34(double theta) const;

  // The pose of link 5 measured and expressed in link 4.
  Eigen::Isometry3d X_45(double theta) const;

  // The pose of link 6 measured and expressed in link 5.
  Eigen::Isometry3d X_56(double theta) const;

 private:
  const double l0_;  // offset of joint 1 in base link in the z direction.
  const double l1_x_;  // offset of joint 2 in link 1 in the x direction.
  const double l1_y_;  // offset of joint 2 in link 1 in the -y direction.
  const double l2_;  // offset of joint 3 in link 2 in the -y direction.
  const double l3_;  // offset of joint 4 in link 3 in the x direction.
  const double l4_;  // offset of joint 5 in link 4 in the x direction.
  const double l5_;  // offset of joint 6 in link 5 in the x direction.
};
}  // namespace IRB140
}  // namespace examples
}  // namespace drake
