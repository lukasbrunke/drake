#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "drake/common/drake_copyable.h"
#include "drake/common/symbolic.h"

#include "drake/multibody/rigid_body_tree.h"

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

  RigidBodyTreed* robot() const {return robot_.get();}

  std::vector<Eigen::Matrix<double, 6, 1>> inverse_kinematics(const Eigen::Isometry3d& link6_pose) const;

  std::vector<double> q1(const Eigen::Isometry3d& link6_pose) const;
  std::vector<double> q2(const Eigen::Isometry3d& link6_pose, double q1) const;
  std::vector<double> q3(const Eigen::Isometry3d& link6_pose, double q1, double q2) const;
  std::vector<Eigen::Vector3d> q456(const Eigen::Isometry3d& link6_pose, double q1, double q2, double q3) const;

  Eigen::Matrix<symbolic::Expression, 4, 4> X_01() const;
  Eigen::Matrix<symbolic::Expression, 4, 4> X_12() const;
  Eigen::Matrix<symbolic::Expression, 4, 4> X_23() const;
  Eigen::Matrix<symbolic::Expression, 4, 4> X_13() const;
  Eigen::Matrix<symbolic::Expression, 4, 4> X_34() const;
  Eigen::Matrix<symbolic::Expression, 4, 4> X_45() const;
  Eigen::Matrix<symbolic::Expression, 4, 4> X_56() const;

  Eigen::Isometry3d X_01(double theta) const;
  Eigen::Isometry3d X_12(double theta) const;
  Eigen::Isometry3d X_23(double theta) const;
  Eigen::Isometry3d X_13(double theta2, double theta3) const;
  Eigen::Isometry3d X_34(double theta) const;
  Eigen::Isometry3d X_45(double theta) const;
  Eigen::Isometry3d X_56(double theta) const;
  Eigen::Isometry3d X_06(const Eigen::Matrix<double, 6, 1>& q) const;

 private:
  std::unique_ptr<RigidBodyTreed> robot_;
  const double l0_;  // offset of joint 1 in base link in the z direction.
  const double l1_x_;  // offset of joint 2 in link 1 in the x direction.
  const double l1_y_;  // offset of joint 2 in link 1 in the -y direction.
  const double l2_;  // offset of joint 3 in link 2 in the -y direction.
  const double l3_;  // offset of joint 4 in link 3 in the x direction.
  const double l4_;  // offset of joint 5 in link 4 in the x direction.

  Eigen::Matrix<symbolic::Variable, 6, 1> c_;  // c_[i] is the cos of i'th joint angle.
  Eigen::Matrix<symbolic::Variable, 6, 1> s_;  // s_[i] is the sin of i'th joint angle.
  symbolic::Variable l0_var_;
  symbolic::Variable l1_x_var_;
  symbolic::Variable l1_y_var_;
  symbolic::Variable l2_var_;
  symbolic::Variable l3_var_;
  symbolic::Variable l4_var_;
  symbolic::Variable c23_var_;
  symbolic::Variable s23_var_;
};
}  // namespace IRB140
}  // namespace examples
}  // namespace drake
