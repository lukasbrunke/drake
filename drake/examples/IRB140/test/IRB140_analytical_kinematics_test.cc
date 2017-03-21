#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"

#include <gtest/gtest.h>

#include "drake/common/drake_path.h"
#include "drake/common/eigen_matrix_compare.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_tree_construction.h"

using Eigen::Isometry3d;

namespace drake {
namespace examples {
namespace IRB140 {
namespace {
std::unique_ptr<RigidBodyTreed> ConstructIRB140() {
  std::unique_ptr<RigidBodyTreed> rigid_body_tree = std::make_unique<RigidBodyTreed>();
  const std::string model_path = drake::GetDrakePath() + "/examples/IRB140/urdf/irb_140.urdf";

  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kFixed,
      nullptr,
      rigid_body_tree.get());

  multibody::AddFlatTerrainToWorld(rigid_body_tree.get());
  return rigid_body_tree;
}

void CompareIsometry3d(const Eigen::Isometry3d& X1, const Eigen::Isometry3d& X2, double tol = 1E-10) {
  EXPECT_TRUE(CompareMatrices(X1.linear(), X2.linear(), tol, MatrixCompareType::absolute));
  EXPECT_TRUE(CompareMatrices(X1.translation(), X2.translation(), tol, MatrixCompareType::absolute));
}

class IRB140Test : public ::testing::Test {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IRB140Test)

  IRB140Test()
      : rigid_body_tree_(ConstructIRB140()),
        analytical_kinematics() {}

  ~IRB140Test() override{}

 protected:
  std::unique_ptr<RigidBodyTreed> rigid_body_tree_;
  IRB140AnalyticalKinematics analytical_kinematics;
};

TEST_F(IRB140Test, link_forward_kinematics) {
  const auto& joint_lb = rigid_body_tree_->joint_limit_min;
  const auto& joint_ub = rigid_body_tree_->joint_limit_max;
  Eigen::Matrix<double, 6, 1> q = joint_lb;
  for (int i = 0; i < 6; ++i) {
    q(i) += (joint_ub(i) - joint_lb(i)) * i / 10.0;
  }

  auto cache = rigid_body_tree_->CreateKinematicsCache();
  cache.initialize(q);
  rigid_body_tree_->doKinematics(cache);

  std::array<Isometry3d, 7> X_WB;  // The pose of body frame `B` in the world frame `W`.
  X_WB[0].linear() = Eigen::Matrix3d::Identity();
  X_WB[0].translation() = Eigen::Vector3d::Zero();
  for (int i = 1; i < 7; ++i) {
    X_WB[i] = rigid_body_tree_->CalcBodyPoseInWorldFrame(cache, *(rigid_body_tree_->FindBody("link_" + std::to_string(i))));
  }

  // X_PC[i] is the pose of child body frame `C` (body[i+1]) in the parent body
  // frame `P` (body[i])
  std::array<Isometry3d, 6> X_PC;
  X_PC[0] = X_WB[1];
  for (int i = 1; i < 6; ++i) {
    X_PC[i].linear() = X_WB[i].linear().transpose() * X_WB[i + 1].linear();
    X_PC[i].translation() = X_WB[i].linear().transpose() * (X_WB[i+1].translation() - X_WB[i].translation());
  }

  const auto X_01 = analytical_kinematics.X_01(q(0));
  CompareIsometry3d(X_PC[0], X_01, 1E-5);

  const auto X_12 = analytical_kinematics.X_12(q(1));
  CompareIsometry3d(X_PC[1], X_12, 1e-5);

  const auto X_23 = analytical_kinematics.X_23(q(2));
  CompareIsometry3d(X_PC[2], X_23, 1e-5);

  const auto X_34 = analytical_kinematics.X_34(q(3));
  CompareIsometry3d(X_PC[3], X_34, 1E-5);

  const auto X_45 = analytical_kinematics.X_45(q(4));
  CompareIsometry3d(X_PC[4], X_45, 1E-5);

  const auto X_56 = analytical_kinematics.X_56(q(5));
  CompareIsometry3d(X_PC[5], X_56, 1E-5);
}
}  // namespace
}  // namespace IRB140
}  // namespace examples
}  // namespace drake