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

  std::array<Isometry3d, 6> X_WB;  // The pose of body frame `B` in the world frame `W`.
  for (int i = 0; i < 6; ++i) {
    X_WB[i] = rigid_body_tree_->CalcBodyPoseInWorldFrame(cache, *(rigid_body_tree_->FindBody("link_" + std::to_string(i + 1))));
  }


  std::array<Isometry3d, 5> X_PC;  // The pose of child body frame `C` in the parent body frame `P`.
  for (int i = 0; i < 5; ++i) {
    X_PC[i].linear() = X_WB[i].linear().transpose() * X_WB[i + 1].linear();
    X_PC[i].translation() = X_WB[i].linear().transpose() * (X_PC[i+1].translation() - X_PC[i].translation());
  }

  const auto X_01 = analytical_kinematics.X_01(q(0));
  EXPECT_TRUE(CompareMatrices(X_PC[0].linear(), X_01.linear()));
  EXPECT_TRUE(CompareMatrices(X_PC[0].translation(), X_01.translation()));
}
}  // namespace
}  // namespace IRB140
}  // namespace examples
}  // namespace drake