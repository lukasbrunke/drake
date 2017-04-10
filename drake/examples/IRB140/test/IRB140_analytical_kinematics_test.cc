#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_matrix_compare.h"

using Eigen::Isometry3d;

namespace drake {
namespace examples {
namespace IRB140 {
namespace {

bool CompareIsometry3d(const Eigen::Isometry3d& X1, const Eigen::Isometry3d& X2, double tol = 1E-10) {
  bool orientation_match = CompareMatrices(X1.linear(), X2.linear(), tol, MatrixCompareType::absolute);
  EXPECT_TRUE(CompareMatrices(X1.linear(), X2.linear(), tol, MatrixCompareType::absolute));
  bool position_match = CompareMatrices(X1.translation(), X2.translation(), tol, MatrixCompareType::absolute);
  EXPECT_TRUE(CompareMatrices(X1.translation(), X2.translation(), tol, MatrixCompareType::absolute));
  return position_match && orientation_match;
}

class IRB140Test : public ::testing::Test {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IRB140Test)

  IRB140Test() : analytical_kinematics() {}

  ~IRB140Test() override{}

 protected:
  IRB140AnalyticalKinematics analytical_kinematics;
};

void printPose(const Eigen::Matrix<symbolic::Expression, 4, 4>& pose) {
  std::cout <<"R\n";
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      std::cout << pose(i, j) << std::endl;
    }
  }
  std::cout <<"p\n";
  for (int i = 0; i < 3; ++i) {
    std::cout << pose(i, 3) << std::endl;
  }
}

void TestForwardKinematics(const IRB140AnalyticalKinematics& analytical_kinematics, const Eigen::Matrix<double, 6, 1>& q) {
  auto cache = analytical_kinematics.robot()->CreateKinematicsCache();
  cache.initialize(q);
  analytical_kinematics.robot()->doKinematics(cache);

  std::array<Isometry3d, 7> X_WB;  // The pose of body frame `B` in the world frame `W`.
  X_WB[0].linear() = Eigen::Matrix3d::Identity();
  X_WB[0].translation() = Eigen::Vector3d::Zero();
  for (int i = 1; i < 7; ++i) {
    X_WB[i] = analytical_kinematics.robot()->CalcBodyPoseInWorldFrame(cache, *(analytical_kinematics.robot()->FindBody("link_" + std::to_string(i))));
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

  const auto X_13 = analytical_kinematics.X_13(q(1), q(2));
  CompareIsometry3d(X_PC[1] * X_PC[2], X_13, 1e-5);

  const auto X_34 = analytical_kinematics.X_34(q(3));
  CompareIsometry3d(X_PC[3], X_34, 1E-5);

  const auto X_45 = analytical_kinematics.X_45(q(4));
  CompareIsometry3d(X_PC[4], X_45, 1E-5);

  const auto X_56 = analytical_kinematics.X_56(q(5));
  CompareIsometry3d(X_PC[5], X_56, 1E-5);

  const auto X_06 = analytical_kinematics.X_06(q);
  CompareIsometry3d(X_06, X_WB[6], 1E-5);
}

TEST_F(IRB140Test, link_forward_kinematics) {
  const auto X_01_sym = analytical_kinematics.X_01();
  const auto X_12_sym = analytical_kinematics.X_12();
  const auto X_23_sym = analytical_kinematics.X_23();
  const auto X_13_sym = analytical_kinematics.X_13();
  const auto X_34_sym = analytical_kinematics.X_34();
  const auto X_45_sym = analytical_kinematics.X_45();
  const auto X_56_sym = analytical_kinematics.X_56();

  const auto X_16_sym = X_13_sym * X_34_sym * X_45_sym * X_56_sym;
  const auto X_06_sym  = X_01_sym * X_16_sym;
  std::cout <<"X_06\n";
  printPose(X_06_sym);

  const int num_joint_sample = 3;
  Eigen::Matrix<double, 6, num_joint_sample> q_sample;
  for (int i = 0; i < 6; ++i) {
    q_sample.row(i) = Eigen::Matrix<double, 1, num_joint_sample>::LinSpaced(analytical_kinematics.robot()->joint_limit_min(i) + 1E-4, analytical_kinematics.robot()->joint_limit_max(i) - 1E-4);
  }
  auto cache = analytical_kinematics.robot()->CreateKinematicsCache();

  for (int i0 = 0; i0 < num_joint_sample; ++i0) {
    for (int i1 = 0; i1 < num_joint_sample; ++i1) {
      for (int i2 = 0; i2 < num_joint_sample; ++i2) {
        for (int i3 = 0; i3 < num_joint_sample; ++i3) {
          for (int i4 = 0; i4 < num_joint_sample; ++i4) {
            for (int i5 = 0; i5 < num_joint_sample; ++i5) {
              Eigen::Matrix<double, 6, 1> q;
              q(0) = q_sample(0, i0);
              q(1) = q_sample(1, i1);
              q(2) = q_sample(2, i2);
              q(3) = q_sample(3, i3);
              q(4) = q_sample(4, i4);
              q(5) = q_sample(5, i5);
              TestForwardKinematics(analytical_kinematics, q);
            }
          }
        }
      }
    }
  }
}

void TestInverseKinematics(const IRB140AnalyticalKinematics& analytical_kinematics, const Eigen::Matrix<double, 6, 1>& q) {
  const Eigen::Matrix<double, 6, 1>
      q_lb = analytical_kinematics.robot()->joint_limit_min;
  const Eigen::Matrix<double, 6, 1>
      q_ub = analytical_kinematics.robot()->joint_limit_max;
  auto cache = analytical_kinematics.robot()->CreateKinematicsCache();
  cache.initialize(q);
  analytical_kinematics.robot()->doKinematics(cache);
  const Isometry3d link6_pose =
      analytical_kinematics.robot()->CalcBodyPoseInWorldFrame(cache,
                                                              *(analytical_kinematics.robot()->FindBody(
                                                                  "link_6")));

  const auto &q_all = analytical_kinematics.inverse_kinematics(link6_pose);
  EXPECT_GE(q_all.size(), 1);
  if (q_all.size() == 0) {
    std::cout << "q\n" << q << std::endl;
    const auto &X_06 = analytical_kinematics.X_06(q);
    CompareIsometry3d(X_06, link6_pose, 1e-5);
    analytical_kinematics.inverse_kinematics(link6_pose);
  }

  for (const auto &q_ik : q_all) {
    EXPECT_TRUE((q_ik.array() >= q_lb.array()).all());
    EXPECT_TRUE((q_ik.array() <= q_ub.array()).all());
    if (!(q_ik.array() >= q_lb.array()).all()) {
      std::cout << q_ik - q_lb << std::endl;
    }
    if (!(q_ik.array() <= q_ub.array()).all()) {
      std::cout << q_ub - q_ik << std::endl;
    }
    cache.initialize(q_ik);
    analytical_kinematics.robot()->doKinematics(cache);
    const Isometry3d link6_pose_ik =
        analytical_kinematics.robot()->CalcBodyPoseInWorldFrame(cache,
                                                                *(analytical_kinematics.robot()->FindBody(
                                                                    "link_6")));
    CompareIsometry3d(link6_pose_ik, link6_pose, 1E-5);
    if (!CompareIsometry3d(link6_pose_ik, link6_pose, 1E-5)) {
      std::cout << "q\n" << q << std::endl;
      std::cout << "q_ik\n" << q_ik << std::endl;
      analytical_kinematics.inverse_kinematics(link6_pose);
    }
  }
}
/*
TEST_F(IRB140Test, inverse_kinematics_test) {
  std::vector<Eigen::Matrix<double, 6, 1>> q_all;
  const int num_joint_sample = 10;
  Eigen::Matrix<double, 6, num_joint_sample> q_sample;
  for (int i = 0; i < 6; ++i) {
    q_sample.row(i) = Eigen::Matrix<double, 1, num_joint_sample>::LinSpaced(
        analytical_kinematics.robot()->joint_limit_min(i) + 1E-4,
        analytical_kinematics.robot()->joint_limit_max(i) - 1E-4);
  }

  for (int i0 = 0; i0 < num_joint_sample; ++i0) {
    for (int i1 = 0; i1 < num_joint_sample; ++i1) {
      for (int i2 = 0; i2 < num_joint_sample; ++i2) {
        for (int i3 = 0; i3 < num_joint_sample; ++i3) {
          for (int i4 = 0; i4 < num_joint_sample; ++i4) {
            for (int i5 = 0; i5 < num_joint_sample; ++i5) {
              Eigen::Matrix<double, 6, 1> q;
              q(0) = q_sample(0, i0);
              q(1) = q_sample(1, i1);
              q(2) = q_sample(2, i2);
              q(3) = q_sample(3, i3);
              q(4) = q_sample(4, i4);
              q(5) = q_sample(5, i5);

              TestInverseKinematics(analytical_kinematics, q);
            }
          }
        }
      }
    }
  }
}*/


TEST_F(IRB140Test, inverse_kinematics_corner_test) {
  // Degenerate case q = 0
  Eigen::Matrix<double, 6, 1> q;
  q.setZero();
  TestInverseKinematics(analytical_kinematics, q);
}
}  // namespace
}  // namespace IRB140
}  // namespace examples
}  // namespace drake