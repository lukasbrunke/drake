#include <gtest/gtest.h>

#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"
#include "drake/examples/IRB140/test/irb140_common.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace examples {
namespace IRB140 {
class GlobalIKTest : public ::testing::Test {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GlobalIKTest)

  GlobalIKTest() :
      analytical_ik_(),
               robot_(analytical_ik_.robot()),
      global_ik_(*robot_, 2),
      ee_idx_(analytical_ik_.robot()->FindBodyIndex("link_6")),
      global_ik_pos_cnstr_(global_ik_.AddWorldPositionConstraint(ee_idx_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())),
      global_ik_orient_cnstr_(global_ik_.AddBoundingBoxConstraint(0, 0, global_ik_.body_rotation_matrix(ee_idx_))) {}

  void SetEndEffectorPose(const Eigen::Isometry3d& ee_pose) {
    global_ik_pos_cnstr_.constraint()->UpdateLowerBound(ee_pose.translation());
    global_ik_pos_cnstr_.constraint()->UpdateUpperBound(ee_pose.translation());
    Eigen::Matrix<double, 9, 1> rotmat_flat;
    rotmat_flat << ee_pose.linear().col(0), ee_pose.linear().col(1), ee_pose.linear().col(2);
    global_ik_orient_cnstr_.constraint()->UpdateLowerBound(rotmat_flat);
    global_ik_orient_cnstr_.constraint()->UpdateUpperBound(rotmat_flat);
  }

  void SolveIK(const Eigen::Isometry3d& ee_pose) {
    q_analytical_ik_ = analytical_ik_.inverse_kinematics(ee_pose);
    SetEndEffectorPose(ee_pose);
    solvers::GurobiSolver gurobi_solver;
    solvers::SolutionResult global_ik_status = gurobi_solver.Solve(global_ik_);
    EXPECT_FALSE(q_analytical_ik_.empty());
    EXPECT_EQ(global_ik_status, solvers::SolutionResult::kSolutionFound);
    q_global_ik_ = global_ik_.ReconstructGeneralizedPositionSolution();
  }

 private:
  IRB140AnalyticalKinematics analytical_ik_;
  RigidBodyTreed* robot_;
  multibody::GlobalInverseKinematics global_ik_;
  int ee_idx_;
  solvers::Binding<solvers::LinearConstraint> global_ik_pos_cnstr_;
  solvers::Binding<solvers::BoundingBoxConstraint> global_ik_orient_cnstr_;
  std::vector<Eigen::Matrix<double, 6, 1>> q_analytical_ik_;
  Eigen::Matrix<double, 6, 1> q_global_ik_;
};

TEST_F(GlobalIKTest, SinglePoseTest) {
  Eigen::Isometry3d ee_pose;
  ee_pose.linear() = Eigen::Matrix3d::Identity();
  ee_pose.translation() << 0.5, 0.2, 0.3;
  SolveIK(ee_pose);

  ee_pose.translation() << 0, -0.2, 0.4;
  SolveIK(ee_pose);

  ee_pose.linear() << Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
  ee_pose.translation() << 0.3, 0.1, 0.4;
  SolveIK(ee_pose);
}
}
}
}