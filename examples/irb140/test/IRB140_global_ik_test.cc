#include <gtest/gtest.h>

#include "drake/examples/irb140/IRB140_analytical_kinematics.h"
#include "drake/examples/irb140/test/irb140_common.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace examples {
namespace IRB140 {
class GlobalIKTest : public ::testing::Test {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GlobalIKTest)

  GlobalIKTest()
      : analytical_ik_(),
        robot_(analytical_ik_.robot()),
        global_ik_(*robot_, multibody::GlobalInverseKinematics::Options()),
        ee_idx_(analytical_ik_.robot()->FindBodyIndex("link_6")),
        global_ik_pos_cnstr_(global_ik_.AddWorldPositionConstraint(
            ee_idx_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero())),
        global_ik_orient_cnstr_(global_ik_.AddBoundingBoxConstraint(
            0, 0, global_ik_.body_rotation_matrix(ee_idx_))) {}

  void SetEndEffectorPose(const Eigen::Isometry3d& ee_pose) {
    global_ik_pos_cnstr_.evaluator()->UpdateLowerBound(ee_pose.translation());
    global_ik_pos_cnstr_.evaluator()->UpdateUpperBound(ee_pose.translation());
    Eigen::Matrix<double, 9, 1> rotmat_flat;
    rotmat_flat << ee_pose.linear().col(0), ee_pose.linear().col(1),
        ee_pose.linear().col(2);
    global_ik_orient_cnstr_.evaluator()->UpdateLowerBound(rotmat_flat);
    global_ik_orient_cnstr_.evaluator()->UpdateUpperBound(rotmat_flat);
  }

  void SolveIK(const Eigen::Isometry3d& ee_pose,
               solvers::SolutionResult global_ik_status_expected) {
    q_analytical_ik_ = analytical_ik_.inverse_kinematics(ee_pose);
    SetEndEffectorPose(ee_pose);
    solvers::GurobiSolver gurobi_solver;
    global_ik_.SetSolverOption(gurobi_solver.id(), "FeasibilityTol",
                               7E-6);
    for (int i = 1; i < robot_->get_num_bodies(); ++i) {
      const auto& body_R = global_ik_.body_rotation_matrix(i);
      Eigen::Matrix<symbolic::Expression, 5, 1> cone_expr;
      cone_expr(0) = 1.0;
      cone_expr(1) = 3.0;
      cone_expr.tail<3>() = body_R.col(0) + body_R.col(1) + body_R.col(2);
      global_ik_.AddRotatedLorentzConeConstraint(cone_expr);
      cone_expr.tail<3>() = body_R.col(0) + body_R.col(1) - body_R.col(2);
      global_ik_.AddRotatedLorentzConeConstraint(cone_expr);
      cone_expr.tail<3>() = body_R.col(0) - body_R.col(1) + body_R.col(2);
      global_ik_.AddRotatedLorentzConeConstraint(cone_expr);
      cone_expr.tail<3>() = body_R.col(0) - body_R.col(1) - body_R.col(2);
      global_ik_.AddRotatedLorentzConeConstraint(cone_expr);
      cone_expr.tail<3>() = body_R.row(0) + body_R.row(1) + body_R.row(2);
      global_ik_.AddRotatedLorentzConeConstraint(cone_expr);
      cone_expr.tail<3>() = body_R.row(0) + body_R.row(1) - body_R.row(2);
      global_ik_.AddRotatedLorentzConeConstraint(cone_expr);
      cone_expr.tail<3>() = body_R.row(0) - body_R.row(1) + body_R.row(2);
      global_ik_.AddRotatedLorentzConeConstraint(cone_expr);
      cone_expr.tail<3>() = body_R.row(0) - body_R.row(1) - body_R.row(2);
      global_ik_.AddRotatedLorentzConeConstraint(cone_expr);
    }
    solvers::SolutionResult global_ik_status = gurobi_solver.Solve(global_ik_);
    switch (global_ik_status_expected) {
      case solvers::SolutionResult::kSolutionFound:
        EXPECT_FALSE(q_analytical_ik_.empty());
        EXPECT_EQ(global_ik_status, solvers::SolutionResult::kSolutionFound);
        q_global_ik_ = global_ik_.ReconstructGeneralizedPositionSolution();
        break;
      case solvers::SolutionResult::kInfeasibleConstraints:
        EXPECT_TRUE(q_analytical_ik_.empty());
        EXPECT_TRUE(global_ik_status ==
                        solvers::SolutionResult::kInfeasibleConstraints ||
                    global_ik_status ==
                        solvers::SolutionResult::kInfeasible_Or_Unbounded);
        break;
      default:
        throw std::runtime_error("Unknown status.");
    }
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
  ee_pose.translation() << 0, -0.5, -0.1;
  SolveIK(ee_pose, solvers::SolutionResult::kSolutionFound);

  ee_pose.translation() << 0, -0.5, 0.1;
  SolveIK(ee_pose, solvers::SolutionResult::kSolutionFound);

  ee_pose.translation() << 0.5, 0.2, 0.3;
  SolveIK(ee_pose, solvers::SolutionResult::kSolutionFound);

  ee_pose.translation() << 0, -0.2, 0.4;
  SolveIK(ee_pose, solvers::SolutionResult::kSolutionFound);

  ee_pose.linear()
      << Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
  ee_pose.translation() << 0.3, 0.1, 0.4;
  SolveIK(ee_pose, solvers::SolutionResult::kSolutionFound);
}

TEST_F(GlobalIKTest, InfeasibleSinglePoseTest) {
  Eigen::Isometry3d ee_pose;
  ee_pose.linear() = Eigen::Matrix3d::Identity();
  ee_pose.translation() << 0.8, -0.3, 0.2;
  SolveIK(ee_pose, solvers::SolutionResult::kInfeasibleConstraints);

  ee_pose.translation() << 0, 0, 0.05;
  SolveIK(ee_pose, solvers::SolutionResult::kInfeasibleConstraints);

  ee_pose.linear()
      << Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
  ee_pose.translation() << 0, -0.5, 0.4;
  SolveIK(ee_pose, solvers::SolutionResult::kInfeasibleConstraints);
}
}
}
}
