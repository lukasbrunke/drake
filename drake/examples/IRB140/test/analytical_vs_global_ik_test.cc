#include <iostream>
#include <fstream>
#include <string>

#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/multibody/rigid_body_ik.h"
#include "drake/multibody/constraint/rigid_body_constraint.h"

using Eigen::Isometry3d;

namespace drake {
namespace examples {
namespace IRB140 {
class DUT {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DUT)

  DUT(const Eigen::Quaterniond& ee_orient)
      : analytical_ik_(),
        global_ik_(*(analytical_ik_.robot()), 2),
        ee_idx_(analytical_ik_.robot()->FindBodyIndex("link_6")),
        global_ik_pos_cnstr_(global_ik_.AddWorldPositionConstraint(ee_idx_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())),
        ee_orient_(ee_orient)
  {
    // Fix the end effector body orientation
    const auto& ee_rotmat = global_ik_.body_rotation_matrix(ee_idx_);
    const Eigen::Matrix3d ee_rotmat_des = ee_orient.toRotationMatrix();
    for (int i = 0; i < 3; ++i) {
      global_ik_.AddBoundingBoxConstraint(ee_rotmat_des.col(i), ee_rotmat_des.col(i), ee_rotmat.col(i));
    }
  }

  std::pair<solvers::SolutionResult, solvers::SolutionResult> SolveIK(const Eigen::Vector3d& link6_pos, std::fstream* output_file) {
    std::pair<solvers::SolutionResult, solvers::SolutionResult> ik_status;
    // Solve IK analytically
    Eigen::Isometry3d link6_pose;
    link6_pose.linear() = ee_orient_.toRotationMatrix();
    link6_pose.translation() = link6_pos;
    const auto& q_analytical = analytical_ik_.inverse_kinematics(link6_pose);
    if (q_analytical.size() > 0) {
      ik_status.first = solvers::SolutionResult::kSolutionFound;
    } else {
      ik_status.first = solvers::SolutionResult::kInfeasibleConstraints;
    }

    // Solve IK using nonlinear IK
    WorldPositionConstraint nl_ik_pos_cnstr(analytical_ik_.robot(), ee_idx_, Eigen::Vector3d::Zero(), link6_pos, link6_pos);
    WorldQuatConstraint nl_ik_quat_cnstr(analytical_ik_.robot(), ee_idx_, Eigen::Vector4d(ee_orient_.w(), ee_orient_.x(), ee_orient_.y(), ee_orient_.z()), 0);
    int nl_ik_info;
    std::vector<std::string> infeasible_constraint;
    Eigen::VectorXd q_nl_ik_guess = Eigen::Matrix<double, 6, 1>::Zero();
    std::array<RigidBodyConstraint*, 2> nl_ik_cnstr = {{&nl_ik_pos_cnstr, &nl_ik_quat_cnstr}};
    IKoptions ik_options(analytical_ik_.robot());
    Eigen::VectorXd q_nl_ik(6);
    inverseKin(analytical_ik_.robot(), q_nl_ik_guess, q_nl_ik_guess, 2, nl_ik_cnstr.data(), ik_options, &q_nl_ik, &nl_ik_info, &infeasible_constraint);

    // Solve IK using global IK
    global_ik_pos_cnstr_.constraint()->UpdateLowerBound(link6_pos);
    global_ik_pos_cnstr_.constraint()->UpdateUpperBound(link6_pos);
    solvers::GurobiSolver gurobi_solver;
    solvers::SolutionResult global_ik_status = gurobi_solver.Solve(global_ik_);
    ik_status.second = global_ik_status;
    Eigen::Matrix<double, 6, 1> q_global;
    q_global.setZero();
    if (global_ik_status == solvers::SolutionResult::kSolutionFound) {
      q_global = global_ik_.ReconstructGeneralizedPositionSolution();
    }

    // Now print to file.
    if (output_file->is_open()) {
      (*output_file) << "\nposition:\n" << link6_pos.transpose() << std::endl;
      (*output_file) << "orientation (quaternion):\n" << ee_orient_.w() << " " << ee_orient_.x() << " " << ee_orient_.y() << " " << ee_orient_.z() << std::endl;
      (*output_file) << "analytical_ik_status: " << ik_status.first << std::endl;
      (*output_file) << "q_analytical:\n";
      for (const auto& qi_analytical : q_analytical) {
        (*output_file) << qi_analytical.transpose() << std::endl;
      }

      (*output_file) << "nonlinear_ik_status: " << nl_ik_info << std::endl;
      (*output_file) << "q_nonlinear_ik:\n" << q_nl_ik.transpose() << std::endl;

      (*output_file) << "global_ik_status: " << ik_status.second << std::endl;
      (*output_file) << "q_global:\n" << q_global.transpose() << std::endl;
    } else {
      throw std::runtime_error("file is not open.\n");
    }
    return ik_status;
  }

 private:
  IRB140AnalyticalKinematics analytical_ik_;
  multibody::GlobalInverseKinematics global_ik_;
  int ee_idx_;
  solvers::Binding<solvers::LinearConstraint> global_ik_pos_cnstr_;
  Eigen::Quaterniond ee_orient_;
};

void DoMain(int argc, char* argv[]) {
  if (argc != 3) {
    throw std::runtime_error("Usage is <infile> rotation_enum.\n");
  }

  std::string file_name(argv[1]);
  int rotation_enum = atoi(argv[2]);
  Eigen::AngleAxisd link6_angleaxis;
  switch (rotation_enum) {
    case 0 : {
      link6_angleaxis = Eigen::AngleAxisd(0, Eigen::Vector3d(1, 0, 0));
      break;
    }
    case 1 : {
      link6_angleaxis = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0));
      break;
    }
    default : {
      throw std::runtime_error("Unsupported rotation.\n");
    }
  }

  Eigen::Vector3d box_size(1, 1, 1);
  Eigen::Vector3d box_center(0.5, 0, 0.4);
  const int kNumPtsPerAxis = 3;
  Eigen::Matrix<double, 3, kNumPtsPerAxis> SamplesPerAxis;
  for (int axis = 0; axis < 3; ++axis) {
    SamplesPerAxis.row(axis) = Eigen::Matrix<double, 1, kNumPtsPerAxis>::LinSpaced(box_center(axis) - box_size(axis) / 2, box_center(axis) + box_size(axis) / 2);
  }
  std::fstream output_file;
  output_file.open(file_name, std::ios::app | std::ios::out);

  Eigen::Quaterniond link6_quat(link6_angleaxis);
  DUT dut(link6_quat);
  for (int i = 0; i < kNumPtsPerAxis; ++i) {
    for (int j = 0; j < kNumPtsPerAxis; ++j) {
      for (int k = 0; k < kNumPtsPerAxis; ++k) {
        Eigen::Vector3d link6_pos(SamplesPerAxis(0, i), SamplesPerAxis(1, j), SamplesPerAxis(2, k));
        Eigen::Isometry3d link6_pose;
        link6_pose.linear() = link6_angleaxis.toRotationMatrix();
        link6_pose.translation() = link6_pos;
        const auto& ik_status = dut.SolveIK(link6_pos, &output_file);
        if (ik_status.first == solvers::SolutionResult::kSolutionFound &&
            (ik_status.second == solvers::SolutionResult::kInfeasible_Or_Unbounded
                || ik_status.second == solvers::SolutionResult::kInfeasibleConstraints)) {
          std::cout << "global IK is infeasible, but analytical IK is feasible.\n";
        }
      }
    }
  }
  output_file.close();
}

}  // namespace IRB140
}  // namespace examples
}  // namespace drake



int main(int argc, char* argv[]) {
  drake::examples::IRB140::DoMain(argc, argv);
  return 0;
}