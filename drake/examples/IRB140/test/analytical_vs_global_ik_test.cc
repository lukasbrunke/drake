#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"
#include "drake/multibody/global_inverse_kinematics.h"

#include <iostream>
#include <fstream>
#include <string>

using Eigen::Isometry3d;

namespace drake {
namespace examples {
namespace IRB140 {
class DUT {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DUT)

  DUT() : analytical_ik_(), global_ik_(*(analytical_ik_.robot()), 2) {}

  std::pair<solvers::SolutionResult, solvers::SolutionResult> SolveIK(const Isometry3d& link6_pose, std::fstream* output_file) {
    std::pair<solvers::SolutionResult, solvers::SolutionResult> ik_status;
    const auto& q_analytical = analytical_ik_.inverse_kinematics(link6_pose);
    if (q_analytical.size() > 0) {
      ik_status.first = solvers::SolutionResult::kSolutionFound;
    } else {
      ik_status.first = solvers::SolutionResult::kInfeasibleConstraints;
    }
    ik_status.second = solvers::SolutionResult::kSolutionFound;

    if (output_file->is_open()) {
      (*output_file) << "position:\n" << link6_pose.translation().transpose() << std::endl;
      Eigen::Quaterniond link6_quat(link6_pose.linear());
      (*output_file) << "orientation:\n" << link6_quat.w() << " " << link6_quat.x() << " " << link6_quat.y() << " " << link6_quat.z() << std::endl;
      (*output_file) << "analytical_ik_status: " << ik_status.first << std::endl;
      (*output_file) << "q_analytical:\n";
      for (const auto& qi_analytical : q_analytical) {
        (*output_file) << qi_analytical.transpose() << std::endl;
      }
    } else {
      throw std::runtime_error("file is not open.\n");
    }
    return ik_status;
  }
 private:
  IRB140AnalyticalKinematics analytical_ik_;
  multibody::GlobalInverseKinematics global_ik_;
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
  const int kNumPtsPerAxis = 11;
  Eigen::Matrix<double, 3, kNumPtsPerAxis> SamplesPerAxis;
  for (int axis = 0; axis < 3; ++axis) {
    SamplesPerAxis.row(axis) = Eigen::Matrix<double, 1, kNumPtsPerAxis>::LinSpaced(box_center(axis) - box_size(axis) / 2, box_center(axis) + box_size(axis) / 2);
  }
  std::fstream output_file;
  output_file.open(file_name, std::ios::app | std::ios::out);
  DUT dut;
  for (int i = 0; i < kNumPtsPerAxis; ++i) {
    for (int j = 0; j < kNumPtsPerAxis; ++j) {
      for (int k = 0; k < kNumPtsPerAxis; ++k) {
        Eigen::Vector3d link6_pos(SamplesPerAxis(0, i), SamplesPerAxis(1, j), SamplesPerAxis(2, k));
        Eigen::Isometry3d link6_pose;
        link6_pose.linear() = link6_angleaxis.toRotationMatrix();
        link6_pose.translation() = link6_pos;
        const auto& ik_status = dut.SolveIK(link6_pose, &output_file);
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