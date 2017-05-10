#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iterator>

#include <gtest/gtest.h>

#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"
#include "drake/examples/IRB140/test/irb140_common.h"
#include "drake/multibody/constraint/rigid_body_constraint.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/multibody/rigid_body_ik.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/common/call_matlab.h"

using Eigen::Isometry3d;

namespace drake {
namespace examples {
namespace IRB140 {
class IKresult {
 public:
  IKresult() {}

  Eigen::Isometry3d& ee_pose() {return ee_pose_;}

  const Eigen::Isometry3d& ee_pose() const {return ee_pose_;}

  solvers::SolutionResult& analytical_ik_status() {return analytical_ik_status_;}

  const solvers::SolutionResult& analytical_ik_status() const {return analytical_ik_status_;}

  solvers::SolutionResult& global_ik_status() {return global_ik_status_;}

  const solvers::SolutionResult& global_ik_status() const {return global_ik_status_;}

  int& nl_ik_status() {return nl_ik_status_;}

  const int& nl_ik_status() const {return nl_ik_status_;}

  int& nl_ik_resolve_status() {return nl_ik_resolve_status_;}

  const int& nl_ik_resolve_status() const {return nl_ik_resolve_status_;}

  std::vector<Eigen::Matrix<double, 6, 1>>& q_analytical_ik() {return q_analytical_ik_;}

  const std::vector<Eigen::Matrix<double, 6, 1>>& q_analytical_ik() const {return q_analytical_ik_;}

  Eigen::Matrix<double, 6, 1>& q_global_ik() {return q_global_ik_;}

  const Eigen::Matrix<double, 6, 1>& q_global_ik() const {return q_global_ik_;}

  Eigen::Matrix<double, 6, 1>& q_nl_ik() {return q_nl_ik_;}

  const Eigen::Matrix<double, 6, 1>& q_nl_ik() const {return q_nl_ik_;}

  Eigen::Matrix<double, 6, 1>& q_nl_ik_resolve() {return q_nl_ik_resolve_;}

  const Eigen::Matrix<double, 6, 1>& q_nl_ik_resolve() const {return q_nl_ik_resolve_;}

  double& global_ik_time() {return global_ik_time_;}

  const double& global_ik_time() const {return global_ik_time_;}

  void printToFile(std::fstream* output_file) const {
    // Now print to file.
    if (output_file->is_open()) {
      Eigen::Vector3d link6_pos = ee_pose_.translation();
      Eigen::Quaterniond ee_orient(ee_pose_.linear());
      (*output_file) << "\nposition:\n" << link6_pos.transpose() << std::endl;
      (*output_file) << "orientation (quaternion):\n"
                     << ee_orient.w() << " " << ee_orient.x() << " "
                     << ee_orient.y() << " " << ee_orient.z() << std::endl;
      (*output_file) << "analytical_ik_status: " << analytical_ik_status_
                     << std::endl;
      (*output_file) << "q_analytical:\n";
      for (const auto& qi_analytical : q_analytical_ik_) {
        (*output_file) << qi_analytical.transpose() << std::endl;
      }

      (*output_file) << "nonlinear_ik_status: " << nl_ik_status_ << std::endl;
      (*output_file) << "q_nonlinear_ik:\n" << q_nl_ik_.transpose() << std::endl;

      (*output_file) << "global_ik_status: " << global_ik_status_ << std::endl;
      (*output_file) << "q_global:\n" << q_global_ik_.transpose() << std::endl;
      (*output_file) << "global_ik_time: " << global_ik_time_ << std::endl;

      (*output_file) << "nonlinear_ik_resolve_status: " << nl_ik_resolve_status_
                     << std::endl;
      (*output_file) << "q_nonlinear_ik_resolve:\n"
                     << q_nl_ik_resolve_.transpose() << std::endl;
    } else {
      throw std::runtime_error("file is not open.\n");
    }
  }

 private:
  Eigen::Isometry3d ee_pose_;
  solvers::SolutionResult analytical_ik_status_;
  solvers::SolutionResult global_ik_status_;
  int nl_ik_status_;
  int nl_ik_resolve_status_;
  std::vector<Eigen::Matrix<double, 6, 1>> q_analytical_ik_;
  Eigen::Matrix<double, 6, 1> q_global_ik_;
  Eigen::Matrix<double, 6, 1> q_nl_ik_;
  Eigen::Matrix<double, 6, 1> q_nl_ik_resolve_;
  double global_ik_time_;
};

class DUT {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DUT)

  DUT(const Eigen::Quaterniond& ee_orient)
      : analytical_ik_(),
        global_ik_(*(analytical_ik_.robot()), 2),
        ee_idx_(analytical_ik_.robot()->FindBodyIndex("link_6")),
        global_ik_pos_cnstr_(global_ik_.AddWorldPositionConstraint(
            ee_idx_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero())),
        ee_orient_(ee_orient) {
    // Fix the end effector body orientation
    const auto& ee_rotmat = global_ik_.body_rotation_matrix(ee_idx_);
    const Eigen::Matrix3d ee_rotmat_des = ee_orient.toRotationMatrix();
    for (int i = 0; i < 3; ++i) {
      global_ik_.AddBoundingBoxConstraint(
          ee_rotmat_des.col(i), ee_rotmat_des.col(i), ee_rotmat.col(i));
    }

    for (int i = 1; i < robot()->get_num_bodies(); ++i) {
      const auto &body_R = global_ik_.body_rotation_matrix(i);
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
  }

  RigidBodyTreed* robot() const {return analytical_ik_.robot();}

  void SolveAnalyticalIK(const Eigen::Vector3d& link6_pos, IKresult* ik_result) {
    // Solve IK analytically
    Eigen::Isometry3d link6_pose;
    link6_pose.linear() = ee_orient_.toRotationMatrix();
    link6_pose.translation() = link6_pos;
    ik_result->q_analytical_ik() = analytical_ik_.inverse_kinematics(link6_pose);
    if (ik_result->q_analytical_ik().size() > 0) {
      ik_result->analytical_ik_status() = solvers::SolutionResult::kSolutionFound;
    } else {
      ik_result->analytical_ik_status() = solvers::SolutionResult::kInfeasibleConstraints;
    }
  }

  void SolveGlobalIK(const Eigen::Vector3d &link6_pos, IKresult *ik_result) {
    global_ik_pos_cnstr_.constraint()->UpdateLowerBound(link6_pos);
    global_ik_pos_cnstr_.constraint()->UpdateUpperBound(link6_pos);
    solvers::GurobiSolver gurobi_solver;
    solvers::MosekSolver mosek_solver;

    global_ik_.SetSolverOption(solvers::SolverType::kGurobi, "FeasibilityTol", 1E-5);
    //global_ik_.SetSolverOption(solvers::SolverType::kGurobi, "OutputFlag", 1);
    solvers::SolutionResult global_ik_status = gurobi_solver.Solve(global_ik_);
    //solvers::SolutionResult global_ik_status = mosek_solver.Solve(global_ik_);
    Eigen::Matrix<double, 6, 1> q_global;
    q_global.setZero();
    if (global_ik_status == solvers::SolutionResult::kSolutionFound) {
      q_global = global_ik_.ReconstructGeneralizedPositionSolution();
    }
    ik_result->global_ik_status() = global_ik_status;
    ik_result->q_global_ik() = q_global;
    ik_result->global_ik_time() = global_ik_.computation_time();
  }

  void SolveNonlinearIK(const Eigen::Vector3d& link6_pos, IKresult* ik_result,
                        const Eigen::Ref<const Eigen::Matrix<double, 6, 1>>& q_guess,
                        bool resolve_flag) {
    WorldPositionConstraint nl_ik_pos_cnstr(analytical_ik_.robot(), ee_idx_,
                                            Eigen::Vector3d::Zero(), link6_pos,
                                            link6_pos);
    WorldQuatConstraint nl_ik_quat_cnstr(
        analytical_ik_.robot(), ee_idx_,
        Eigen::Vector4d(ee_orient_.w(), ee_orient_.x(), ee_orient_.y(),
                        ee_orient_.z()),
        0);
    int nl_ik_info;
    std::vector<std::string> infeasible_constraint;
    std::array<RigidBodyConstraint*, 2> nl_ik_cnstr = {
        {&nl_ik_pos_cnstr, &nl_ik_quat_cnstr}};
    IKoptions ik_options(analytical_ik_.robot());
    Eigen::VectorXd q_ik_guess = q_guess;
    Eigen::VectorXd q_nl_ik(6);
    inverseKin(analytical_ik_.robot(), q_ik_guess, q_ik_guess, 2,
               nl_ik_cnstr.data(), ik_options, &q_nl_ik, &nl_ik_info,
               &infeasible_constraint);
    if (resolve_flag) {
      ik_result->nl_ik_resolve_status() = nl_ik_info;
      ik_result->q_nl_ik_resolve() = q_nl_ik;
    } else {
      ik_result->nl_ik_status() = nl_ik_info;
      ik_result->q_nl_ik() = q_nl_ik;
    }
  }
  void SolveIK(
      const Eigen::Vector3d& link6_pos, std::fstream* output_file) {
    IKresult ik_result;
    // Solve IK using analytical IK
    SolveAnalyticalIK(link6_pos, &ik_result);


    // Solve IK using nonlinear IK
    SolveNonlinearIK(link6_pos, &ik_result, Eigen::VectorXd::Zero(6), false);

    // Solve IK using global IK
    SolveGlobalIK(link6_pos, &ik_result);
    if (ik_result.global_ik_status() == solvers::SolutionResult::kSolutionFound) {
      SolveNonlinearIK(link6_pos, &ik_result, ik_result.q_global_ik(), true);
    }

    if (ik_result.analytical_ik_status() == solvers::SolutionResult::kSolutionFound &&
        (ik_result.global_ik_status() ==
            solvers::SolutionResult::kInfeasible_Or_Unbounded ||
            ik_result.global_ik_status() ==
                solvers::SolutionResult::kInfeasibleConstraints)) {
      std::cout
          << "global IK is infeasible, but analytical IK is feasible.\n";
    }

    // For print out.
    ik_result.ee_pose().linear() = ee_orient_.toRotationMatrix();
    ik_result.ee_pose().translation() = link6_pos;
    ik_result.printToFile(output_file);
  }

  int ee_idx() const {return ee_idx_;}

 private:
  IRB140AnalyticalKinematics analytical_ik_;
  multibody::GlobalInverseKinematics global_ik_;
  int ee_idx_;
  solvers::Binding<solvers::LinearConstraint> global_ik_pos_cnstr_;
  Eigen::Quaterniond ee_orient_;
};

void RemoveFileIfExist(const std::string& file_name) {
  std::ifstream file(file_name);
  if (file) {
    if (remove(file_name.c_str()) != 0) {
      throw std::runtime_error("Error deleting file " + file_name);
    }
  }
  file.close();
}

void DoMain(int argc, char* argv[]) {
  if (argc != 3) {
    throw std::runtime_error("Usage is <infile> rotation_enum.\n");
  }

  std::string file_name(argv[1]);
  int rotation_enum = atoi(argv[2]);
  Eigen::AngleAxisd link6_angleaxis;
  switch (rotation_enum) {
    case 0: {
      link6_angleaxis = Eigen::AngleAxisd(0, Eigen::Vector3d(1, 0, 0));
      break;
    }
    case 1: {
      link6_angleaxis = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1, 0, 0));
      break;
    }
    case 2: {
      link6_angleaxis = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0, 0, 1));
      break;
    }
    default: { throw std::runtime_error("Unsupported rotation.\n"); }
  }

  // Remove the file if it exists
  RemoveFileIfExist(file_name);

  Eigen::Vector3d box_size(1, 1, 1);
  Eigen::Vector3d box_center(0.5, 0, 0.4);
  const int kNumPtsPerAxis = 21;
  Eigen::Matrix<double, 3, kNumPtsPerAxis> SamplesPerAxis;
  for (int axis = 0; axis < 3; ++axis) {
    SamplesPerAxis.row(axis) =
        Eigen::Matrix<double, 1, kNumPtsPerAxis>::LinSpaced(
            box_center(axis) - box_size(axis) / 2,
            box_center(axis) + box_size(axis) / 2);
  }

  std::fstream output_file;
  output_file.open(file_name, std::ios::app | std::ios::out);

  Eigen::Quaterniond link6_quat(link6_angleaxis);
  DUT dut(link6_quat);
  int sample_count = 0;
  for (int i = 0; i < kNumPtsPerAxis; ++i) {
    for (int j = 0; j < kNumPtsPerAxis; ++j) {
      for (int k = 0; k < kNumPtsPerAxis; ++k) {
        Eigen::Vector3d link6_pos(SamplesPerAxis(0, i), SamplesPerAxis(1, j),
                                  SamplesPerAxis(2, k));
        Eigen::Isometry3d link6_pose;
        link6_pose.linear() = link6_angleaxis.toRotationMatrix();
        link6_pose.translation() = link6_pos;
        dut.SolveIK(link6_pos, &output_file);
        std::cout << "sample count: " << sample_count << std::endl;
        ++sample_count;
      }
    }
  }
  output_file.close();
}

std::vector<std::string> BreakLineBySpaces(const std::string& line) {
  std::istringstream iss(line);
  std::vector<std::string> strings{std::istream_iterator<std::string>{iss},
                                   std::istream_iterator<std::string>{}};
  return strings;
}

void ReadOutputFile(std::ifstream& file, std::vector<IKresult>* ik_results) {
  std::string line;
  if (file.is_open()) {
    while (getline(file, line)) {
      IKresult ik_result;
      getline(file, line);
      if (line == "position:") {
        getline(file, line);
        // Parse the position
        const auto pos_str = BreakLineBySpaces(line);
        for (int i = 0; i < 3; ++i) {
          ik_result.ee_pose().translation()(i) = std::atof(pos_str[i].c_str());
        }
      } else {
        throw std::runtime_error("oops");
      }

      getline(file, line);
      if (line == "orientation (quaternion):") {
        getline(file, line);
        // parse the orientation
        const auto quat_str = BreakLineBySpaces(line);
        Eigen::Quaterniond quat(std::atof(quat_str[0].c_str()), std::atof(quat_str[1].c_str()), std::atof(quat_str[2].c_str()), std::atof(quat_str[3].c_str()));
        ik_result.ee_pose().linear() = quat.toRotationMatrix();
      } else {
        throw std::runtime_error("oops");
      }

      getline(file, line);
      const auto analytical_ik_status_str = BreakLineBySpaces(line);
      if (analytical_ik_status_str[0] == "analytical_ik_status:") {
        ik_result.analytical_ik_status() = static_cast<solvers::SolutionResult>(std::atoi(analytical_ik_status_str[1].c_str()));
      } else {
        throw std::runtime_error("oops");
      }

      getline(file, line);
      if (line == "q_analytical:") {
        while (getline(file, line)) {
          const auto q_analytical_ik_str = BreakLineBySpaces(line);
          if (q_analytical_ik_str[0] == "nonlinear_ik_status:") {
            ik_result.nl_ik_status() = std::atoi(q_analytical_ik_str[1].c_str());
            break;
          } else {
            Eigen::Matrix<double, 6, 1> q_analytical_ik;
            for (int i = 0; i < 6; ++i) {
              q_analytical_ik(i) = std::atof(q_analytical_ik_str[i].c_str());
            }
            ik_result.q_analytical_ik().push_back(q_analytical_ik);
          }
        }
      } else {
        throw std::runtime_error("oops");
      }

      getline(file, line);
      if (line == "q_nonlinear_ik:") {
        getline(file, line);
        const auto q_nl_ik_str = BreakLineBySpaces(line);
        for (int i = 0; i < 6; ++i) {
          ik_result.q_nl_ik()(i) = std::atof(q_nl_ik_str[i].c_str());
        }
      } else {
        throw std::runtime_error("oops");
      }

      getline(file, line);
      const auto global_ik_status_str = BreakLineBySpaces(line);
      if (global_ik_status_str[0] == "global_ik_status:") {
        ik_result.global_ik_status() = static_cast<solvers::SolutionResult>(std::atoi(global_ik_status_str[1].c_str()));
      } else {
        throw std::runtime_error("oops");
      }

      getline(file, line);
      if (line == "q_global:") {
        getline(file, line);
        const auto q_global_ik_str = BreakLineBySpaces(line);
        for (int i = 0; i < 6; ++i) {
          ik_result.q_global_ik()(i) = std::atof(q_global_ik_str[i].c_str());
        }
      } else {
        throw std::runtime_error("oops");
      }

      getline(file, line);
      const auto global_ik_time_str = BreakLineBySpaces(line);
      if (global_ik_time_str[0] == "global_ik_time:") {
        ik_result.global_ik_time() = std::atof(global_ik_time_str[1].c_str());
      } else {
        throw std::runtime_error("oops");
      }

      getline(file, line);
      const auto nl_ik_resolve_status_str = BreakLineBySpaces(line);
      if (nl_ik_resolve_status_str[0] == "nonlinear_ik_resolve_status:") {
        ik_result.nl_ik_resolve_status() = std::atoi(nl_ik_resolve_status_str[1].c_str());
      } else {
        throw std::runtime_error("oops");
      }

      getline(file, line);
      if (line == "q_nonlinear_ik_resolve:") {
        getline(file, line);
        const auto q_nl_ik_resolve_str = BreakLineBySpaces(line);
        for (int i = 0; i < 6; ++i) {
          ik_result.q_nl_ik_resolve()(i) = std::atof(q_nl_ik_resolve_str[i].c_str());
        }
      } else {
        throw std::runtime_error("oops");
      }

      ik_results->push_back(ik_result);
    }
  }

}
void DebugOutputFile(int argc, char* argv[]) {
  if (argc != 5) {
    throw std::runtime_error("Usage is <infile>. num_pts_per_axis <outfile1> <outfile2>");
  }
  std::string in_file_name(argv[1]);
  int num_pts_per_axis = atoi(argv[2]);
  std::string out_file_name1(argv[3]);
  std::string out_file_name2(argv[4]);
  std::vector<IKresult> ik_results;
  ik_results.reserve(num_pts_per_axis * num_pts_per_axis * num_pts_per_axis);

  std::ifstream in_file(in_file_name);
  ReadOutputFile(in_file, &ik_results);
  in_file.close();

  RemoveFileIfExist(out_file_name1);
  std::fstream output_file1;
  output_file1.open(out_file_name1, std::ios::app | std::ios::out);

  RemoveFileIfExist(out_file_name2);
  std::fstream output_file2;
  output_file2.open(out_file_name2, std::ios::app | std::ios::out);

  Eigen::Quaterniond link6_quat(ik_results[0].ee_pose().linear());
  DUT dut(link6_quat);
  // Only find the case that analytical ik is feasible, but global IK is infeasible
  RigidBodyTreed* robot = dut.robot();
  KinematicsCache<double> cache = robot->CreateKinematicsCache();

  for (auto& ik_result : ik_results) {
    /*if (ik_result.analytical_ik_status() == solvers::SolutionResult::kSolutionFound
        && ik_result.global_ik_status() != ik_result.analytical_ik_status()) {
      // ik_result.printToFile(&output_file);

      // First make sure analytical IK is correct.
      EXPECT_TRUE((ik_result.q_analytical_ik()[0].array() >= robot->joint_limit_min.array()).all());
      EXPECT_TRUE((ik_result.q_analytical_ik()[0].array() <= robot->joint_limit_max.array()).all());
      cache.initialize(ik_result.q_analytical_ik()[0]);
      robot->doKinematics(cache);
      std::array<Eigen::Isometry3d, 6> link_pose;
      for (int i = 0; i < 6; ++i) {
        link_pose[i] = robot->CalcBodyPoseInWorldFrame(cache, *(robot->FindBody("link_" + std::to_string(i + 1))));
      }
      CompareIsometry3d(link_pose[5], ik_result.ee_pose(), 1E-5);

      // Now solve global IK
      dut.SolveGlobalIK(ik_result.ee_pose().translation(), &ik_result);
      ik_result.printToFile(&output_file1);
    }*/
    /*if (ik_result.analytical_ik_status() == solvers::SolutionResult::kInfeasibleConstraints
        && ik_result.global_ik_status() == solvers::SolutionResult::kSolutionFound) {
      dut.SolveAnalyticalIK(ik_result.ee_pose().translation(), &ik_result);
      ik_result.printToFile(&output_file1);
    }*/
    /*if (ik_result.global_ik_status() == solvers::SolutionResult::kSolutionFound
        && ik_result.analytical_ik_status() == solvers::SolutionResult::kSolutionFound) {
      cache.initialize(ik_result.q_global_ik());
      dut.robot()->doKinematics(cache);
      Eigen::Isometry3d ee_pose = dut.robot()->CalcBodyPoseInWorldFrame(cache,
                                                                        dut.robot()->get_body(
                                                                            dut.ee_idx()));
      double pos_error =
          (ee_pose.translation() - ik_result.ee_pose().translation()).norm();
      if (pos_error <= 0.04) {
        dut.SolveGlobalIK(ik_result.ee_pose().translation(), &ik_result);
        dut.SolveNonlinearIK(ik_result.ee_pose().translation(), &ik_result, ik_result.q_global_ik(), true);
        ik_result.printToFile(&output_file1);
      }
    }*/
    /*if (ik_result.global_ik_status() == solvers::SolutionResult::kInvalidInput) {
      dut.SolveGlobalIK(ik_result.ee_pose().translation(), &ik_result);
      ik_result.printToFile(&output_file1);
    }*/
    /*if ((ik_result.global_ik_status() == solvers::SolutionResult::kInfeasibleConstraints
        || ik_result.global_ik_status() == solvers::SolutionResult::kInfeasible_Or_Unbounded)
        && ik_result.analytical_ik_status() == solvers::SolutionResult::kSolutionFound) {
      dut.SolveGlobalIK(ik_result.ee_pose().translation(), &ik_result);
      ik_result.printToFile(&output_file1);
    }*/
    /*if (ik_result.global_ik_status() == solvers::SolutionResult::kSolutionFound) {
      dut.SolveNonlinearIK(ik_result.ee_pose().translation(), &ik_result, ik_result.q_global_ik(), true);
      ik_result.printToFile(&output_file1);
    }*/
    if (ik_result.global_ik_status() == solvers::SolutionResult::kInfeasible_Or_Unbounded
        || ik_result.global_ik_status() == solvers::SolutionResult::kInfeasibleConstraints) {
      dut.SolveGlobalIK(ik_result.ee_pose().translation(), &ik_result);
      ik_result.printToFile(&output_file1);
    }
    ik_result.printToFile(&output_file2);
  }
  output_file1.close();
  output_file2.close();
}

void WriteRuntimeToFile(const Eigen::VectorXd& runtime, const std::string& out_file_name) {
  RemoveFileIfExist(out_file_name);
  std::fstream output_file;
  output_file.open(out_file_name, std::ios::app | std::ios::out);
  if (output_file.is_open()) {
    output_file << runtime;
  }
  output_file.close();
}

void AnalyzeNonlinearIKresult(const std::vector<IKresult>& ik_results) {
  // Analyze whether global IK gives a better initial seed to nonlinear IK.
  int both_infeasible_count = 0;
  int both_feasible_count = 0;
  int resolve_feasible_count = 0; // IK fails initially, but succeeds with global
                                  // IK result as the seed.
  int initial_feasible_count = 0; // IK succeeds initially, but fails with global
                                  // IK result as the seed.
  for (const auto& ik_result : ik_results) {
    if (ik_result.analytical_ik_status() == solvers::SolutionResult::kSolutionFound) {
      if (ik_result.nl_ik_status() <= 3) {
        if (ik_result.nl_ik_resolve_status() <= 3) {
          ++both_feasible_count;
        } else {
          ++initial_feasible_count;
        }
      } else {
        // Initially IK fails.
        if (ik_result.nl_ik_resolve_status() <= 3) {
          ++resolve_feasible_count;
        } else {
          ++both_infeasible_count;
        }
      }
    }
  }
  std::cout << "Both initial IK and resolve IK succeed: " << both_feasible_count << std::endl;
  std::cout << "Neither initial IK nor resolve IK succeed: " << both_infeasible_count << std::endl;
  std::cout << "Initial IK fails, but resolve IK succeed: " << resolve_feasible_count << std::endl;
  std::cout << "Initial IK succeeds, but resolve IK fails: " << initial_feasible_count << std::endl;
}

void AnalyzeOutputFile(int argc, char* argv[]) {
  using common::CallMatlab;
  if (argc != 6) {
    throw std::runtime_error("Usage is <infile> num_pts_per_axis <outfile1> <outfile2> <outfile3>");
  }
  std::string in_file_name(argv[1]);
  int num_pts_per_axis = atoi(argv[2]);
  std::string out_file_name1(argv[3]);
  std::string out_file_name2(argv[4]);
  std::string out_file_name3(argv[5]);

  std::vector<IKresult> ik_results;
  ik_results.reserve(num_pts_per_axis * num_pts_per_axis * num_pts_per_axis);

  std::ifstream in_file(in_file_name);
  ReadOutputFile(in_file, &ik_results);
  in_file.close();

  std::vector<IKresult> both_feasible;
  std::vector<IKresult> both_infeasible;
  std::vector<IKresult> relaxation; // global IK is feasible, but analytical IK
                                    // is not.
  for (const auto& ik_result : ik_results) {
    if (ik_result.global_ik_status() == solvers::SolutionResult::kSolutionFound
        && ik_result.analytical_ik_status() == solvers::SolutionResult::kSolutionFound) {
      both_feasible.push_back(ik_result);
    } else if ((ik_result.global_ik_status() == solvers::SolutionResult::kInfeasible_Or_Unbounded
                || ik_result.global_ik_status() == solvers::SolutionResult::kInfeasibleConstraints)
                && ik_result.analytical_ik_status() == solvers::SolutionResult::kInfeasibleConstraints) {
      both_infeasible.push_back(ik_result);
    } else if (ik_result.global_ik_status() == solvers::SolutionResult::kSolutionFound
        && ik_result.analytical_ik_status() == solvers::SolutionResult::kInfeasibleConstraints) {
      relaxation.push_back(ik_result);
    }
  }

  Eigen::VectorXd both_feasible_time(both_feasible.size());
  for (int i = 0; i < static_cast<int>(both_feasible.size()); ++i) {
    // Draw the histogram of the computation time.
    both_feasible_time(i) = both_feasible[i].global_ik_time();
  }
  WriteRuntimeToFile(both_feasible_time, out_file_name1);

  Eigen::VectorXd relaxation_time(relaxation.size());
  for (int i = 0; i < static_cast<int>(relaxation.size()); ++i) {
    relaxation_time(i) = relaxation[i].global_ik_time();
  }
  WriteRuntimeToFile(relaxation_time, out_file_name2);

  Eigen::VectorXd both_infeasible_time(both_infeasible.size());
  for (int i = 0; i < static_cast<int>(both_infeasible.size()); ++i) {
    both_infeasible_time(i) = both_infeasible[i].global_ik_time();
  }
  WriteRuntimeToFile(both_infeasible_time, out_file_name3);

  Eigen::Quaterniond link6_quat(ik_results[0].ee_pose().linear());
  DUT dut(link6_quat);
  Eigen::Matrix2Xd global_ik_ee_error(2, both_feasible.size());
  KinematicsCache<double> cache = dut.robot()->CreateKinematicsCache();
  std::vector<bool> q_global_ik_at_joint_limits(both_feasible.size());
  Eigen::Matrix2Xd global_ik_ee_error_at_joint_limits(2, 0);
  Eigen::Matrix2Xd global_ik_ee_error_not_at_joint_limits(2, 0);
  for (int i = 0; i < static_cast<int>(both_feasible.size()); ++i) {
    cache.initialize(both_feasible[i].q_global_ik());
    q_global_ik_at_joint_limits[i] = ((both_feasible[i].q_global_ik() - dut.robot()->joint_limit_min).array() < 1E-2).any()
        || ((both_feasible[i].q_global_ik() - dut.robot()->joint_limit_max).array() > -1E-2).any();

    dut.robot()->doKinematics(cache);
    Eigen::Isometry3d ee_pose_global_ik = dut.robot()->CalcBodyPoseInWorldFrame(cache, dut.robot()->get_body(dut.ee_idx()));
    global_ik_ee_error(0, i) = (ee_pose_global_ik.translation() - both_feasible[i].ee_pose().translation()).norm();
    global_ik_ee_error(1, i) = Eigen::AngleAxisd(ee_pose_global_ik.linear().transpose() * both_feasible[i].ee_pose().linear()).angle();
    if (q_global_ik_at_joint_limits[i]) {
      global_ik_ee_error_at_joint_limits.conservativeResize(Eigen::NoChange, global_ik_ee_error_at_joint_limits.cols() + 1);
      global_ik_ee_error_at_joint_limits.col(global_ik_ee_error_at_joint_limits.cols() - 1) = global_ik_ee_error.col(i);
    } else {
      global_ik_ee_error_not_at_joint_limits.conservativeResize(Eigen::NoChange, global_ik_ee_error_not_at_joint_limits.cols() + 1);
      global_ik_ee_error_not_at_joint_limits.col(global_ik_ee_error_not_at_joint_limits.cols() - 1) = global_ik_ee_error.col(i);
    }
  }

  auto h_fig = CallMatlab(1, "figure", 1);
  CallMatlab("hold", "on");
  auto h_pose_error_at_joint_limits = CallMatlab(1, "plot",
                                 global_ik_ee_error_at_joint_limits.row(0) * 100.0,
                                 global_ik_ee_error_at_joint_limits.row(1) / M_PI * 180.0);
  CallMatlab("set", h_pose_error_at_joint_limits[0], "LineStyle", "none", "Marker", "o", "Color", Eigen::Vector3d(1, 0, 0));
  CallMatlab("set", h_pose_error_at_joint_limits[0], "DisplayName", "Joint limits active");

  auto h_pose_error_not_at_joint_limits = CallMatlab(1, "plot",
                                                 global_ik_ee_error_not_at_joint_limits.row(0) * 100.0,
                                                 global_ik_ee_error_not_at_joint_limits.row(1) / M_PI * 180.0);
  CallMatlab("set", h_pose_error_not_at_joint_limits[0], "LineStyle", "none", "Marker", "x", "Color", Eigen::Vector3d(0, 0, 1));
  CallMatlab("set", h_pose_error_not_at_joint_limits[0], "DisplayName", "Joint limits inactive");

  CallMatlab("legend", "show");
  auto h_xlabel = CallMatlab(1, "xlabel", "position error (cm)");
  auto h_ylabel = CallMatlab(1, "ylabel", "orientation angle error (degree)");
  CallMatlab("set", h_xlabel[0], "FontSize", 25);
  CallMatlab("set", h_ylabel[0], "FontSize", 25);
  auto h_axis = CallMatlab(1, "gca");
  CallMatlab("set", h_axis[0], "FontSize", 20);

  AnalyzeNonlinearIKresult(ik_results);
}
}  // namespace IRB140
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  //drake::examples::IRB140::DoMain(argc, argv);
  //drake::examples::IRB140::DebugOutputFile(argc, argv);
  drake::examples::IRB140::AnalyzeOutputFile(argc, argv);
  return 0;
}
