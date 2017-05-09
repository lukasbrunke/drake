#include <memory>

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "drake/common/drake_path.h"
#include "drake/examples/kuka_iiwa_arm/dev/tools/simple_tree_visualizer.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/examples/schunk_wsg/schunk_wsg_constants.h"
#include "drake/examples/schunk_wsg/schunk_wsg_lcm.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/lcmt_schunk_wsg_command.hpp"
#include "drake/lcmt_schunk_wsg_status.hpp"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/util/drakeGeometryUtil.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/shapes/geometry.h"
#include "drake/multibody/joints/fixed_joint.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
using schunk_wsg::SchunkWsgStatusSender;
using schunk_wsg::SchunkWsgTrajectoryGenerator;
using systems::DrakeVisualizer;
using systems::InputPortDescriptor;
using systems::OutputPortDescriptor;
using systems::RigidBodyPlant;

std::unique_ptr<RigidBodyTreed> ConstructKuka() {
  std::unique_ptr<RigidBodyTreed> rigid_body_tree = std::make_unique<RigidBodyTreed>();

  const std::string model_path = drake::GetDrakePath() +
      "/manipulation/models/iiwa_description/urdf/"
          "iiwa14_polytope_collision.urdf";

  const std::string table_path = drake::GetDrakePath() + "/examples/kuka_iiwa_arm/models/table/"
      "extra_heavy_duty_table_surface_only_collision.sdf";

  auto table1_frame = std::make_shared<RigidBodyFrame<double>>(
      "iiwa_table",
      rigid_body_tree->get_mutable_body(0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  auto table2_frame = std::make_shared<RigidBodyFrame<double>>(
      "object_table",
      rigid_body_tree->get_mutable_body(0), Eigen::Vector3d(0.8, 0, 0), Eigen::Vector3d::Zero());

  parsers::sdf::AddModelInstancesFromSdfFile(table_path, drake::multibody::joints::kFixed, table1_frame, rigid_body_tree.get());

  parsers::sdf::AddModelInstancesFromSdfFile(table_path, drake::multibody::joints::kFixed,
                                             table2_frame, rigid_body_tree.get());

  const double kTableTopZInWorld = 0.736 + 0.057 / 2;
  const Eigen::Vector3d kRobotBase(-0.243716, -0.625087, kTableTopZInWorld);

  auto robot_base_frame = std::make_shared<RigidBodyFrame<double>>(
      "iiwa_base", rigid_body_tree->get_mutable_body(0), kRobotBase, Eigen::Vector3d::Zero());
  rigid_body_tree->addFrame(robot_base_frame);

  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kFixed,
      robot_base_frame,
      rigid_body_tree.get());

  auto iiwa_frame_ee = rigid_body_tree->findFrame("iiwa_frame_ee");
  const std::string schunk_path = drake::GetDrakePath() + "/examples/schunk_wsg/models/schunk_wsg_50_fixed_joint.sdf";
  parsers::sdf::AddModelInstancesFromSdfFile(schunk_path, drake::multibody::joints::kFixed, iiwa_frame_ee, rigid_body_tree.get());

  return rigid_body_tree;
}

void AddBottle(RigidBodyTreed* tree, const Eigen::Vector3d& kBottlePos) {
  const std::string bottle_path = drake::GetDrakePath() + "/manipulation/models/objects/wine_bottle/urdf/bottle.urdf";
  auto bottle_frame = std::make_shared<RigidBodyFrame<double>>("bottle", tree->get_mutable_body(0), kBottlePos, Eigen::Vector3d(M_PI_2, 0, 0));
  parsers::urdf::AddModelInstanceFromUrdfFile(bottle_path, drake::multibody::joints::kFixed, bottle_frame, tree);
  tree->addFrame(bottle_frame);
}

void AddMicrowave(RigidBodyTreed* tree, const Eigen::Vector3d& kMicrowavePos) {
  const std::string microwave_path = drake::GetDrakePath() + "/manipulation/models/objects/microwave/urdf/microwave.urdf";
  auto microwave_frame = std::make_shared<RigidBodyFrame<double>>("microwave", tree->get_mutable_body(0), kMicrowavePos, Eigen::Vector3d(0, 0, 0));
  parsers::urdf::AddModelInstanceFromUrdfFile(microwave_path, drake::multibody::joints::kFixed, microwave_frame, tree);
  tree->addFrame(microwave_frame);
}

Eigen::Matrix<double, 7, 1> SolveGlobalIKreachable(RigidBodyTreed* tree, const Eigen::Vector3d& bottle_pos) {
  multibody::GlobalInverseKinematics global_ik(*tree);
  int link7_idx = tree->FindBodyIndex("iiwa_link_7");
  auto link7_rotmat = global_ik.body_rotation_matrix(link7_idx);
  auto link7_pos = global_ik.body_position(link7_idx);
  global_ik.AddBoundingBoxConstraint(Eigen::Vector3d(0, 0, -1), Eigen::Vector3d(0, 0, -1), link7_rotmat.col(2));
  global_ik.AddBoundingBoxConstraint(0, 0, link7_rotmat(2, 1));
  global_ik.AddBoundingBoxConstraint(0, 0, link7_rotmat(1, 1));
  global_ik.AddBoundingBoxConstraint(0, 0, link7_rotmat(0, 0));
  global_ik.AddBoundingBoxConstraint(0, 0, link7_rotmat(2, 0));
  const Eigen::Vector3d kGraspPt(0, 0, 0.2);
  global_ik.AddWorldPositionConstraint(
      link7_idx, kGraspPt,
      Eigen::Vector3d(bottle_pos(0), bottle_pos(1) - 0.2, bottle_pos(2)),
      Eigen::Vector3d(bottle_pos(0), bottle_pos(1) - 0.1, bottle_pos(2)));

  solvers::GurobiSolver gurobi_solver;
  global_ik.SetSolverOption(solvers::SolverType::kGurobi, "OutputFlag", true);
  auto sol_result = gurobi_solver.Solve(global_ik);
  EXPECT_EQ(sol_result, solvers::SolutionResult::kSolutionFound);
  return global_ik.ReconstructGeneralizedPositionSolution();
};

int DoMain() {
  drake::lcm::DrakeLcm lcm;
  auto tree = ConstructKuka();
  auto robot_base_frame = tree->findFrame("iiwa_base");
  const Eigen::Vector3d kBasePos = robot_base_frame->get_transform_to_body().translation();
  const Eigen::Vector3d kBottleReachablePos(kBasePos(0) + 0.8, kBasePos(1) + 0.2, kBasePos(2) + 0.04);
  std::cout << kBottleReachablePos.transpose() << std::endl;
  AddBottle(tree.get(), kBottleReachablePos);
  const Eigen::Vector3d kMicrowavePos(kBasePos(0) + 0.2, kBasePos(1) - 0.2, kBasePos(2));
  AddMicrowave(tree.get(), kMicrowavePos);
  tools::SimpleTreeVisualizer simple_tree_visualizer(*tree.get(), &lcm);
  auto q_feasible = SolveGlobalIKreachable(tree.get(), kBottleReachablePos);
  simple_tree_visualizer.visualize(q_feasible);
  //simple_tree_visualizer.visualize(Eigen::Matrix<double, 7, 1>::Zero());
  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::kuka_iiwa_arm::DoMain();
}