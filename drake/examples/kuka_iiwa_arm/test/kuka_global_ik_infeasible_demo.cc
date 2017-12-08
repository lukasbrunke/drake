#include <memory>

#include <gtest/gtest.h>

#include "drake/common/drake_path.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/util/drakeGeometryUtil.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/shapes/geometry.h"
#include "drake/multibody/joints/fixed_joint.h"
#include "drake/examples/kuka_iiwa_arm/test/kuka_global_ik_util.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
using systems::DrakeVisualizer;

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

void AddFridge(RigidBodyTreed* tree, const Eigen::Vector3d& kFridgePos) {
  const std::string microwave_path = drake::GetDrakePath() + "/manipulation/models/objects/fridge/urdf/fridge.urdf";
  auto microwave_frame = std::make_shared<RigidBodyFrame<double>>("microwave", tree->get_mutable_body(0), kFridgePos, Eigen::Vector3d(0, 0, 0));
  parsers::urdf::AddModelInstanceFromUrdfFile(microwave_path, drake::multibody::joints::kFixed, microwave_frame, tree);
  tree->addFrame(microwave_frame);
}

std::vector<Eigen::Matrix3Xd> SetFreeSpace(RigidBodyTreed* tree) {
  std::vector<Eigen::Matrix3Xd> box_vertices;

  Eigen::Isometry3d box_pose;
  box_pose.linear().setIdentity();
  box_pose.translation() << 0.3, -0.6, 1.5;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.6, 0.6, 0.4), box_pose, "box1"));

  box_pose.translation() << 0.53, -0.6, 1.05;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.36, 0.6, 0.6), box_pose, "box2"));

  //box_pose.translation() << 0.4, -0.8, 1.05;
  //box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.4, 0.4, 0.6), box_pose, "box3"));

  return box_vertices;
}

std::vector<std::pair<int, Eigen::Vector3d>> AddBodyCollisionPoint(RigidBodyTreed* tree) {
  std::vector<std::pair<int, Eigen::Vector3d>> collision_pts;
  int link5_idx = tree->FindBodyIndex("iiwa_link_5");
  Eigen::Vector3d pt;
  pt << 0.06, 0, 0;
  AddSphereToBody(tree, link5_idx, pt, "link5_pt1");
  collision_pts.emplace_back(link5_idx, pt);

  pt << -0.06, 0, 0;
  AddSphereToBody(tree, link5_idx, pt, "link5_pt2");
  collision_pts.emplace_back(link5_idx, pt);

  pt << 0, 0.06, 0;
  AddSphereToBody(tree, link5_idx, pt, "link5_pt3");
  collision_pts.emplace_back(link5_idx, pt);

  pt << 0, -0.06, 0;
  AddSphereToBody(tree, link5_idx, pt, "link5_pt4");
  collision_pts.emplace_back(link5_idx, pt);

  int link6_idx = tree->FindBodyIndex("iiwa_link_6");
  pt << 0.05, 0, -0.05;
  AddSphereToBody(tree, link6_idx, pt, "link6_pt1");
  collision_pts.emplace_back(link6_idx, pt);

  pt << -0.05, 0, -0.05;
  AddSphereToBody(tree, link6_idx, pt, "link6_pt2");
  collision_pts.emplace_back(link6_idx, pt);

  pt << 0.04, -0.04, 0.05;
  AddSphereToBody(tree, link6_idx, pt, "link6_pt3");
  collision_pts.emplace_back(link6_idx, pt);

  pt << -0.04, -0.04, 0.05;
  AddSphereToBody(tree, link6_idx, pt, "link6_pt4");
  collision_pts.emplace_back(link6_idx, pt);

  pt << 0.05, 0.06, 0;
  AddSphereToBody(tree, link6_idx, pt, "link6_pt5");
  collision_pts.emplace_back(link6_idx, pt);

  pt << -0.05, 0.06, 0;
  AddSphereToBody(tree, link6_idx, pt, "link6_pt6");
  collision_pts.emplace_back(link6_idx, pt);

  return collision_pts;
}

Eigen::Matrix<double, 7, 1> SolveGlobalIKreachable(
    RigidBodyTreed* tree, const Eigen::Vector3d& bottle_pos,
    const std::vector<Eigen::Matrix3Xd>& free_space_vertices,
    const std::vector<std::pair<int, Eigen::Vector3d>>& collision_pts, bool with_collision) {
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

  if (with_collision) {
    for (const auto& body_pt : collision_pts) {
      global_ik.BodyPointInOneOfRegions(body_pt.first, body_pt.second, free_space_vertices);
    }
  }

  solvers::GurobiSolver gurobi_solver;
  global_ik.SetSolverOption(solvers::SolverType::kGurobi, "OutputFlag", true);
  auto sol_result = gurobi_solver.Solve(global_ik);
  if (with_collision) {
    EXPECT_TRUE(sol_result == solvers::SolutionResult::kInfeasible_Or_Unbounded
      || sol_result == solvers::SolutionResult::kInfeasibleConstraints);
  } else {
    EXPECT_EQ(sol_result, solvers::SolutionResult::kSolutionFound);
  }

  return global_ik.ReconstructGeneralizedPositionSolution();
};

int DoMain() {
  drake::lcm::DrakeLcm lcm;
  auto tree = ConstructKuka();
  multibody::AddFlatTerrainToWorld(tree.get());
  auto robot_base_frame = tree->findFrame("iiwa_base");
  const Eigen::Vector3d kBasePos = robot_base_frame->get_transform_to_body().translation();
  const Eigen::Vector3d kBottleReachablePos(kBasePos(0) + 0.8, kBasePos(1) + 0.2, kBasePos(2) + 0.04);
  std::cout << "bottle pos: " << kBottleReachablePos.transpose() << std::endl;
  AddBottle(tree.get(), kBottleReachablePos);
  //const Eigen::Vector3d kMicrowavePos(kBasePos(0) + 0.2, kBasePos(1) - 0.2, kBasePos(2));
  //AddMicrowave(tree.get(), kMicrowavePos);
  const Eigen::Vector3d kFridgePos(kBasePos(0) + 0.2, kBasePos(1) - 0.2, kBasePos(2));
  AddFridge(tree.get(), kFridgePos);
  auto free_space_vertices = SetFreeSpace(tree.get());
  auto collision_pts = AddBodyCollisionPoint(tree.get());
  manipulation::SimpleTreeVisualizer simple_tree_visualizer(*tree.get(), &lcm);
  auto q_feasible = SolveGlobalIKreachable(tree.get(), kBottleReachablePos, free_space_vertices, collision_pts, false);
  simple_tree_visualizer.visualize(q_feasible);
  SolveGlobalIKreachable(tree.get(), kBottleReachablePos, free_space_vertices, collision_pts, true);
  simple_tree_visualizer.visualize(Eigen::Matrix<double, 7, 1>::Zero());
  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::examples::kuka_iiwa_arm::DoMain();
}
