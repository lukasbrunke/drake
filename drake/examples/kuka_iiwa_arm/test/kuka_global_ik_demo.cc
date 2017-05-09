#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_path.h"
#include "drake/examples/kuka_iiwa_arm/dev/tools/simple_tree_visualizer.h"
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
#include "drake/examples/kuka_iiwa_arm/test/kuka_global_ik_util.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
using systems::DrakeVisualizer;

void AddObjects(RigidBodyTreed* rigid_body_tree) {
  const Eigen::Vector3d kRobotBasePos = rigid_body_tree->findFrame("iiwa_base")->get_transform_to_body().translation();
  const std::string mug_path = drake::GetDrakePath() + "/manipulation/models/objects/coffee_mug/urdf/coffee_mug.urdf";
  const Eigen::Vector3d kMugPos(kRobotBasePos(0) + 0.8, kRobotBasePos(1), kRobotBasePos(2));
  auto mug_frame = std::make_shared<RigidBodyFrame<double>>("mug", rigid_body_tree->get_mutable_body(0), kMugPos, Eigen::Vector3d(0, 0, M_PI));
  parsers::urdf::AddModelInstanceFromUrdfFile(mug_path, drake::multibody::joints::kFixed, mug_frame, rigid_body_tree);
  rigid_body_tree->addFrame(mug_frame);

  const std::string beets_path = drake::GetDrakePath() + "/manipulation/models/objects/beets_can/urdf/beets.urdf";
  const Eigen::Vector3d kBeetsPos(kMugPos(0) - 0.1, kMugPos(1) + 0.15, kMugPos(2) - 0.01);
  auto beets_frame = std::make_shared<RigidBodyFrame<double>>("beets", rigid_body_tree->get_mutable_body(0), kBeetsPos, Eigen::Vector3d::Zero());
  parsers::urdf::AddModelInstanceFromUrdfFile(beets_path, drake::multibody::joints::kFixed, beets_frame, rigid_body_tree);
  multibody::AddFlatTerrainToWorld(rigid_body_tree);
  rigid_body_tree->addFrame(beets_frame);

  const std::string bowl_path = drake::GetDrakePath() + "/manipulation/models/objects/bowl/urdf/bowl.urdf";
  const Eigen::Vector3d kBowlPos(kMugPos(0), kMugPos(1) - 0.25, kMugPos(2));
  auto bowl_frame = std::make_shared<RigidBodyFrame<double>>("bowl", rigid_body_tree->get_mutable_body(0), kBowlPos, Eigen::Vector3d::Zero());
  parsers::urdf::AddModelInstanceFromUrdfFile(bowl_path, drake::multibody::joints::kFixed, bowl_frame, rigid_body_tree);
  multibody::AddFlatTerrainToWorld(rigid_body_tree);
  rigid_body_tree->addFrame(bowl_frame);

  const std::string bottle_path = drake::GetDrakePath() + "/manipulation/models/objects/wine_bottle/urdf/bottle.urdf";
  const Eigen::Vector3d kBottlePos(kMugPos(0) - 0.25, kMugPos(1) - 0.05, kMugPos(2));
  auto bottle_frame = std::make_shared<RigidBodyFrame<double>>("bottle", rigid_body_tree->get_mutable_body(0), kBottlePos, Eigen::Vector3d::Zero());
  parsers::urdf::AddModelInstanceFromUrdfFile(bottle_path, drake::multibody::joints::kFixed, bottle_frame, rigid_body_tree);
  rigid_body_tree->addFrame(bottle_frame);
}

std::vector<Eigen::Matrix3Xd> SetFreeSpace(RigidBodyTreed* tree) {
  //const Eigen::Vector3d kBottlePos = tree->findFrame("bottle")->get_transform_to_body().translation();
  //const Eigen::Vector3d kMugPos = tree->findFrame("mug")->get_transform_to_body().translation();
  //const Eigen::Vector3d kBeetsPos = tree->findFrame("beets")->get_transform_to_body().translation();
  //const Eigen::Vector3d kBowlPos = tree->findFrame("bowl")->get_transform_to_body().translation();

  std::vector<Eigen::Matrix3Xd> box_vertices;

  Eigen::Isometry3d box_pose;
  box_pose.linear().setIdentity();
  box_pose.translation() << 0.32, -0.85, 1.02;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.25, 0.25, 0.5), box_pose, "box1"));


  box_pose.translation() << 0.42, -0.64, 1.02;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.15, 0.23, 0.5), box_pose, "box2"));

  box_pose.translation() << 0.57, -0.73, 0.83;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.2, 0.1, 0.12), box_pose, "box3"));

  box_pose.translation() << 0.55, -0.6, 1.08;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.3, 0.7, 0.4), box_pose, "box4"));

  box_pose.translation() << 0.6, -0.4, 1.02;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.2, 0.35, 0.5), box_pose, "box5"));

  box_pose.translation() << 0.32, -0.32, 1.02;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.4, 0.2, 0.5), box_pose, "box6"));

  box_pose.translation() << 0.25, -0.52, 1.02;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.3, 0.23, 0.5), box_pose, "box7"));

  box_pose.translation() << 0.25, -0.68, 1.23;
  box_vertices.push_back(AddBoxToTree(tree, Eigen::Vector3d(0.25, 0.14, 0.2), box_pose, "box8"));
  return box_vertices;
}

std::vector<Eigen::Matrix<double, 7, 1>> SolveGlobalIK(RigidBodyTreed* tree, const Eigen::Ref<Eigen::Vector3d>& mug_center, const std::vector<Eigen::Matrix3Xd>& free_space_vertices) {
  multibody::GlobalInverseKinematics global_ik(*tree);
  int link7_idx = tree->FindBodyIndex("iiwa_link_7");
  auto link7_rotmat = global_ik.body_rotation_matrix(link7_idx);
  auto link7_pos = global_ik.body_position(link7_idx);
  // y axis of link 7 frame is horizontal
  global_ik.AddLinearConstraint(link7_rotmat(2, 1) == 0);
  // z axis of link 7 frame points to the center of the cup, with a certain
  // distance to the cup
  global_ik.AddWorldPositionConstraint(link7_idx, Eigen::Vector3d(0, 0, 0.2), mug_center, mug_center);
  // height constraint
  global_ik.AddLinearConstraint(link7_pos(2), mug_center(2) - 0.02, mug_center(2) + 0.05);

  // link 7 above the table
  const double kTableHeight = mug_center(2) - 0.05;
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(0.04, 0, 0) + link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(-0.04, 0, 0) + link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(0, 0.04, 0) + link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(0, -0.04, 0) + link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(0.03, 0.03, 0) + link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(0.03, -0.03, 0) + link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(-0.03, 0.03, 0) + link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(-0.03, -0.03, 0) + link7_pos)(2) >= kTableHeight + 0.03);

  // Collision avoidance constraint
  if (true) {
    global_ik.BodyPointInOneOfRegions(link7_idx,
                                      Eigen::Vector3d::Zero(),
                                      free_space_vertices);
    global_ik.BodyPointInOneOfRegions(link7_idx,
                                      Eigen::Vector3d(0, 0.05, 0.1),
                                      free_space_vertices);
    global_ik.BodyPointInOneOfRegions(link7_idx,
                                      Eigen::Vector3d(0, -0.05, 0.1),
                                      free_space_vertices);
    global_ik.BodyPointInOneOfRegions(link7_idx,
                                      Eigen::Vector3d(0, -0.05, 0),
                                      free_space_vertices);
    global_ik.BodyPointInOneOfRegions(link7_idx,
                                      Eigen::Vector3d(0, 0.05, 0),
                                      free_space_vertices);
    int link5_idx = tree->FindBodyIndex("iiwa_link_5");
    global_ik.BodyPointInOneOfRegions(link5_idx,
                                      Eigen::Vector3d(0, 0, 0),
                                      free_space_vertices);
    global_ik.BodyPointInOneOfRegions(link5_idx,
                                      Eigen::Vector3d(0.04, 0, 0),
                                      free_space_vertices);
    global_ik.BodyPointInOneOfRegions(link5_idx,
                                      Eigen::Vector3d(-0.04, 0, 0),
                                      free_space_vertices);
    global_ik.BodyPointInOneOfRegions(link5_idx,
                                      Eigen::Vector3d(0, 0.04, 0),
                                      free_space_vertices);
    global_ik.BodyPointInOneOfRegions(link5_idx,
                                      Eigen::Vector3d(0, -0.04, 0),
                                      free_space_vertices);
  }
  solvers::GurobiSolver gurobi_solver;
  global_ik.SetSolverOption(solvers::SolverType::kGurobi, "OutputFlag", true);
  //int num_solutions = 2;
  //global_ik.SetSolverOption(solvers::SolverType::kGurobi, "PoolSearchMode", 1);
  //global_ik.SetSolverOption(solvers::SolverType::kGurobi, "PoolSolutions", num_solutions);
  solvers::SolutionResult sol_result = gurobi_solver.Solve(global_ik);
  if (sol_result != solvers::SolutionResult::kSolutionFound) {
    throw std::runtime_error("global ik fails.");
  }
  std::vector<Eigen::Matrix<double, 7, 1>> q;
  for (int i = 0; i < 1; ++i) {
    q.push_back(global_ik.ReconstructGeneralizedPositionSolution(i));
  }
  return q;
}

class GraspConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GraspConstraint)

  GraspConstraint(RigidBodyTreed* tree) : Constraint(4, 7),
                                          tree_(tree),
                                          cache_(tree_->CreateKinematicsCache()),
                                          link7_idx_(tree_->FindBodyIndex("iiwa_link_7")) {}

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd& y) const override {
    y.resize(4);
    cache_.initialize(x);
    tree_->doKinematics(cache_);
    const Eigen::Isometry3d link7_pose = tree_->CalcBodyPoseInWorldFrame(cache_, tree_->get_body(link7_idx_));
    y.head<3>() = link7_pose.linear() * Eigen::Vector3d(0, 0, 0.2) + link7_pose.translation();
    y(3) = link7_pose.linear()(2, 1);
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd& y) const  override{
    y.resize(4);
    Eigen::Matrix<double, 7, 1> q = math::autoDiffToValueMatrix(x);
    cache_.initialize(q);
    tree_->doKinematics(cache_);
    Eigen::Vector3d grasp_pt(0, 0, 0.2);
    Eigen::Vector4d y_val;
    Eigen::Matrix<double, 4, 7> dy_val;
    dy_val.setZero();
    y_val.head<3>() = tree_->transformPoints(cache_, grasp_pt, link7_idx_,0);
    dy_val.block<3, 7>(0, 0) = tree_->transformPointsJacobian(cache_, grasp_pt, link7_idx_, 0, true);
    Eigen::Vector3d y_axis(0, 1, 0);
    Eigen::Vector3d y_axis_pos = tree_->transformPoints(cache_, y_axis, link7_idx_, 0);
    Eigen::Vector3d ee_origin_pos = tree_->transformPoints(cache_, Eigen::Vector3d::Zero(), link7_idx_, 0);
    y_val(3) = y_axis_pos(2) - ee_origin_pos(2);
    Eigen::Matrix<double, 3, 7> dee_origin_pos = tree_->transformPointsJacobian(cache_, Eigen::Vector3d::Zero(), link7_idx_, 0, true);
    Eigen::Matrix<double, 3, 7> dy_axis_pos = tree_->transformPointsJacobian(cache_, y_axis, link7_idx_, 0, true);
    dy_val.row(3) = dy_axis_pos.row(2) - dee_origin_pos.row(2);
    Eigen::MatrixXd dy_val_dynamic = dy_val * math::autoDiffToGradientMatrix(x);
    Eigen::VectorXd y_val_dynamic = y_val;
    math::initializeAutoDiffGivenGradientMatrix(y_val_dynamic, dy_val_dynamic, y);
  }

  void SetMugCenter(const Eigen::Ref<const Eigen::Vector3d>& mug_center) {
    Eigen::Vector4d bnd;
    bnd << mug_center, 0;
    set_bounds(bnd, bnd);
  }

 private:
  RigidBodyTreed* tree_;
  mutable KinematicsCache<double> cache_;
  int link7_idx_;
};

Eigen::Matrix<double, 7, 1> SolveNonlinearIK(RigidBodyTreed* tree, const Eigen::Ref<Eigen::Vector3d>& mug_center) {
  solvers::MathematicalProgram nl_ik;
  auto q = nl_ik.NewContinuousVariables<7>();
  nl_ik.AddBoundingBoxConstraint(tree->joint_limit_min, tree->joint_limit_max, q);

  auto grasp_cnstr = std::make_shared<GraspConstraint>(tree);
  grasp_cnstr->SetMugCenter(mug_center);
  nl_ik.AddConstraint(grasp_cnstr, q);

  solvers::SolutionResult nl_ik_result = nl_ik.Solve();
  std::cout << "nonlinear IK status: " << nl_ik_result << std::endl;
  return nl_ik.GetSolution(q);
};

int DoMain() {
  drake::lcm::DrakeLcm lcm;
  auto tree = ConstructKuka();
  AddObjects(tree.get());
  const std::vector<Eigen::Matrix3Xd> free_space_vertices = SetFreeSpace(tree.get());
  tools::SimpleTreeVisualizer simple_tree_visualizer(*tree.get(), &lcm);
  // Palm faces +y axis of ee_frame. The face of the palm is at about (0, 0.1, 0)
  // of the ee_frame.
  // x axis of ee_frame is the longer axis of the palm.
  // Palm faces +z axis of link 7 frame. The face of the palm is at about (0, 0, 0.15)
  // of the link 7 frame
  // y axis of link 7 frame is the longer axis of the palm.
  //auto cache = tree->CreateKinematicsCache();
  //cache.initialize(Eigen::Matrix<double, 7, 1>::Zero());
  //tree->doKinematics(cache);
  //auto link5_pose = tree->CalcBodyPoseInWorldFrame(cache, *(tree->FindBody("iiwa_link_5")));
  //std::cout << link5_pose.matrix() << std::endl;
  //auto link_7_pose = tree->CalcBodyPoseInWorldFrame(cache, *link_7);
  //Eigen::Vector3d pt1_pos = link_7_pose.linear() * Eigen::Vector3d(0.1, 0, 0) + link_7_pose.translation();
  //Eigen::Vector3d pt2_pos = link_7_pose.linear() * Eigen::Vector3d(0, 0.1, 0) + link_7_pose.translation();
  //Eigen::Vector3d pt3_pos = link_7_pose.linear() * Eigen::Vector3d(0, 0, 0.1) + link_7_pose.translation();
  //std::cout << pt1_pos.transpose() << std::endl;
  //std::cout << pt2_pos.transpose() << std::endl;
  //std::cout << pt3_pos.transpose() << std::endl;
  auto mug_frame = tree->findFrame("mug");
  Eigen::Vector3d mug_pos = mug_frame->get_transform_to_body().translation();
  Eigen::Vector3d mug_center = mug_pos;
  mug_center(2) += 0.05;
  auto q_global = SolveGlobalIK(tree.get(), mug_center, free_space_vertices);
  for (int i = 0; i < static_cast<int>(q_global.size()); ++i) {
    simple_tree_visualizer.visualize(q_global[i]);
  }

  //Eigen::Matrix<double, 7, 1> q_nl = SolveNonlinearIK(tree.get(), mug_center);
  //simple_tree_visualizer.visualize(Eigen::Matrix<double, 7, 1>::Zero());
  //std::cout << q_global.transpose() << std::endl;
  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::kuka_iiwa_arm::DoMain();
}