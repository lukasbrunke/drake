#include <memory>

#include "drake/common/find_resource.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/examples/kuka_iiwa_arm/test/kuka_global_ik_util.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/util/drakeGeometryUtil.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
using systems::DrakeVisualizer;

std::vector<BodyContactSphere> GetBodyContactSpheres(
    const RigidBodyTreed& tree) {
  std::vector<BodyContactSphere> points;
  const int link7_idx = tree.FindBodyIndex("iiwa_link_7");
  points.push_back(
      BodyContactSphere(link7_idx, Eigen::Vector3d(0, 0.07, 0.1), "pt1", 0));
  points.push_back(
      BodyContactSphere(link7_idx, Eigen::Vector3d(0, -0.07, 0.1), "pt2", 0));
  points.push_back(BodyContactSphere(link7_idx, Eigen::Vector3d(0, 0, -0.02),
                                     "wrist_sphere", 0.05));
  points.push_back(
      BodyContactSphere(link7_idx, Eigen::Vector3d(0, -0.07, 0.05), "pt4", 0));
  points.push_back(
      BodyContactSphere(link7_idx, Eigen::Vector3d(0, 0.07, 0.05), "pt6", 0));

  const int link5_idx = tree.FindBodyIndex("iiwa_link_5");
  points.push_back(BodyContactSphere(link5_idx, Eigen::Vector3d(0, 0, 0),
                                     "link5_sphere", 0.07));
  points.push_back(BodyContactSphere(
      link5_idx, Eigen::Vector3d(0.05, 0.06, 0.1), "pt11", 0));
  points.push_back(BodyContactSphere(
      link5_idx, Eigen::Vector3d(-0.05, 0.06, 0.1), "pt12", 0));

  const int link6_idx = tree.FindBodyIndex("iiwa_link_6");
  points.push_back(BodyContactSphere(
      link6_idx, Eigen::Vector3d(0.05, -0.06, 0.055), "pt13", 0));
  points.push_back(BodyContactSphere(
      link6_idx, Eigen::Vector3d(-0.05, -0.06, 0.055), "pt14", 0));
  points.push_back(BodyContactSphere(link6_idx, Eigen::Vector3d(0, 0, -0.035),
                                     "link6_sphere1", 0.06));
  points.push_back(BodyContactSphere(link6_idx, Eigen::Vector3d(0, 0.03, 0.01),
                                     "link6_sphere2", 0.05));
  points.push_back(
      BodyContactSphere(link6_idx, Eigen::Vector3d(0, -0.08, 0), "pt16", 0));

  return points;
}

void AddObjects(RigidBodyTreed* rigid_body_tree) {
  const Eigen::Vector3d kRobotBasePos = rigid_body_tree->findFrame("iiwa_base")
                                            ->get_transform_to_body()
                                            .translation();
  const std::string mug_path = FindResourceOrThrow(
      "drake/manipulation/models/objects/coffee_mug/urdf/coffee_mug.urdf");
  const Eigen::Vector3d kMugPos(kRobotBasePos(0) + 0.8, kRobotBasePos(1),
                                kRobotBasePos(2));
  auto mug_frame = std::make_shared<RigidBodyFrame<double>>(
      "mug", rigid_body_tree->get_mutable_body(0), kMugPos,
      Eigen::Vector3d(0, 0, 1.1 * M_PI));
  parsers::urdf::AddModelInstanceFromUrdfFile(
      mug_path, drake::multibody::joints::kFixed, mug_frame, rigid_body_tree);
  rigid_body_tree->addFrame(mug_frame);

  const std::string beets_path = FindResourceOrThrow(
      "drake/manipulation/models/objects/beets_can/urdf/beets.urdf");
  const Eigen::Vector3d kBeetsPos(kMugPos(0) - 0.1, kMugPos(1) + 0.15,
                                  kMugPos(2) - 0.01);
  auto beets_frame = std::make_shared<RigidBodyFrame<double>>(
      "beets", rigid_body_tree->get_mutable_body(0), kBeetsPos,
      Eigen::Vector3d::Zero());
  parsers::urdf::AddModelInstanceFromUrdfFile(beets_path,
                                              drake::multibody::joints::kFixed,
                                              beets_frame, rigid_body_tree);
  multibody::AddFlatTerrainToWorld(rigid_body_tree);
  rigid_body_tree->addFrame(beets_frame);

  const std::string bowl_path = FindResourceOrThrow(
      "drake/manipulation/models/objects/bowl/urdf/bowl.urdf");
  const Eigen::Vector3d kBowlPos(kMugPos(0) + 0.04, kMugPos(1) - 0.25,
                                 kMugPos(2));
  auto bowl_frame = std::make_shared<RigidBodyFrame<double>>(
      "bowl", rigid_body_tree->get_mutable_body(0), kBowlPos,
      Eigen::Vector3d::Zero());
  parsers::urdf::AddModelInstanceFromUrdfFile(
      bowl_path, drake::multibody::joints::kFixed, bowl_frame, rigid_body_tree);
  multibody::AddFlatTerrainToWorld(rigid_body_tree);
  rigid_body_tree->addFrame(bowl_frame);

  const std::string bottle_path = FindResourceOrThrow(
      "drake/manipulation/models/objects/wine_bottle/urdf/bottle.urdf");
  const Eigen::Vector3d kBottlePos(kMugPos(0) - 0.25, kMugPos(1) - 0.05,
                                   kMugPos(2));
  auto bottle_frame = std::make_shared<RigidBodyFrame<double>>(
      "bottle", rigid_body_tree->get_mutable_body(0), kBottlePos,
      Eigen::Vector3d::Zero());
  parsers::urdf::AddModelInstanceFromUrdfFile(bottle_path,
                                              drake::multibody::joints::kFixed,
                                              bottle_frame, rigid_body_tree);
  rigid_body_tree->addFrame(bottle_frame);
}


std::vector<Box> FreeSpaceBoxes() {
  std::vector<Box> boxes;
  Eigen::Isometry3d box_pose;
  box_pose.linear().setIdentity();
  box_pose.translation() << 0.29, -0.85, 1.07;
  boxes.push_back(Box(Eigen::Vector3d(0.45, 0.25, 0.6), box_pose, "box1",
                      Eigen::Vector4d(0.1, 0.3, 0.7, 0.3)));

  box_pose.translation() << 0.42, -0.67, 1.02;
  boxes.push_back(Box(Eigen::Vector3d(0.15, 0.29, 0.5), box_pose, "box2",
                      Eigen::Vector4d(0.4, 0.1, 0.5, 0.3)));

  box_pose.translation() << 0.5, -0.73, 1.02;
  boxes.push_back(Box(Eigen::Vector3d(0.31, 0.1, 0.5), box_pose, "box3",
                      Eigen::Vector4d(0.8, 0.5, 0.1, 0.3)));

  box_pose.translation() << 0.5, -0.6, 1.08;
  boxes.push_back(Box(Eigen::Vector3d(0.31, 0.7, 0.4), box_pose, "box4",
                      Eigen::Vector4d(0.3, 0.1, 0.8, 0.3)));

  box_pose.translation() << 0.42, -0.75, 1.02;
  boxes.push_back(Box(Eigen::Vector3d(0.17, 0.45, 0.5), box_pose, "box5",
                      Eigen::Vector4d(0.2, 0.8, 0.3, 0.3)));

  box_pose.translation() << 0.2, -0.42, 1.02;
  boxes.push_back(Box(Eigen::Vector3d(0.4, 0.42, 0.5), box_pose, "box6",
                      Eigen::Vector4d(0.1, 0.5, 0.5, 0.3)));

  return boxes;
}

std::vector<multibody::GlobalInverseKinematics::Polytope3D> SetFreeSpace(
    const std::vector<Box>& free_space_boxes) {
  // const Eigen::Vector3d kBottlePos =
  // tree->findFrame("bottle")->get_transform_to_body().translation();
  // const Eigen::Vector3d kMugPos =
  // tree->findFrame("mug")->get_transform_to_body().translation();
  // const Eigen::Vector3d kBeetsPos =
  // tree->findFrame("beets")->get_transform_to_body().translation();
  // const Eigen::Vector3d kBowlPos =
  // tree->findFrame("bowl")->get_transform_to_body().translation();

  std::vector<multibody::GlobalInverseKinematics::Polytope3D> polytopes;
  for (const auto& box : free_space_boxes) {
    Eigen::Matrix<double, 6, 3> A;
    A << box.pose.linear().transpose(), -box.pose.linear().transpose();
    Eigen::Matrix<double, 6, 1> b;
    b << box.pose.linear().transpose() * box.pose.translation() + box.size / 2,
        -box.pose.linear().transpose() * box.pose.translation() + box.size / 2;
    polytopes.emplace_back(A, b);
  }
  return polytopes;
}

std::vector<Eigen::Matrix<double, 7, 1>> SolveGlobalIK(
    RigidBodyTreed* tree, const Eigen::Ref<Eigen::Vector3d>& mug_center,
    const std::vector<multibody::GlobalInverseKinematics::Polytope3D>&
        free_space_polytopes,
    const std::vector<BodyContactSphere>& body_contact_spheres) {
  multibody::GlobalInverseKinematics::Options global_ik_options;
  global_ik_options.num_intervals_per_half_axis = 2;
  global_ik_options.linear_constraint_only = false;
  multibody::GlobalInverseKinematics global_ik(*tree, global_ik_options);
  int link7_idx = tree->FindBodyIndex("iiwa_link_7");
  auto link7_rotmat = global_ik.body_rotation_matrix(link7_idx);
  auto link7_pos = global_ik.body_position(link7_idx);
  // y axis of link 7 frame is horizontal
  global_ik.AddLinearConstraint(link7_rotmat(2, 1) == 0);
  // z axis of link 7 frame points to the center of the cup, with a certain
  // distance to the cup
  global_ik.AddWorldPositionConstraint(link7_idx, Eigen::Vector3d(0, 0, 0.2),
                                       mug_center, mug_center);
  // height constraint
  global_ik.AddLinearConstraint(link7_pos(2), mug_center(2) - 0.02,
                                mug_center(2) + 0.15);

  // link 7 above the table
  const double kTableHeight = mug_center(2) - 0.05;
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(0.04, 0, 0) +
                                 link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(-0.04, 0, 0) +
                                 link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(0, 0.04, 0) +
                                 link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(0, -0.04, 0) +
                                 link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint((link7_rotmat * Eigen::Vector3d(0.03, 0.03, 0) +
                                 link7_pos)(2) >= kTableHeight + 0.03);
  global_ik.AddLinearConstraint(
      (link7_rotmat * Eigen::Vector3d(0.03, -0.03, 0) + link7_pos)(2) >=
      kTableHeight + 0.03);
  global_ik.AddLinearConstraint(
      (link7_rotmat * Eigen::Vector3d(-0.03, 0.03, 0) + link7_pos)(2) >=
      kTableHeight + 0.03);
  global_ik.AddLinearConstraint(
      (link7_rotmat * Eigen::Vector3d(-0.03, -0.03, 0) + link7_pos)(2) >=
      kTableHeight + 0.03);

  // Collision avoidance constraint
  if (true) {
    for (const auto& body_contact_sphere : body_contact_spheres) {
      global_ik.BodySphereInOneOfPolytopes(
          body_contact_sphere.link_idx, body_contact_sphere.p_BQ,
          body_contact_sphere.radius, free_space_polytopes);
    }
  }
  global_ik.AddBoundingBoxConstraint(-1, -0.55, global_ik.body_position(5)(1));
  solvers::GurobiSolver gurobi_solver;
  // solvers::MosekSolver mosek_solver;
  // mosek_solver.set_stream_logging(true, "");
  global_ik.SetSolverOption(solvers::GurobiSolver::id(), "OutputFlag", true);
  const int num_solutions = 10;
  if (num_solutions > 1) {
    global_ik.SetSolverOption(solvers::GurobiSolver::id(), "PoolSearchMode", 1);
    global_ik.SetSolverOption(solvers::GurobiSolver::id(), "PoolSolutions",
                              num_solutions);
  }
  solvers::SolutionResult sol_result = gurobi_solver.Solve(global_ik);
  if (sol_result != solvers::SolutionResult::kSolutionFound) {
    throw std::runtime_error("global ik fails.");
  }
  std::vector<Eigen::Matrix<double, 7, 1>> q;

  for (int i = 0; i < num_solutions; ++i) {
    q.push_back(global_ik.ReconstructGeneralizedPositionSolution(0, i));
  }
  return q;
}

class GraspConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GraspConstraint)

  GraspConstraint(RigidBodyTreed* tree)
      : Constraint(4, 7),
        tree_(tree),
        cache_(tree_->CreateKinematicsCache()),
        link7_idx_(tree_->FindBodyIndex("iiwa_link_7")) {}

  void SetMugCenter(const Eigen::Ref<const Eigen::Vector3d>& mug_center) {
    Eigen::Vector4d bnd;
    bnd << mug_center, 0;
    set_bounds(bnd, bnd);
  }

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override {
    y->resize(4);
    cache_.initialize(x);
    tree_->doKinematics(cache_);
    const Eigen::Isometry3d link7_pose =
        tree_->CalcBodyPoseInWorldFrame(cache_, tree_->get_body(link7_idx_));
    y->head<3>() = link7_pose.linear() * Eigen::Vector3d(0, 0, 0.2) +
                   link7_pose.translation();
    (*y)(3) = link7_pose.linear()(2, 1);
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override {
    y->resize(4);
    Eigen::Matrix<double, 7, 1> q = math::autoDiffToValueMatrix(x);
    cache_.initialize(q);
    tree_->doKinematics(cache_);
    Eigen::Vector3d grasp_pt(0, 0, 0.2);
    Eigen::Vector4d y_val;
    Eigen::Matrix<double, 4, 7> dy_val;
    dy_val.setZero();
    y_val.head<3>() = tree_->transformPoints(cache_, grasp_pt, link7_idx_, 0);
    dy_val.block<3, 7>(0, 0) =
        tree_->transformPointsJacobian(cache_, grasp_pt, link7_idx_, 0, true);
    Eigen::Vector3d y_axis(0, 1, 0);
    Eigen::Vector3d y_axis_pos =
        tree_->transformPoints(cache_, y_axis, link7_idx_, 0);
    Eigen::Vector3d ee_origin_pos =
        tree_->transformPoints(cache_, Eigen::Vector3d::Zero(), link7_idx_, 0);
    y_val(3) = y_axis_pos(2) - ee_origin_pos(2);
    Eigen::Matrix<double, 3, 7> dee_origin_pos = tree_->transformPointsJacobian(
        cache_, Eigen::Vector3d::Zero(), link7_idx_, 0, true);
    Eigen::Matrix<double, 3, 7> dy_axis_pos =
        tree_->transformPointsJacobian(cache_, y_axis, link7_idx_, 0, true);
    dy_val.row(3) = dy_axis_pos.row(2) - dee_origin_pos.row(2);
    Eigen::MatrixXd dy_val_dynamic = dy_val * math::autoDiffToGradientMatrix(x);
    Eigen::VectorXd y_val_dynamic = y_val;
    math::initializeAutoDiffGivenGradientMatrix(y_val_dynamic, dy_val_dynamic,
                                                *y);
  }

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const {
    throw std::runtime_error("Not supported.");
  }

 private:
  RigidBodyTreed* tree_;
  mutable KinematicsCache<double> cache_;
  int link7_idx_;
};

Eigen::Matrix<double, 7, 1> SolveNonlinearIK(
    RigidBodyTreed* tree, const Eigen::Ref<Eigen::Vector3d>& mug_center) {
  solvers::MathematicalProgram nl_ik;
  auto q = nl_ik.NewContinuousVariables<7>();
  nl_ik.AddBoundingBoxConstraint(tree->joint_limit_min, tree->joint_limit_max,
                                 q);

  auto grasp_cnstr = std::make_shared<GraspConstraint>(tree);
  grasp_cnstr->SetMugCenter(mug_center);
  nl_ik.AddConstraint(grasp_cnstr, q);

  solvers::SolutionResult nl_ik_result = nl_ik.Solve();
  std::cout << "nonlinear IK status: " << nl_ik_result << std::endl;
  return nl_ik.GetSolution(q);
};

enum class Command {
  FindPosture,
  VisualizeBox,
  VisualizePosture,
};

Command Int2Command(int command) {
  switch (command) {
    case 0:
      return Command::FindPosture;
    case 1:
      return Command::VisualizeBox;
    case 2:
      return Command::VisualizePosture;
    default:
      throw std::runtime_error("Unknown command.");
  }
}

int DoMain(int argc, char** argv) {
  if (argc != 2) {
    throw std::runtime_error("The command should be kuka_global_ik_demo <cmd>");
  }
  const int int_command = atoi(argv[1]);

  const Command command = Int2Command(int_command);
  drake::lcm::DrakeLcm lcm;
  auto tree = ConstructKuka();
  AddObjects(tree.get());
  const std::vector<Box> free_space_boxes = FreeSpaceBoxes();
  const std::vector<BodyContactSphere> body_contact_spheres =
      GetBodyContactSpheres(*tree);
  const auto free_space_polytopes = SetFreeSpace(free_space_boxes);
  // Palm faces +y axis of ee_frame. The face of the palm is at about (0, 0.1,
  // 0)
  // of the ee_frame.
  // x axis of ee_frame is the longer axis of the palm.
  // Palm faces +z axis of link 7 frame. The face of the palm is at about (0, 0,
  // 0.15)
  // of the link 7 frame
  // y axis of link 7 frame is the longer axis of the palm.
  // auto cache = tree->CreateKinematicsCache();
  // cache.initialize(Eigen::Matrix<double, 7, 1>::Zero());
  // tree->doKinematics(cache);
  // auto link5_pose = tree->CalcBodyPoseInWorldFrame(cache,
  // *(tree->FindBody("iiwa_link_5")));
  // std::cout << link5_pose.matrix() << std::endl;
  // auto link_7_pose = tree->CalcBodyPoseInWorldFrame(cache, *link_7);
  // Eigen::Vector3d pt1_pos = link_7_pose.linear() * Eigen::Vector3d(0.1, 0, 0)
  // + link_7_pose.translation();
  // Eigen::Vector3d pt2_pos = link_7_pose.linear() * Eigen::Vector3d(0, 0.1, 0)
  // + link_7_pose.translation();
  // Eigen::Vector3d pt3_pos = link_7_pose.linear() * Eigen::Vector3d(0, 0, 0.1)
  // + link_7_pose.translation();
  // std::cout << pt1_pos.transpose() << std::endl;
  // std::cout << pt2_pos.transpose() << std::endl;
  // std::cout << pt3_pos.transpose() << std::endl;
  auto mug_frame = tree->findFrame("mug");
  Eigen::Vector3d mug_pos = mug_frame->get_transform_to_body().translation();
  Eigen::Vector3d mug_center = mug_pos;
  mug_center(2) += 0.05;

  std::vector<Eigen::Matrix<double, 7, 1>> q_visualize;
  switch (command) {
    case Command::FindPosture: {
      auto q_global = SolveGlobalIK(tree.get(), mug_center,
                                    free_space_polytopes, body_contact_spheres);
      q_visualize = q_global;
      break;
    }
    case Command::VisualizeBox: {
      q_visualize.push_back(Eigen::VectorXd::Zero(7));
      break;
    }
    case Command::VisualizePosture: {
      // The posture is stored in "iiwa_collision_free_postures.txt"
      q_visualize.push_back((Eigen::VectorXd(7) << -2.43652, -1.68951, 2.22475,
                             -1.34332, -2.39852, 0.679638, 0.621955)
                                .finished());
      q_visualize.push_back((Eigen::VectorXd(7) << -2.41753, -1.5708, 2.22536,
                             -1.5029, -2.62575, 0.646562, 0.763762)
                                .finished());
      q_visualize.push_back((Eigen::VectorXd(7) << 0.753897, 1.5708, 2.2089,
                             1.53765, 0.461033, 0.630911, -2.31224)
                                .finished());
      q_visualize.push_back((Eigen::VectorXd(7) << 2.60512, -1.5708, 0.421987,
                             0.927804, -1.78631, 1.04628, 0.679504)
                                .finished());
      q_visualize.push_back((Eigen::VectorXd(7) << 2.60117, -1.87278, 2.03608,
                             0.611646, -1.30804, -1.32409, 1.84767)
                                .finished());
      q_visualize.push_back((Eigen::VectorXd(7) << -0.0835957, 1.87362,
                             -1.22199, 0.675552, -2.82894, -1.58692, 1.63452)
                                .finished());
      break;
    }
  }
  // Add free space boxes to visualization
  for (const auto& box : free_space_boxes) {
    AddBoxToTree(tree.get(), box.size, box.pose, box.name, box.color);
  }
  // Add body contact points to visualization
  for (const auto& sphere : body_contact_spheres) {
    AddSphereToBody(tree.get(), sphere.link_idx, sphere.p_BQ, sphere.name,
                    std::max(sphere.radius, 0.01));
  }
  manipulation::SimpleTreeVisualizer simple_tree_visualizer(*tree.get(), &lcm);
  auto cache = tree->CreateKinematicsCache();
  auto link_7 = tree->FindBody("iiwa_link_7");
  for (int i = 0; i < static_cast<int>(q_visualize.size()); ++i) {
    simple_tree_visualizer.visualize(q_visualize[i]);
    std::cout << "q:\n" << q_visualize[i].transpose() << std::endl;
    cache.initialize(q_visualize[i]);
    tree->doKinematics(cache);
    auto link_7_pose = tree->CalcBodyPoseInWorldFrame(cache, *link_7);
    std::cout << "link7 pos:\n" << link_7_pose.matrix() << std::endl;
    getchar();
  }

  // Eigen::Matrix<double, 7, 1> q_nl = SolveNonlinearIK(tree.get(),
  // mug_center);
  // simple_tree_visualizer.visualize(Eigen::Matrix<double, 7, 1>::Zero());
  // std::cout << q_global.transpose() << std::endl;
  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char** argv) {
  return drake::examples::kuka_iiwa_arm::DoMain(argc, argv);
}
