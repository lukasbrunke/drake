#include <memory>

#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/examples/kuka_iiwa_arm/test/kuka_global_ik_util.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/global_inverse_kinematics.h"
#include "drake/multibody/parsers/sdf_parser.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/rotation_constraint.h"

DEFINE_string(cabinet_size, "small",
              "Cabinet size could be small, medium or large.");
DEFINE_bool(find_path, false, "Find path or a single posture.");

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
solvers::VectorXDecisionVariable BodySphereInOneOfPolytopes(
    solvers::MathematicalProgram* prog, const Matrix3<symbolic::Variable>& R,
    const Vector3<symbolic::Variable>& pos,
    const Eigen::Ref<const Eigen::Vector3d>& p_BQ, double radius,
    const std::vector<multibody::GlobalInverseKinematics::Polytope3D>&
        polytopes) {
  DRAKE_DEMAND(radius >= 0);
  const int num_polytopes = static_cast<int>(polytopes.size());
  const auto z = prog->NewBinaryVariables(num_polytopes, "z");
  // z1 + ... + zn = 1
  prog->AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(num_polytopes), 1,
                                    z);

  const auto y =
      prog->NewContinuousVariables<3, Eigen::Dynamic>(3, num_polytopes, "y");
  const Vector3<symbolic::Expression> p_WQ = pos + R * p_BQ;
  // p_WQ = y.col(0) + ... + y.col(n)
  prog->AddLinearEqualityConstraint(
      p_WQ - y.cast<symbolic::Expression>().rowwise().sum(),
      Eigen::Vector3d::Zero());

  for (int i = 0; i < num_polytopes; ++i) {
    DRAKE_DEMAND(polytopes[i].A.rows() == polytopes[i].b.rows());
    prog->AddLinearConstraint(
        polytopes[i].A * y.col(i) <=
        (polytopes[i].b - polytopes[i].A.rowwise().norm() * radius) * z(i));
  }

  return z;
}

enum class CabinetType { kSmall, kMedium, kLarge };

Eigen::Vector3d CabinetSize(CabinetType type) {
  if (type == CabinetType::kSmall) {
    return Eigen::Vector3d(0.24, 0.2, 0.12);
  } else if (type == CabinetType::kMedium) {
    return Eigen::Vector3d(0.24, 0.22, 0.12);
  } else {
    return Eigen::Vector3d(0.4, 0.26, 0.12);
  }
}
double CabinetFrontWidth() { return 0.03; }

double CabinetThickness() { return 0.01; }

std::vector<Box> Cabinet(CabinetType type) {
  const double thickness = CabinetThickness();
  const auto cabinet_size = CabinetSize(type);
  std::vector<Box> cabinet;
  Eigen::Isometry3d box_pose;

  const Eigen::Vector4d wood_color(0.75, 0.5, 0.25, 1);
  const Eigen::Vector4d cherry_color(0.4, 0.2, 0, 1);
  const Eigen::Vector4d lighter_wood_color(0.6, 0.4, 0, 1);
  Eigen::Vector4d color;
  switch (type) {
    case CabinetType::kSmall: {
      color = wood_color;
      break;
    }
    case CabinetType::kMedium: {
      color = cherry_color;
      break;
    }
    case CabinetType::kLarge: {
      color = lighter_wood_color;
      break;
    }
    default: { throw std::runtime_error("Unknown cabinet type."); }
  }

  // bottom
  box_pose.linear().setIdentity();
  box_pose.translation() << 0, 0, -thickness / 2;
  cabinet.push_back(
      Box(Eigen::Vector3d(cabinet_size[0], cabinet_size[1], thickness),
          box_pose, "bottom", color));

  // top
  box_pose.translation() << 0, 0, cabinet_size[2] + thickness / 2;
  cabinet.push_back(
      Box(Eigen::Vector3d(cabinet_size[0], cabinet_size[1], thickness),
          box_pose, "top", color));
  // left
  box_pose.translation() << -cabinet_size[0] / 2 - thickness / 2, 0,
      cabinet_size[2] / 2;
  cabinet.push_back(Box(Eigen::Vector3d(thickness, cabinet_size[1],
                                        cabinet_size[2] + 2 * thickness),
                        box_pose, "left", color));
  // right
  box_pose.translation() << cabinet_size[0] / 2 + thickness / 2, 0,
      cabinet_size[2] / 2;
  cabinet.push_back(Box(Eigen::Vector3d(thickness, cabinet_size[1],
                                        cabinet_size[2] + 2 * thickness),
                        box_pose, "right", color));

  // back
  box_pose.translation() << 0, -cabinet_size[1] / 2 - thickness / 2,
      cabinet_size[2] / 2;
  cabinet.push_back(
      Box(Eigen::Vector3d(cabinet_size[0] + 2 * thickness, thickness,
                          cabinet_size[2] + 2 * thickness),
          box_pose, "back", color));

  // front
  const double front_width = CabinetFrontWidth();
  box_pose.translation() << cabinet_size[0] / 4,
      cabinet_size[1] / 2 - front_width / 2, cabinet_size[2] / 2;
  cabinet.push_back(
      Box(Eigen::Vector3d(cabinet_size[0] / 2, front_width, cabinet_size[2]),
          box_pose, "front", color));

  return cabinet;
}

std::vector<BodyContactSphere> GetSchunkBodyContactSpheres(
    const RigidBodyTreed& tree) {
  std::vector<BodyContactSphere> points;
  const int schunk_idx = tree.FindBodyIndex("body");
  /*points.emplace_back(schunk_idx, Eigen::Vector3d(-0.057, 0.105, 0), "pt1",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.057, 0.085, 0), "pt2",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.057, 0.065, 0), "pt3",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.057, 0.105, 0), "pt4",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.057, 0.085, 0), "pt5",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.057, 0.065, 0), "pt6",
                      0.008);*/
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.065, 0.03, 0.02), "pt7",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.065, 0.03, 0.02), "pt8",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.065, 0.03, -0.02), "pt9",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.065, 0.03, -0.02), "pt10",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.065, -0.03, 0.02), "pt11",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.065, -0.03, 0.02), "pt12",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.065, -0.03, -0.02), "pt13",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.065, -0.03, -0.02), "pt14",
                      0.008);

  points.emplace_back(schunk_idx, Eigen::Vector3d(0.065, 0, 0.02), "pt15",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.065, 0, 0.02), "pt16",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.065, 0, -0.02), "pt17",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.065, 0, -0.02), "pt18",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.065, -0.015, 0.02), "pt19",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.065, -0.015, 0.02), "pt20",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(0.065, -0.015, -0.02), "pt21",
                      0.008);
  points.emplace_back(schunk_idx, Eigen::Vector3d(-0.065, -0.015, -0.02),
                      "pt22", 0.008);

  return points;
}

std::vector<Box> FreeSpaceBoxes(const Eigen::Vector3d& mug_pos,
                                CabinetType type, bool find_path) {
  std::vector<Box> boxes;

  auto cabinet_size = CabinetSize(type);

  const double mug_radius = 0.037;
  const double min_mug_x = mug_pos(0) - mug_radius;
  Eigen::Isometry3d box_pose;
  box_pose.linear().setIdentity();
  box_pose.translation() << (min_mug_x - cabinet_size[0] / 2) / 2, 0.1,
      cabinet_size[2] / 2;
  boxes.emplace_back(
      Eigen::Vector3d((cabinet_size[0] / 2 + min_mug_x),
                      (box_pose.translation()(1) + cabinet_size[1] / 2) * 2,
                      cabinet_size[2]),
      box_pose, "box1", Eigen::Vector4d(0.1, 0.3, 0.5, 0.3));

  const double max_mug_y = mug_pos(1) + mug_radius;
  box_pose.translation() << 0,
      (cabinet_size[1] / 2 - CabinetFrontWidth() + max_mug_y) / 2,
      cabinet_size[2] / 2;
  boxes.emplace_back(
      Eigen::Vector3d(cabinet_size[0],
                      (cabinet_size[1] / 2 - CabinetFrontWidth() - max_mug_y),
                      cabinet_size[2]),
      box_pose, "box2", Eigen::Vector4d(0.4, 0.1, 0.5, 0.3));

  const double thickness = CabinetThickness();
  if (find_path) {
    box_pose.translation() << -0.05, cabinet_size[1] / 2 + thickness + 0.1,
        cabinet_size[2] / 2;
    boxes.emplace_back(Eigen::Vector3d(cabinet_size[0] + 2 * thickness - 0.1,
                                       0.2, cabinet_size[2]),
                       box_pose, "box3", Eigen::Vector4d(0.1, 0.6, 0.2, 0.3));
  }

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

void AddObjects(RigidBodyTreed* rigid_body_tree,
                const Eigen::Vector3d& mug_pos) {
  const std::string mug_path = FindResourceOrThrow(
      "drake/manipulation/models/objects/coffee_mug/urdf/coffee_mug.urdf");

  auto mug_frame = std::make_shared<RigidBodyFrame<double>>(
      "mug", rigid_body_tree->get_mutable_body(0), mug_pos,
      Eigen::Vector3d(0, 0, 1.1 * M_PI));
  parsers::urdf::AddModelInstanceFromUrdfFile(
      mug_path, drake::multibody::joints::kFixed, mug_frame, rigid_body_tree);
  rigid_body_tree->addFrame(mug_frame);
}

void AddAntiPodalGraspConstraint(solvers::MathematicalProgram* prog,
                                 const Matrix3<symbolic::Variable>& R,
                                 const Vector3<symbolic::Variable>& pos,
                                 const Eigen::Vector3d& mug_pos,
                                 const Eigen::Vector3d& cabinet_size) {
  // x axis of the hand is horizontal.
  prog->AddLinearConstraint(R(2, 0) == 0);

  const Eigen::Vector3d p_BQ(0, 0.09, 0);
  const Eigen::Vector3d mug_center(mug_pos(0), mug_pos(1), mug_pos(2) + 0.05);
  prog->AddLinearEqualityConstraint(pos + R * p_BQ, mug_center);

  // fingertip in the cabinet.
  std::array<Eigen::Vector3d, 2> finger_tips;
  finger_tips[0] << -0.057, 0.105, 0;
  finger_tips[1] << 0.057, 0.105, 0;
  for (const auto& finger_tip : finger_tips) {
    const Vector3<symbolic::Expression> finger_tip_pos = pos + R * finger_tip;
    for (int i = 0; i < 3; ++i) {
      prog->AddLinearConstraint(finger_tip_pos(i) >=
                                -cabinet_size(i) / 2 + 0.01);
      prog->AddLinearConstraint(finger_tip_pos(i) <=
                                cabinet_size(i) / 2 - 0.01);
    }
  }
}

Eigen::VectorXd SolveGlobalIK(
    RigidBodyTreed* tree, const Eigen::Vector3d& mug_pos,
    const std::vector<multibody::GlobalInverseKinematics::Polytope3D>&
        free_space_polytopes,
    const std::vector<BodyContactSphere>& body_contact_spheres,
    CabinetType type) {
  multibody::GlobalInverseKinematics::Options global_ik_options;
  global_ik_options.num_intervals_per_half_axis = 4;
  global_ik_options.linear_constraint_only = false;
  multibody::GlobalInverseKinematics global_ik(*tree, global_ik_options);
  const int schunk_idx = tree->FindBodyIndex("body");
  auto R = global_ik.body_rotation_matrix(schunk_idx);
  auto p = global_ik.body_position(schunk_idx);

  AddAntiPodalGraspConstraint(&global_ik, R, p, mug_pos, CabinetSize(type));
  for (const auto& body_contact_sphere : body_contact_spheres) {
    global_ik.BodySphereInOneOfPolytopes(
        body_contact_sphere.link_idx, body_contact_sphere.p_BQ,
        body_contact_sphere.radius, free_space_polytopes);
  }

  solvers::GurobiSolver gurobi_solver;
  global_ik.SetSolverOption(solvers::GurobiSolver::id(), "OutputFlag", true);

  const auto result = gurobi_solver.Solve(global_ik);
  if (result != solvers::SolutionResult::kSolutionFound) {
    return (Eigen::Matrix<double, 7, 1>() << 0, 0.23, 0.05, 0, 0, 0, 1)
        .finished();
  } else {
    const auto R_sol = global_ik.GetSolution(R);
    const auto p_sol = global_ik.GetSolution(p);
    std::cout << "R:\n"
              << R_sol << "\n"
              << "R' * R:\n"
              << R_sol.transpose() * R_sol << "\n";

    std::cout << "contact sphere positions:\n";
    for (const auto& sphere : body_contact_spheres) {
      std::cout << (p_sol + R_sol * sphere.p_BQ).transpose() << "\n";
    }
    const auto q_reconstruct =
        global_ik.ReconstructGeneralizedPositionSolution(0);
    std::cout << "q_reconstruct: " << q_reconstruct.transpose() << "\n";
    const Eigen::Matrix3d R_reconstruct =
        math::RotationMatrixd(
            Eigen::Quaterniond(q_reconstruct(3), q_reconstruct(4),
                               q_reconstruct(5), q_reconstruct(6)))
            .matrix();
    Eigen::Isometry3d X_reconstruct;
    X_reconstruct.setIdentity();
    X_reconstruct.translation() = q_reconstruct.head<3>();
    X_reconstruct.linear() = R_reconstruct;
    std::cout << "reconstructed pose.\n" << X_reconstruct.matrix() << "\n";
    std::cout << "reconstructed contact sphere positions:\n";
    for (const auto& sphere : body_contact_spheres) {
      const Eigen::Vector3d p_WQ = X_reconstruct * sphere.p_BQ;
      std::cout << p_WQ.transpose() << "\n";
      AddSphereToBody(tree, 0, p_WQ, sphere.name + "_", 0.005);
    }
    return q_reconstruct;
  }
}

std::vector<Eigen::VectorXd> SolvePathGlobalIK(
    const RigidBodyTreed&, const Eigen::Vector3d& mug_pos,
    const std::vector<multibody::GlobalInverseKinematics::Polytope3D>&
        free_space_polytopes,
    const std::vector<BodyContactSphere>& body_contact_spheres,
    CabinetType type) {
  const int nT = 6;
  solvers::MixedIntegerRotationConstraintGenerator rotation_generator(
      solvers::MixedIntegerRotationConstraintGenerator::Approach::
          kBilinearMcCormick,
      2, solvers::IntervalBinning::kLogarithmic);

  solvers::MathematicalProgram prog;
  auto pos = prog.NewContinuousVariables<3, Eigen::Dynamic>(3, nT);
  std::vector<Matrix3<symbolic::Variable>> R(nT);
  const bool linear_constraint_only = true;
  for (int i = 0; i < nT; ++i) {
    R[i] =
        solvers::NewRotationMatrixVars(&prog, "R[" + std::to_string(i) + "]");
    if (!linear_constraint_only) {
      solvers::AddRotationMatrixOrthonormalSocpConstraint(&prog, R[i]);
    }
    rotation_generator.AddToProgram(R[i], &prog);
  }

  // Continuity constraint
  const double pos_diff_tol = 0.025;
  const double angle_diff_tol = 7.0 / 180 * M_PI;
  for (int i = 1; i < nT; ++i) {
    for (int j = 0; j < 3; ++j) {
      prog.AddLinearConstraint(pos(j, i) - pos(j, i - 1) <= pos_diff_tol);
      prog.AddLinearConstraint(pos(j, i) - pos(j, i - 1) >= -pos_diff_tol);
    }
    Eigen::Matrix<symbolic::Expression, 10, 1> R_diff_lorentz;
    R_diff_lorentz << std::sqrt(4 - 4 * std::cos(angle_diff_tol)),
        R[i].col(0) - R[i - 1].col(0), R[i].col(1) - R[i - 1].col(1),
        R[i].col(2) - R[i - 1].col(2);
    prog.AddLorentzConeConstraint(R_diff_lorentz);
  }

  // Antipodal grasp in the end.
  AddAntiPodalGraspConstraint(&prog, R[nT - 1], pos.col(nT - 1), mug_pos,
                              CabinetSize(type));

  // Collision avoidance
  for (int i = 0; i < nT; ++i) {
    for (const auto& body_contact_sphere : body_contact_spheres) {
      BodySphereInOneOfPolytopes(
          &prog, R[i], pos.col(i), body_contact_sphere.p_BQ,
          body_contact_sphere.radius, free_space_polytopes);
    }
  }

  // initially schunk is outside the box.
  prog.AddLinearConstraint(pos(1, 0) >= CabinetSize(type)(1) / 2 + 0.02);

  prog.SetSolverOption(solvers::GurobiSolver::id(), "OutputFlag", true);

  solvers::GurobiSolver solver;
  const auto result = solver.Solve(prog);
  if (result != solvers::SolutionResult::kSolutionFound) {
    Eigen::VectorXd q(7);
    q << -0.045, 0.23, 0.05, 0, 0, 0, 1;
    return {q};
  }
  std::vector<Eigen::VectorXd> q(nT);
  for (int i = 0; i < nT; ++i) {
    q[i].resize(7);
    q[i].head<3>() = prog.GetSolution(pos.col(i));
    const auto R_sol = prog.GetSolution(R[i]);
    const math::RotationMatrixd R_proj =
        math::RotationMatrixd::ProjectToRotationMatrix(R_sol);
    q[i].tail<4>() = R_proj.ToQuaternionAsVector4();
  }
  return q;
}

int DoMain(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Simple sdformat usage example, just"
      "make sure drake-visualizer is running!");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CabinetType cabinet_type;
  if (FLAGS_cabinet_size == "small") {
    cabinet_type = CabinetType::kSmall;
  } else if (FLAGS_cabinet_size == "medium") {
    cabinet_type = CabinetType::kMedium;
  } else if (FLAGS_cabinet_size == "large") {
    cabinet_type = CabinetType::kLarge;
  } else {
    throw std::runtime_error("Unsupported cabinet size.");
  }
  drake::lcm::DrakeLcm lcm;
  auto tree = ConstructSchunkGripper();
  const Eigen::Vector3d mug_pos(0.035, -0.04, 0);

  const std::vector<Box> cabinet = Cabinet(cabinet_type);
  const std::vector<BodyContactSphere> body_contact_spheres =
      GetSchunkBodyContactSpheres(*tree);
  const std::vector<Box> free_space_boxes =
      FreeSpaceBoxes(mug_pos, cabinet_type, FLAGS_find_path);
  const auto free_space_polytopes = SetFreeSpace(free_space_boxes);

  Eigen::Matrix<double, 7, 1> q_visualize;
  q_visualize.head<3>() << 0, 0.2, 0;
  q_visualize.tail<4>() << 1, 0, 0, 0;

  std::vector<Eigen::VectorXd> q_path;
  if (!FLAGS_find_path) {
    q_visualize = SolveGlobalIK(tree.get(), mug_pos, free_space_polytopes,
                                body_contact_spheres, cabinet_type);
    q_visualize(1) += 0.05;
  } else {
    q_path = SolvePathGlobalIK(*tree, mug_pos, free_space_polytopes,
                               body_contact_spheres, cabinet_type);
    for (auto& q_path_i : q_path) {
      q_path_i(1) += 0.05;
    }
  }

  AddObjects(tree.get(), mug_pos);
  for (const auto& box : cabinet) {
    AddBoxToTree(tree.get(), box.size, box.pose, box.name, box.color);
  }
  for (const auto& box : free_space_boxes) {
    AddBoxToTree(tree.get(), box.size, box.pose, box.name, box.color);
  }
  for (const auto& sphere : body_contact_spheres) {
    AddSphereToBody(tree.get(), sphere.link_idx, sphere.p_BQ, sphere.name,
                    std::max(sphere.radius, 0.005));
  }

  manipulation::SimpleTreeVisualizer simple_tree_visualizer(*tree.get(), &lcm);
  std::cout << "q_visualize: " << q_visualize.transpose() << "\n";
  if (!FLAGS_find_path) {
    simple_tree_visualizer.visualize(q_visualize);
  } else {
    for (const auto& q : q_path) {
      simple_tree_visualizer.visualize(q);
      getchar();
    }
  }

  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::examples::kuka_iiwa_arm::DoMain(argc, argv);
}
