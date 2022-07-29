#include <iostream>
#include <limits>

#include "drake/common/find_resource.h"
#include "drake/geometry/collision_filter_declaration.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/polytope_cover.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/revolute_mobilizer.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
const double kInf = std::numeric_limits<double>::infinity();

std::vector<geometry::GeometryId> CollectGeometries(
    const MultibodyPlant<double>& plant, ModelInstanceIndex instance) {
  std::vector<geometry::GeometryId> geometry_ids;
  for (const auto& body_index : plant.GetBodyIndices(instance)) {
    const auto body_geometry_ids =
        plant.GetCollisionGeometriesForBody(plant.get_body(body_index));
    geometry_ids.insert(geometry_ids.end(), body_geometry_ids.begin(),
                        body_geometry_ids.end());
  }
  return geometry_ids;
}

class DualUr3Diagram {
 public:
  DualUr3Diagram() : meshcat_{std::make_shared<geometry::Meshcat>()} {
    systems::DiagramBuilder<double> builder;
    auto [plant, sg] = AddMultibodyPlantSceneGraph(&builder, 0.);
    plant_ = &plant;
    scene_graph_ = &sg;

    multibody::Parser parser(plant_);
    const std::string ur3e_file_path = FindResourceOrThrow(
        "drake/manipulation/models/ur3e_description/urdf/"
        "ur3e_robot_primitive_collision.urdf");
    left_ur_ = parser.AddModelFromFile(ur3e_file_path, "left_ur");
    const math::RigidTransformd X_WL(Eigen::Vector3d(0, 0.3, 0));
    plant_->WeldFrames(plant_->world_frame(),
                       plant_->GetFrameByName("ur_base_link", left_ur_), X_WL);

    right_ur_ = parser.AddModelFromFile(ur3e_file_path, "right_ur");
    const math::RigidTransformd X_WR(Eigen::Vector3d(0, -0.3, 0));
    plant_->WeldFrames(plant_->world_frame(),
                       plant_->GetFrameByName("ur_base_link", right_ur_), X_WR);
    // Modify the joint limits to strictly within [-pi, pi]
    for (const auto& model_instance : {left_ur_, right_ur_}) {
      for (const auto& joint_index : plant_->GetJointIndices(model_instance)) {
        auto& joint = plant_->get_mutable_joint(joint_index);
        if (dynamic_cast<multibody::RevoluteJoint<double>*>(&joint)) {
          Eigen::VectorXd position_lower_limits = joint.position_lower_limits();
          if (position_lower_limits(0) <= -M_PI) {
            position_lower_limits(0) = -0.95 * M_PI;
          }
          Eigen::VectorXd position_upper_limits = joint.position_upper_limits();
          if (position_upper_limits(0) >= M_PI) {
            position_upper_limits(0) = 0.95 * M_PI;
          }
          joint.set_position_limits(position_lower_limits,
                                    position_upper_limits);
        }
      }
    }

    // TODO(hongkai.dai): add gripper.

    // Set collision filters. Each UR3 robot should ignore the collision with
    // itself.
    const auto left_ur_geometry_ids = CollectGeometries(*plant_, left_ur_);
    const auto right_ur_geometry_ids = CollectGeometries(*plant_, right_ur_);
    scene_graph_->collision_filter_manager().Apply(
        geometry::CollisionFilterDeclaration().ExcludeWithin(
            geometry::GeometrySet(left_ur_geometry_ids)));
    scene_graph_->collision_filter_manager().Apply(
        geometry::CollisionFilterDeclaration().ExcludeWithin(
            geometry::GeometrySet(right_ur_geometry_ids)));

    plant_->Finalize();
    geometry::MeshcatVisualizerParams meshcat_params{};
    meshcat_params.role = geometry::Role::kProximity;
    meshcat_->SetProperty("/drake/visualizer/left_ur/ur_base_link/21", "top_color", {1., 0., 0.});
    visualizer_ = &geometry::MeshcatVisualizer<double>::AddToBuilder(
        &builder, *scene_graph_, meshcat_, meshcat_params);
    diagram_ = builder.Build();
  };

  const systems::Diagram<double>& diagram() const { return *diagram_; }

  const multibody::MultibodyPlant<double>& plant() const { return *plant_; }

  const geometry::SceneGraph<double>& scene_graph() const {
    return *scene_graph_;
  }

  ModelInstanceIndex left_ur() const { return left_ur_; }

  ModelInstanceIndex right_ur() const { return right_ur_; }

 private:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  multibody::MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  std::shared_ptr<geometry::Meshcat> meshcat_;
  geometry::MeshcatVisualizer<double>* visualizer_;
  ModelInstanceIndex left_ur_;
  ModelInstanceIndex right_ur_;
};

Eigen::VectorXd FindInitialPosture(const MultibodyPlant<double>& plant,
                                   ModelInstanceIndex left_ur,
                                   ModelInstanceIndex right_ur,
                                   systems::Context<double>* plant_context) {
  InverseKinematics ik(plant, plant_context);
  const auto& left_wrist_3_link =
      plant.GetFrameByName("ur_wrist_3_link", left_ur);
  const auto& right_wrist_3_link =
      plant.GetFrameByName("ur_wrist_3_link", right_ur);

  ik.AddPositionConstraint(
      left_wrist_3_link, Eigen::Vector3d::Zero(), plant.world_frame(),
      Eigen::Vector3d::Constant(-kInf), Eigen::Vector3d(kInf, 0, kInf));
  ik.AddPositionConstraint(
      right_wrist_3_link, Eigen::Vector3d::Zero(), plant.world_frame(),
      Eigen::Vector3d(-kInf, 0, -kInf), Eigen::Vector3d::Constant(kInf));

  ik.AddMinimumDistanceConstraint(0.03);

  Eigen::Matrix<double, 12, 1> q_init;
  q_init << 0.1, 0.2, 0.1, -0.1, 0.2, -0.1, 0.1, -0.2, 0.2, -0.1, 0.1, -0.1;
  ik.get_mutable_prog()->SetInitialGuess(ik.q(), q_init);
  const auto result = solvers::Solve(ik.prog());
  if (!result.is_success()) {
    drake::log()->warn("Cannot find the posture\n");
  }
  return result.GetSolution(ik.q());
}

void BuildCandidateCspacePolytope(const Eigen::VectorXd& q_free,
                                  Eigen::MatrixXd* C, Eigen::VectorXd* d) {
  const int C_rows = 32;
  C->resize(C_rows, 12);
  // Create arbitrary polytope normals.
  // clang-format off
  (*C) << -2.1, 0.3, 1, 0.2, -0.1, 0.5, 0.1, -0.2, 0.15, 0.25, 0.15, -0.1,
          -3.5, -0.1, 1.5, 0.2, -0.1, 0.4, -0.2, 1.2, .3, -0.2, -0.1, 0.5,
          0.5, 4.1, -0.3, -0.2, -2.1, -0.1, 1.2, -0.2, 0.3, -0.1, 0.1, -0.1,
          -0.2, -3.8, 0.3, -3.1, -0.5, -1.2, 0.5, -2.1, -0.1, 0.5, -1.2, 0.3,
          0.2, -1.2, 3.4, 1.5, -0.1, 0.5, -0.5, -1.2, 0.4, 0.6, -0.1, 1.5,
          0.1, -0.2, -2.9, -0.5, 1.5, 0.7, -0.4, 0.5, 1.7, 1.2, -0.2, 0.4,
          0.2, -0.1, 0.3, 0.3, 0.5, 1.8, 1.9, -0.1, 0.4, -1.5, 0.3, 0.7,
          -0.3, 1.2, 0.8, -0.5, -1.1, 1.3, -0.2, 1.5, -0.3, -1.2, 1.5, 0.8,
          1.2, -0.3, 0.15, 0.4, -0.6, 0.7, 1.8, -0.4, 0.2, 2.1, -0.5, -0.3,
          1.2, -0.4, 0.5, -0.3, -0.5, -1.5, 1.8, 0.2, -1.9, 0.3, 2.1, -0.5,
          0.5, -0.3, 1.5, 1.9, 2.1, -0.4, -2.2, -0.1, -0.5, -1.5, 0.7, -0.2,
          -0.5, 1.2, 1.8, -0.6, 0.3, 0.4, -0.6, 1.8, -0.3, 1.1, 0.6, 1.5,
          1.5, -0.4, 0.8, -2.1, 2.5, 0.3, -1.5, 0.3, 2.8, -0.3, -0.9, -1.1,
          0.4, 1.5, 0.7, 1.5, -2.1, 0.5, 2.1, -0.4, 1.5, 0.8, -0.2, 2.0,
          -0.1, 0.5, 1.3, -2.1, 0.5, -2.2, 0.7, 1.6, 0.9, -2.1, 0.3, -0.8,
          0.4, 1.5, -2.1, 0.3, 1.5, -0.2, 0.8, 1.6, -0.5, 1.5, -0.4, 0.2,
          0.1, 1.2, -0.7, 1.5, 2.1, -0.4, 1.3, -2.1, -0.4, -0.9, 1.5, -0.3,
          0.2, -2.2, -0.8, -4.3, 0.1, 0.8, 1.2, 0.4, -2.5, 0.7, -0.2, 1.1,
          0.4, 1.5, 0.3, -2.1, 0.4, 1.8, 0.7, 1.6, 0.4, -0.5, -0.8, -2.5,
          1.1, -2.1, 0.5, 1.5, -0.3, 0.8, -0.4, 1.6, -2.2, 0.4, 0.3, 1.1,
          0.1, 0.8, -2.1, 0.5, 1.2, -0.3, 0.6, -2.1, 0.5, 1.5, 0.7, -0.4,
          0.4, 0.7, 1.5, -2.1, 0.3, -2.2, 0.4, -0.5, -0.2, -0.3, 1.2, -0.3,
          -0.2, -2.1, 0.4, 2.1, 0.2, 0.5, -0.4, -0.5, 2.2, -1.5, 0.3, 0.8,
          0.4, 0.8, 0.3, -2.1, 0.5, 1.5, -0.3, -0.9, 1.3, 0.2, 0.7, -2.1,
          0.1, -1.5, 0.3, -0.2, -0.5, 1.8, 1.4, -0.2, 1.9, -2.1, 0.3, 0.8,
          0.4, -2.2, -3.4, 0.1, -1.2, 0.5, 0.8, -0.3, 1.2, 2.1, -0.5, 0.7,
          0.5, 0.3, -1.2, -0.4, 1.5, 1.8, -0.2, 1.5, 0.5, -2.1, 0.4, 0.5,
          0.3, -0.4, -5.2, 0.1, 0.5, 0.8, 1.7, 2.1, -3.2, 0.5, 3.2, -1.4,
          0.8, 1.2, -0.2, -0.5, 1.5, -0.3, -0.2, -2.1, 0.4, -2.8, 0.5, 1.4,
          0.5, -1.5, -2.1, 0.4, 0.5, -1.2, 0.8, -2.1, 0.5, -2.2, 1.5, 0.4,
          0.7, -2.1, 2.5, 0.6, 1.5, -2.1, 0.9, 1.5, -0.4, 1.5, 0.2, 0.3,
          1, 0.1, -0.2, 0.1, 0.2, -0.2, 0.1, 0.3, 0.2, -0.2, 0.2, 0.1;
  // clang-format on
  for (int i = 0; i < C_rows; ++i) {
    C->row(i).normalize();
  }
  *d = (*C) * (q_free / 2).array().tan().matrix() +
       0.0001 * Eigen::VectorXd::Ones(C_rows);
  if (!geometry::optimization::HPolyhedron(*C, *d).IsBounded()) {
    throw std::runtime_error("C*t <= d is not bounded");
  }
}

void SearchCspacePolytope(
    const std::string& write_file,
    const std::optional<std::string>& read_file = std::nullopt,
    bool do_bisection = true) {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();

  const DualUr3Diagram dual_ur_diagram{};
  auto diagram_context = dual_ur_diagram.diagram().CreateDefaultContext();
  auto& plant_context = dual_ur_diagram.plant().GetMyMutableContextFromRoot(
      diagram_context.get());
  const auto q0 =
      FindInitialPosture(dual_ur_diagram.plant(), dual_ur_diagram.left_ur(),
                         dual_ur_diagram.right_ur(), &plant_context);
  dual_ur_diagram.plant().SetPositions(&plant_context, q0);
  dual_ur_diagram.diagram().Publish(*diagram_context);
  std::cout << "Type to continue\n";
  std::cin.get();

  Eigen::MatrixXd C_init;
  Eigen::VectorXd d_init;
  std::unordered_map<SortedPair<geometry::GeometryId>,
                     std::pair<BodyIndex, Eigen::VectorXd>>
      separating_plane_init;
  if (read_file.has_value()) {
    ReadCspacePolytopeFromFile(read_file.value(), dual_ur_diagram.plant(),
                               dual_ur_diagram.scene_graph().model_inspector(),
                               &C_init, &d_init, &separating_plane_init);
  } else {
    BuildCandidateCspacePolytope(q0, &C_init, &d_init);
  }

  const double separating_polytope_delta{0.01};
  const CspaceFreeRegion dut(
      dual_ur_diagram.diagram(), &(dual_ur_diagram.plant()),
      &(dual_ur_diagram.scene_graph()), SeparatingPlaneOrder::kAffine,
      CspaceRegionType::kGenericPolytope, separating_polytope_delta);

  CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{};

  CspaceFreeRegion::BinarySearchOption binary_search_option{
      .epsilon_max = 0.2,
      .epsilon_min = 1E-6,
      .max_iters = 2,
      .compute_polytope_volume = false,
      .num_threads = 20};
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, false);
  CspaceFreeRegionSolution cspace_free_region_solution;
  Eigen::VectorXd q_star = Eigen::Matrix<double, 12, 1>::Zero();
  const Eigen::VectorXd t0 =
      dut.rational_forward_kinematics().ComputeTValue(q0, q_star);
  if (do_bisection) {
    drake::log()->info("Start bisection");
    dut.CspacePolytopeBinarySearch(
        q_star, filtered_collision_pairs, C_init, d_init, binary_search_option,
        solver_options, t0, std::nullopt, &cspace_free_region_solution);
  }
  CspaceFreeRegion::BilinearAlternationOption bilinear_alternation_option{
      .max_iters = 10,
      .convergence_tol = 0.000,
      .lagrangian_backoff_scale = 0.001,
      .redundant_tighten = 0.5,
      .compute_polytope_volume = false,
      .num_threads = 20};
  Eigen::MatrixXd C0;
  Eigen::VectorXd d0;
  if (do_bisection) {
    C0 = cspace_free_region_solution.C;
    d0 = cspace_free_region_solution.d;
  } else {
    C0 = C_init;
    d0 = d_init;
  }
  drake::log()->info("Start bilinear alternation");
  std::vector<double> polytope_volumes;
  std::vector<double> ellipsoid_determinants;
  dut.CspacePolytopeBilinearAlternation(
      q_star, filtered_collision_pairs, C0, d0, bilinear_alternation_option,
      solver_options, t0, std::nullopt, &cspace_free_region_solution,
      &polytope_volumes, &ellipsoid_determinants);

  const Eigen::VectorXd t_upper =
      (dual_ur_diagram.plant().GetPositionUpperLimits() / 2)
          .array()
          .tan()
          .matrix();
  const Eigen::VectorXd t_lower =
      (dual_ur_diagram.plant().GetPositionLowerLimits() / 2)
          .array()
          .tan()
          .matrix();
  WriteCspacePolytopeToFile(
      cspace_free_region_solution, dual_ur_diagram.plant(),
      dual_ur_diagram.scene_graph().model_inspector(), write_file, 10);
}

void VisualizaPostures(const std::string& cspace_polytope_file) {
  const DualUr3Diagram dual_ur_diagram{};
  auto diagram_context = dual_ur_diagram.diagram().CreateDefaultContext();
  auto& plant_context = dual_ur_diagram.plant().GetMyMutableContextFromRoot(
      diagram_context.get());
  Eigen::MatrixXd C;
  Eigen::VectorXd d;
  std::unordered_map<SortedPair<geometry::GeometryId>,
                     std::pair<BodyIndex, Eigen::VectorXd>>
      separating_planes;
  ReadCspacePolytopeFromFile(cspace_polytope_file, dual_ur_diagram.plant(),
                             dual_ur_diagram.scene_graph().model_inspector(),
                             &C, &d, &separating_planes);

  solvers::MathematicalProgram prog;
  auto t1 = prog.NewContinuousVariables(12);
  auto t2 = prog.NewContinuousVariables(12);
  prog.AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf), d,
                           t1);
  prog.AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf), d,
                           t2);
  // prog.AddBoundingBoxConstraint(t_lower, t_upper, t1);
  // prog.AddBoundingBoxConstraint(t_lower, t_upper, t2);
  // Add the cost max (t1-t2)^2 = (A*[t1;t2])^2;
  Eigen::Matrix<double, 12, 24> A;
  A << Eigen::Matrix<double, 12, 12>::Identity(),
      -Eigen::Matrix<double, 12, 12>::Identity();
  prog.AddQuadraticCost(-A.transpose() * A, Eigen::VectorXd::Zero(24), {t1, t2},
                        false /*is_convex=false*/);
  prog.SetInitialGuess(t1, Eigen::VectorXd::Ones(12));
  prog.SetInitialGuess(t2, -Eigen::VectorXd::Ones(12));
  const auto result = solvers::Solve(prog);
  DRAKE_DEMAND(result.is_success());
  std::cout << result.get_optimal_cost() << "\n";
  const Eigen::VectorXd t1_val = result.GetSolution(t1);
  const Eigen::VectorXd t2_val = result.GetSolution(t2);
  const Eigen::MatrixXd q1_val = 2 * t1_val.array().atan().matrix();
  const Eigen::MatrixXd q2_val = 2 * t2_val.array().atan().matrix();
  std::cout << q1_val.transpose() << "\n";
  std::cout << q2_val.transpose() << "\n";
  dual_ur_diagram.plant().SetPositions(&plant_context, q1_val);
  dual_ur_diagram.diagram().Publish(*diagram_context);
  std::cout << fmt::format("Posture 1, type to continue\n");
  std::cin.get();
  dual_ur_diagram.plant().SetPositions(&plant_context, q2_val);
  dual_ur_diagram.diagram().Publish(*diagram_context);
  std::cin.get();
  std::cin.get();
}

int DoMain() {
  const std::string cspace_polytope_read_file = "dual_ur_cspace_polytope.txt";
  const std::string cspace_polytope_write_file = "dual_ur_cspace_polytope1.txt";
  //SearchCspacePolytope(cspace_polytope_write_file);
  VisualizaPostures(cspace_polytope_read_file);
  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
