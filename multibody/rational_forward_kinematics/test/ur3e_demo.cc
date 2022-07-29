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
#include "drake/multibody/tree/multibody_tree_indexes.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/revolute_mobilizer.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solution_result.h"
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

class Ur3Diagram {
 public:
  Ur3Diagram() : meshcat_{std::make_shared<geometry::Meshcat>()} {
    systems::DiagramBuilder<double> builder;
    auto [plant, sg] = AddMultibodyPlantSceneGraph(&builder, 0.);
    plant_ = &plant;
    scene_graph_ = &sg;

    multibody::Parser parser(plant_);
    const std::string ur3e_file_path = FindResourceOrThrow(
        "drake/manipulation/models/ur3e_description/urdf/"
        "ur3e_robot_primitive_collision.urdf");
    ur_ = parser.AddModelFromFile(ur3e_file_path, "ur");
    plant_->WeldFrames(plant_->world_frame(),
                       plant_->GetFrameByName("ur_base_link", ur_));

    // Modify the joint limits to strictly within [-pi, pi]
    for (const auto& joint_index : plant_->GetJointIndices(ur_)) {
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
        joint.set_position_limits(position_lower_limits, position_upper_limits);
      }
    }

    const std::string schunk_file_path = FindResourceOrThrow(
        "drake/manipulation/models/wsg_50_description/sdf/"
        "schunk_wsg_50_welded_fingers.sdf");
    const Frame<double>& ur_tool0 = plant_->GetFrameByName("ur_tool0", ur_);
    const math::RigidTransformd X_TS(
        math::RollPitchYaw<double>(M_PI / 2, 0, 67.5 / 180 * M_PI),
        Eigen::Vector3d(0, 0, 0.05));
    const auto wsg_instance =
        parser.AddModelFromFile(schunk_file_path, "gripper");
    const auto& schunk_frame = plant_->GetFrameByName("body", wsg_instance);
    plant_->WeldFrames(ur_tool0, schunk_frame, X_TS);

    // Add shelf.
    const std::string shelf_file_path = FindResourceOrThrow(
        "drake/multibody/rational_forward_kinematics/models/shelves.sdf");
    const auto shelf_instance =
        parser.AddModelFromFile(shelf_file_path, "shelves");
    const auto& shelf_frame =
        plant_->GetFrameByName("shelves_body", shelf_instance);
    const math::RigidTransformd X_WShelf(Eigen::Vector3d(0.5, 0, 0.2));
    plant_->WeldFrames(plant_->world_frame(), shelf_frame, X_WShelf);

    // Set collision filters. Each UR3 robot should ignore the collision with
    // itself.
    const auto ur_geometry_ids = CollectGeometries(*plant_, ur_);
    scene_graph_->collision_filter_manager().Apply(
        geometry::CollisionFilterDeclaration().ExcludeWithin(
            geometry::GeometrySet(ur_geometry_ids)));

    // SceneGraph should ignore the collision between any geometries on the
    // gripper, and between the gripper and wrist
    geometry::GeometrySet gripper_wrist_geometries;
    auto add_gripper_geometries = [this, &gripper_wrist_geometries](
                                      const std::string& body_name,
                                      ModelInstanceIndex model_instance) {
      const geometry::FrameId frame_id = plant_->GetBodyFrameIdOrThrow(
          plant_->GetBodyByName(body_name, model_instance).index());
      gripper_wrist_geometries.Add(frame_id);
    };
    add_gripper_geometries("body", wsg_instance);
    add_gripper_geometries("left_finger", wsg_instance);
    add_gripper_geometries("right_finger", wsg_instance);
    add_gripper_geometries("ur_wrist_1_link", ur_);
    add_gripper_geometries("ur_wrist_2_link", ur_);
    add_gripper_geometries("ur_wrist_3_link", ur_);

    scene_graph_->collision_filter_manager().Apply(
        geometry::CollisionFilterDeclaration().ExcludeWithin(
            gripper_wrist_geometries));

    plant_->Finalize();
    geometry::MeshcatVisualizerParams meshcat_params{};
    meshcat_params.role = geometry::Role::kProximity;
    visualizer_ = &geometry::MeshcatVisualizer<double>::AddToBuilder(
        &builder, *scene_graph_, meshcat_, meshcat_params);
    diagram_ = builder.Build();
  };

  const systems::Diagram<double>& diagram() const { return *diagram_; }

  const multibody::MultibodyPlant<double>& plant() const { return *plant_; }

  const geometry::SceneGraph<double>& scene_graph() const {
    return *scene_graph_;
  }

  ModelInstanceIndex ur() const { return ur_; }

 private:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  multibody::MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  std::shared_ptr<geometry::Meshcat> meshcat_;
  geometry::MeshcatVisualizer<double>* visualizer_;
  ModelInstanceIndex ur_;
};

Eigen::VectorXd FindInitialPosture(const MultibodyPlant<double>& plant,
                                   ModelInstanceIndex ur,
                                   systems::Context<double>* plant_context) {
  InverseKinematics ik(plant, plant_context);
  unused(ur);
  const auto& wrist_3_link = plant.GetFrameByName("ur_wrist_3_link", ur);

  const auto& shelf = plant.GetFrameByName("shelves_body");
  ik.AddPositionConstraint(wrist_3_link, Eigen::Vector3d::Zero(), shelf,
                           Eigen::Vector3d(-0.2, -0.2, -0.2),
                           Eigen::Vector3d(0., 0.2, 0.2));

  ik.AddMinimumDistanceConstraint(0.02);

  Eigen::Matrix<double, 6, 1> q_init;
  q_init << 0.1, 0.2, 0.1, -0.1, 0.2, -0.1;
  ik.get_mutable_prog()->SetInitialGuess(ik.q(), q_init);
  const auto result = solvers::Solve(ik.prog());
  if (!result.is_success()) {
    drake::log()->warn("Cannot find the posture\n");
  }
  return result.GetSolution(ik.q());
}

void BuildCandidateCspacePolytope(const Eigen::VectorXd& q_free,
                                  Eigen::MatrixXd* C, Eigen::VectorXd* d) {
  const int C_rows = 14;
  C->resize(C_rows, 6);
  // Create arbitrary polytope normals.
  // clang-format off
  (*C) << 3.1, 0.3, 1, 0.2, -0.1, 0.5,
          -3.5, -0.1, 1.5, 0.2, -0.1, 0.4,
          0.5, 4.1, -0.3, -0.2, -2.1, -0.1,
          -0.2, -3.8, 0.3, -3.1, -0.5, -1.2,
          0.2, -1.2, 3.4, 1.5, -0.1, 0.5,
          0.1, -0.2, -2.9, -0.5, 1.5, 0.7,
          0.2, -0.1, 0.3, 5.3, 0.5, 0.8,
          -0.3, 1.2, 0.8, -4.5, -1.1, 0.3,
          1.2, -0.3, 0.15, 0.4, 4.6, 0.7,
          0.2, -0.4, 0.5, -0.3, -5.5, -0.5,
          0.5, -0.3, 1.5, 1.9, 2.1, -4.4,
          -0.5, 1.2, 0.8, -0.6, 0.3, 3.4,
          1.5, -0.4, 0.8, -2.1, 2.5, 0.3,
          0.4, 1.5, 0.7, 1.5, -2.1, 0.5;
  // clang-format on
  for (int i = 0; i < C_rows; ++i) {
    C->row(i).normalize();
  }
  *d = (*C) * (q_free / 2).array().tan().matrix() +
       0.0001 * Eigen::VectorXd::Ones(C_rows);
  if (!geometry::optimization::HPolyhedron(*C, *d).IsBounded()) {
    for (int i = 0; i < C->cols(); ++i) {
      solvers::MathematicalProgram prog;
      auto t = prog.NewContinuousVariables(C->cols());
      prog.AddLinearConstraint(*C, Eigen::VectorXd::Constant(d->rows(), -kInf),
                               *d, t);
      auto cost =
          prog.AddLinearCost(Vector1d(1), 0, Vector1<symbolic::Variable>(t(i)));
      auto result = solvers::Solve(prog);
      if (result.get_solution_result() == solvers::SolutionResult::kUnbounded ||
          result.get_solution_result() ==
              solvers::SolutionResult::kInfeasibleOrUnbounded ||
          result.get_solution_result() ==
              solvers::SolutionResult::kDualInfeasible) {
        std::cout << "t(" << i << ") can be -inf\n";
      }
      cost.evaluator()->UpdateCoefficients(Vector1d(-1), 0);
      result = solvers::Solve(prog);
      if (result.get_solution_result() == solvers::SolutionResult::kUnbounded ||
          result.get_solution_result() ==
              solvers::SolutionResult::kInfeasibleOrUnbounded ||
          result.get_solution_result() ==
              solvers::SolutionResult::kDualInfeasible) {
        std::cout << "t(" << i << ") can be inf\n";
      }
    }
    throw std::runtime_error("C*t <= d is not bounded");
  }
}

void SearchCspacePolytope(const std::string& write_file) {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();

  const Ur3Diagram ur_diagram{};
  auto diagram_context = ur_diagram.diagram().CreateDefaultContext();
  auto& plant_context =
      ur_diagram.plant().GetMyMutableContextFromRoot(diagram_context.get());
  const auto q0 =
      FindInitialPosture(ur_diagram.plant(), ur_diagram.ur(), &plant_context);
  ur_diagram.plant().SetPositions(&plant_context, q0);
  ur_diagram.diagram().Publish(*diagram_context);

  Eigen::MatrixXd C_init;
  Eigen::VectorXd d_init;
  BuildCandidateCspacePolytope(q0, &C_init, &d_init);

  const double separating_polytope_delta{0.01};
  const CspaceFreeRegion dut(
      ur_diagram.diagram(), &(ur_diagram.plant()), &(ur_diagram.scene_graph()),
      SeparatingPlaneOrder::kConstant, CspaceRegionType::kGenericPolytope,
      separating_polytope_delta);
  const Eigen::VectorXd q_star = Eigen::Matrix<double, 6, 1>::Zero();
  const Eigen::VectorXd t0 =
      dut.rational_forward_kinematics().ComputeTValue(q0, q_star);

  CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{};

  CspaceFreeRegion::BinarySearchOption binary_search_option{
      .epsilon_max = 0.005,
      .epsilon_min = 1E-6,
      .max_iters = 2,
      .compute_polytope_volume = false};
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, false);
  CspaceFreeRegionSolution cspace_free_region_solution;
  dut.CspacePolytopeBinarySearch(
      q_star, filtered_collision_pairs, C_init, d_init, binary_search_option,
      solver_options, t0, std::nullopt, &cspace_free_region_solution);
  CspaceFreeRegion::BilinearAlternationOption bilinear_alternation_option{
      .max_iters = 20,
      .convergence_tol = 0.001,
      .redundant_tighten = 0.5,
      .compute_polytope_volume = false};
  std::vector<double> polytope_volumes;
  std::vector<double> ellipsoid_determinants;
  dut.CspacePolytopeBilinearAlternation(
      q_star, filtered_collision_pairs, cspace_free_region_solution.C,
      cspace_free_region_solution.d, bilinear_alternation_option,
      solver_options, t0, std::nullopt, &cspace_free_region_solution,
      &polytope_volumes, &ellipsoid_determinants);

  const Eigen::VectorXd t_upper =
      (ur_diagram.plant().GetPositionUpperLimits() / 2).array().tan().matrix();
  const Eigen::VectorXd t_lower =
      (ur_diagram.plant().GetPositionLowerLimits() / 2).array().tan().matrix();
  WriteCspacePolytopeToFile(cspace_free_region_solution, ur_diagram.plant(),
                            ur_diagram.scene_graph().model_inspector(),
                            write_file, 10);
}

int DoMain() {
  const std::string cspace_polytope_file = "ur_cspace_polytope.txt";
  SearchCspacePolytope(cspace_polytope_file);
  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
