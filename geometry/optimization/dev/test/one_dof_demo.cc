#include "drake/geometry/geometry_roles.h"
#include "drake/geometry/optimization/dev/cspace_free_polytope.h"
#include "drake/geometry/optimization/dev/separating_plane.h"
#include "drake/geometry/optimization/iris.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/plant/coulomb_friction.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/rational/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics_old/cspace_free_region.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace geometry {
namespace optimization {
class OneDofDiagram {
 public:
  OneDofDiagram() {
    systems::DiagramBuilder<double> builder;
    std::tie(plant_, scene_graph_) =
        multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);

    ProximityProperties proximity_properties{};
    AddContactMaterial(0.1, 250, multibody::CoulombFriction<double>{0.9, 0.5},
                       &proximity_properties);

    world_box_ = plant_->RegisterCollisionGeometry(
        plant_->world_body(), math::RigidTransformd(Eigen::Vector3d(2, 0, 0)),
        geometry::Box(1, 1, 5), "world_box", proximity_properties);

    const multibody::SpatialInertia<double> spatial_inertia(
        1, Eigen::Vector3d::Zero(),
        multibody::UnitInertia<double>(0.01, 0.01, 0.01, 0, 0, 0));
    const auto body0_index =
        plant_->AddRigidBody("body0", spatial_inertia).index();
    body0_box_ = plant_->RegisterCollisionGeometry(
        plant_->get_body(body0_index),
        math::RigidTransformd(Eigen::Vector3d(0, 0, 2)), geometry::Box(1, 1, 1),
        "body0_box", proximity_properties);
    const auto& joint0 = plant_->AddJoint<multibody::RevoluteJoint>(
        "joint0", plant_->world_body(), math::RigidTransformd(),
        plant_->get_body(body0_index), math::RigidTransformd(),
        Eigen::Vector3d::UnitY());
    plant_->get_mutable_joint(joint0.index())
        .set_position_limits(Vector1d(-3), Vector1d(3));
    plant_->Finalize();
    diagram_ = builder.Build();
  }

  [[nodiscard]] const systems::Diagram<double>& diagram() const {
    return *diagram_;
  }

  [[nodiscard]] const multibody::MultibodyPlant<double>& plant() const {
    return *plant_;
  }

  [[nodiscard]] const geometry::SceneGraph<double>& scene_graph() const {
    return *scene_graph_;
  }

  [[nodiscard]] geometry::GeometryId world_box() const { return world_box_; }

  [[nodiscard]] geometry::GeometryId body0_box() const { return body0_box_; }

 private:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  multibody::MultibodyPlant<double>* plant_;
  geometry::SceneGraph<double>* scene_graph_;
  geometry::GeometryId world_box_;
  geometry::GeometryId body0_box_;
};

HPolyhedron FindIrisRegion(const multibody::MultibodyPlant<double>& plant,
                           const Eigen::VectorXd& q_seed,
                           const Eigen::VectorXd& q_star,
                           systems::Context<double>* plant_context) {
  IrisOptions iris_options;
  iris_options.require_sample_point_is_contained = true;
  iris_options.configuration_space_margin = 1E-5;
  iris_options.relative_termination_threshold = 0.001;
  iris_options.num_collision_infeasible_samples = 5;

  plant.SetPositions(plant_context, q_seed);
  const auto iris_region = IrisInRationalConfigurationSpace(
      plant, *plant_context, q_star, iris_options);
  return iris_region;
}

CspaceFreePolytope::SearchResult FindCspacePolytope(
    const OneDofDiagram& one_dof_diagram, const Vector1d& q_star,
    const Eigen::MatrixXd& C_init, const Eigen::VectorXd& d_init) {
  CspaceFreePolytope cspace_free_polytope(
      &(one_dof_diagram.plant()), &(one_dof_diagram.scene_graph()),
      SeparatingPlaneOrder::kAffine, q_star);

  CspaceFreePolytope::BinarySearchOptions binary_search_options;
  binary_search_options.find_lagrangian_options.ignore_redundant_C = true;
  binary_search_options.scale_min = 0.01;
  binary_search_options.scale_max = 1;
  binary_search_options.max_iter = 10;

  HPolyhedron hpoly_init(C_init, d_init);

  const auto search_result = cspace_free_polytope.BinarySearch(
      {}, C_init, d_init, hpoly_init.MaximumVolumeInscribedEllipsoid().center(),
      binary_search_options);
  return search_result.value();
}

multibody::rational_old::CspaceFreeRegionSolution FindCspaceRegionOld(
    const OneDofDiagram& one_dof_diagram, const Vector1d& q_star,
    const Eigen::MatrixXd& C_init, const Eigen::VectorXd& d_init,
    const Vector1d& q_seed) {
  multibody::rational_old::CspaceFreeRegion cspace_free_region(
      one_dof_diagram.diagram(), &(one_dof_diagram.plant()),
      &(one_dof_diagram.scene_graph()),
      multibody::rational_old::SeparatingPlaneOrder::kAffine,
      multibody::rational_old::CspaceRegionType::kGenericPolytope);
  const Eigen::VectorXd s_lower =
      cspace_free_region.rational_forward_kinematics().ComputeTValue(
          one_dof_diagram.plant().GetPositionLowerLimits(), q_star);
  const Eigen::VectorXd s_upper =
      cspace_free_region.rational_forward_kinematics().ComputeTValue(
          one_dof_diagram.plant().GetPositionUpperLimits(), q_star);
  const Eigen::VectorXd s_seed =
      cspace_free_region.rational_forward_kinematics().ComputeTValue(q_seed,
                                                                     q_star);
  multibody::rational_old::CspaceFreeRegion::BinarySearchOption
      binary_search_options;
  binary_search_options.epsilon_max = 0;
  binary_search_options.epsilon_min = multibody::rational_old::FindEpsilonLower(
      C_init, d_init, s_lower, s_upper, s_seed, std::nullopt);
  binary_search_options.max_iters = 10;
  binary_search_options.lagrangian_backoff_scale = 1E-3;
  binary_search_options.search_d = false;

  solvers::SolverOptions solver_options;

  multibody::rational_old::CspaceFreeRegionSolution binary_search_result;
  cspace_free_region.CspacePolytopeBinarySearch(
      q_star, {}, C_init, d_init, binary_search_options, solver_options, s_seed,
      std::nullopt, &binary_search_result);
  return binary_search_result;
}

int DoMain() {
  OneDofDiagram one_dof_diagram{};
  auto diagram_context = one_dof_diagram.diagram().CreateDefaultContext();
  auto& plant_context = one_dof_diagram.plant().GetMyMutableContextFromRoot(
      diagram_context.get());

  Vector1d q_star(0);
  Vector1d q_seed(0);

  const HPolyhedron iris_region =
      FindIrisRegion(one_dof_diagram.plant(), q_seed, q_star, &plant_context);

  const auto search_result_old = FindCspaceRegionOld(
      one_dof_diagram, q_star, iris_region.A(), iris_region.b(), q_seed);

  const auto search_result = FindCspacePolytope(
      one_dof_diagram, q_star, iris_region.A(), iris_region.b());

  return 0;
}
}  // namespace optimization
}  // namespace geometry
}  // namespace drake

int main() { return drake::geometry::optimization::DoMain(); }
