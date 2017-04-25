#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_path.h"
#include "drake/examples/kuka_iiwa_arm/dev/tools/simple_tree_visualizer.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_world/iiwa_wsg_diagram_factory.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_world/world_sim_tree_builder.h"
#include "drake/examples/schunk_wsg/schunk_wsg_constants.h"
#include "drake/examples/schunk_wsg/schunk_wsg_lcm.h"
#include "drake/lcm/drake_lcm.h"
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
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/matrix_gain.h"
#include "drake/util/drakeGeometryUtil.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
using schunk_wsg::SchunkWsgStatusSender;
using schunk_wsg::SchunkWsgTrajectoryGenerator;
using systems::Context;
using systems::Diagram;
using systems::DiagramBuilder;
using systems::DrakeVisualizer;
using systems::InputPortDescriptor;
using systems::OutputPortDescriptor;
using systems::RigidBodyPlant;
using systems::Simulator;

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

  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kFixed,
      robot_base_frame,
      rigid_body_tree.get());

  auto iiwa_frame_ee = rigid_body_tree->findFrame("iiwa_frame_ee");
  const std::string schunk_path = drake::GetDrakePath() + "/examples/schunk_wsg/models/schunk_wsg_50_fixed_joint.sdf";
  parsers::sdf::AddModelInstancesFromSdfFile(schunk_path, drake::multibody::joints::kFixed, iiwa_frame_ee, rigid_body_tree.get());
  
  multibody::AddFlatTerrainToWorld(rigid_body_tree.get());

  return rigid_body_tree;
}

int DoMain() {
  drake::lcm::DrakeLcm lcm;
  auto tree = ConstructKuka();
  tools::SimpleTreeVisualizer simple_tree_visualizer(*tree.get(), &lcm);
  simple_tree_visualizer.visualize(Eigen::VectorXd::Zero(7));
  return 0;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::kuka_iiwa_arm::DoMain();
}