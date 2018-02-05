#include "drake/common/find_resource.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_tree_construction.h"
#include "drake/systems/trajectory_optimization/rigid_body_tree_multiple_shooting.h"

namespace drake {
namespace examples {
namespace minitaur {
/**
 * Construct a RigidBodyPlant containing Minitaur. This plant can be used for
 * both trajectory optimization and simulation.
 */
std::unique_ptr<systems::RigidBodyPlant<double>> ConstructPlant() {
  // First construct an empty rigid body tree. We will later parse the URDF
  // file, and store the kinematics and dynamics information of the robot
  // in this tree.
  auto tree = std::make_unique<RigidBodyTree<double>>();

  // Now add the URDF to this tree.
  // Note that the orientation of the floating point is parameterized
  // by roll-pitch-yaw angles. It is better to switch to quaternions in the
  // future. But we need to figure out how to impose the unit length constraint
  // on the quaternion, which the time derivative of the quaternion is
  // perpendicular to the quaternion.
  parsers::urdf::AddModelInstanceFromUrdfFile(
      FindResourceOrThrow("drake/examples/minitaur/models/minitaur.urdf"),
      multibody::joints::kRollPitchYaw, /* Floating base orientation is
                                           parameterized by roll-pitch-yaw */
      nullptr, /* The floating base is not welded to any frame. */
      tree.get());

  multibody::AddFlatTerrainToWorld(tree.get(), 100., 10.);

  // Instantiate a RigidBodyPlant from the RigidBodyTree.
  // The RigidBodyPlant is used for trajectory optimization.
  // This plant is going to be simulated by time stepping simulator, with
  // time step being dt.
  const double dt{1E-3};
  return std::make_unique<systems::RigidBodyPlant<double>>(std::move(tree), dt);
}

int DoMain() {
  auto plant = ConstructPlant();

  // Construct a visualizer to draw a posture of the robot.
  // Drake uses LCM to communicate between to the visualizer.
  drake::lcm::DrakeLcm lcm;
  manipulation::SimpleTreeVisualizer simple_visualizer(
      plant->get_rigid_body_tree(), &lcm);

  // Set up a trajectory optimization problem
  return 0;
}
}  // namespace minitaur
}  // namespace examples
}  // namespace drake

int main() { return drake::examples::minitaur::DoMain(); }
