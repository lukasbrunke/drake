#include "drake/comon/find_resource.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_plant/rigid_body_plant.h"
#include "drake/multibody/rigid_body_plant/drake_visualizer.h"
#include "drake/systems/trajectory_optimization/direct_transcription.h"

namespace drake {
namespace examples {
namespace minitaur {
int DoMain() {
  // First construct an empty rigid body tree. We will later parse the URDF
  // file, and store the kinematics and dynamics information of the robot
  // in this tree.
  auto tree = std::make_unique<RigidBodyTree<double>>();

}
}  // namespace minitaur
}  // namespace examples
}  // namespace drake

