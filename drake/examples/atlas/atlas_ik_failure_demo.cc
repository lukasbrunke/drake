#include "drake/multibody/rigid_body_ik.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/common/find_resource.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/lcm/drake_lcm.h"

namespace drake {
namespace examples {
namespace atlas {
int DoMain() {
  std::unique_ptr<RigidBodyTree<double>> rigid_body_tree =
      std::make_unique<RigidBodyTree<double>>();
  const std::string model_path = FindResourceOrThrow(
      "drake/examples/atlas/urdf/atlas_minimal_contact.urdf");

  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kRollPitchYaw,
      nullptr,
      rigid_body_tree.get());

  drake::lcm::DrakeLcm lcm;
  manipulation::SimpleTreeVisualizer visualizer(*rigid_body_tree.get(), &lcm);

  const int kNQ = rigid_body_tree->get_num_positions();
  Eigen::VectorXd q0 = Eigen::VectorXd::Zero(kNQ);
  visualizer.visualize(q0);

  return 0;
}
}  // namespace atlas
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::atlas::DoMain();
}