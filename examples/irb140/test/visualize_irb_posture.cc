#include <gflags/gflags.h>

#include "drake/examples/irb140/test/irb140_common.h"
#include "drake/common/find_resource.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree.h"

DEFINE_int32(orientation_id, 0, "The ID of the ee orientation.");

namespace drake {
namespace examples {
namespace irb140 {
int DoMain(int argc, char* argv[]) {
  gflags::SetUsageMessage("Make sure drake-visualizer is running!");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto tree = drake::examples::IRB140::ConstructIRB140();

  const std::string kEndEffectorPath =
      "drake/examples/irb140/urdf/end_effector.urdf";
  const Eigen::Vector3d ee_pos(-0.5, -0.4, 0.5);
  const Eigen::Vector3d ee_rpy(0, 0, FLAGS_orientation_id == 0 ? 0 : M_PI_2);
  auto ee_weld_frame = std::allocate_shared<RigidBodyFrame<double>>(
      Eigen::aligned_allocator<RigidBodyFrame<double>>(), "world", nullptr,
      ee_pos /* ee position */, ee_rpy /* ee orientation */);

  drake::parsers::urdf::AddModelInstanceFromUrdfFile(
      FindResourceOrThrow(kEndEffectorPath), drake::multibody::joints::kFixed,
      ee_weld_frame, true, tree.get());

  Eigen::Matrix<double, 6, 1> q;
  q.setZero();
  drake::examples::IRB140::VisualizePosture(*tree, q);
  return 0;
}
}  // namespace irb140
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::examples::irb140::DoMain(argc, argv);
}
