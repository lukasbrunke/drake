#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"

#include <gtest/gtest.h>

#include "drake/common/drake_path.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/rigid_body_tree_construction.h"

namespace drake {
namespace examples {
namespace IRB140 {
namespace {
std::unique_ptr<RigidBodyTreed> ConstructIRB140() {
  std::unique_ptr<RigidBodyTreed> rigid_body_tree = std::make_unique<RigidBodyTreed>();
  const std::string model_path = drake::GetDrakePath() + "/examples/IRB140/urdf/irb_140.urdf";

  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kFixed,
      nullptr,
      rigid_body_tree.get());

  multibody::AddFlatTerrainToWorld(rigid_body_tree.get());
  return rigid_body_tree;
}
class IRB140Test : public ::testing::Test {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IRB140Test)

  IRB140Test()
      : rigid_body_tree_(ConstructIRB140()) {}

  ~IRB140Test() override{}

 protected:
  std::unique_ptr<RigidBodyTreed> rigid_body_tree_;
};
}  // namespace
}  // namespace IRB140
}  // namespace examples
}  // namespace drake