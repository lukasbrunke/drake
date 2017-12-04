#include "drake/examples/irb140/test/irb140_common.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"

namespace drake {
namespace examples {
namespace IRB140 {
bool CompareIsometry3d(const Eigen::Isometry3d& X1, const Eigen::Isometry3d& X2, double tol) {
  bool orientation_match = CompareMatrices(X1.linear(), X2.linear(), tol, MatrixCompareType::absolute);
  EXPECT_TRUE(CompareMatrices(X1.linear(), X2.linear(), tol, MatrixCompareType::absolute));
  bool position_match = CompareMatrices(X1.translation(), X2.translation(), tol, MatrixCompareType::absolute);
  EXPECT_TRUE(CompareMatrices(X1.translation(), X2.translation(), tol, MatrixCompareType::absolute));
  return position_match && orientation_match;
}

std::unique_ptr<RigidBodyTreed> ConstructIRB140() {
  std::unique_ptr<RigidBodyTree<double>> rigid_body_tree =
      std::make_unique<RigidBodyTree<double>>();
  const std::string model_path = FindResourceOrThrow("drake/examples/irb140/urdf/irb_140_shift.urdf");


  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path,
      drake::multibody::joints::kFixed,
      nullptr,
      rigid_body_tree.get());

  //AddFlatTerrainToWorld(rigid_body_tree.get());

  return rigid_body_tree;
}

void VisualizePosture(const RigidBodyTreed& tree, const Eigen::Ref<const Eigen::VectorXd>& q) {
  lcm::DrakeLcm lcm;
  manipulation::SimpleTreeVisualizer simple_tree_visualizer(tree, &lcm);
  simple_tree_visualizer.visualize(q);
}
}  // namespace IRB140
}  // namespace examples
}  // namespace drake
