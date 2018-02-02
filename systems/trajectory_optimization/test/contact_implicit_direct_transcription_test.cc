#include "drake/systems/trajectory_optimization/contact_implicit_direct_transcription.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/math/autodiff.h"
namespace drake {
namespace systems {
namespace trajectory_optimization {
namespace {
GTEST_TEST(GeneralizedConstraintForceEvaluatorTest, TestEval) {
  auto tree = std::make_unique<RigidBodyTree<double>>();
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow("drake/examples/simple_four_bar/FourBar.urdf"),
      multibody::joints::kFixed, tree.get());
  const int num_lambda = tree->getNumPositionConstraints();
  
  auto cache_helper = std::make_shared<KinematicsCacheWithVHelper<AutoDiffXd>>(*tree);

  GeneralizedConstraintForceEvaluator evaluator(*tree, num_lambda, cache_helper);

  // Set q to some arbitrary number.
  Eigen::VectorXd q(tree->get_num_positions());
  for (int i = 0; i < q.rows(); ++i) {
    q(i) = i + 1;
  }
  // Set lambda to some arbitrary number.
  Eigen::VectorXd lambda(num_lambda);
  for (int i = 0; i < lambda.rows(); ++i) {
    lambda(i) = 2 * i + 3;
  }
  Eigen::VectorXd x(q.rows() + lambda.rows());
  x << q, lambda;
  const auto tx = math::initializeAutoDiff(x);
  AutoDiffVecXd ty;
  evaluator.Eval(tx, ty);
  EXPECT_EQ(ty.rows(), tree->get_num_velocities());
}
}  // namespace
}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace drake
