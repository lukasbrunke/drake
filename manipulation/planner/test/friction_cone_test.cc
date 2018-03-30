#include "drake/manipulation/planner/friction_cone.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace manipulation {
namespace planner {
namespace {
template <int NumEdges>
void CheckFrictionConeEdges(const Eigen::Ref<const Eigen::Vector3d>& n,
                            double mu) {
  const auto edges = GenerateLinearizedFrictionConeEdges<NumEdges>(n, mu);
  const Eigen::Vector3d n_normalized = n.normalized();
  const double tol{1E-14};
  EXPECT_TRUE(
      CompareMatrices(edges.rowwise().sum(), NumEdges * n_normalized, tol));

  const Eigen::Matrix<double, 3, NumEdges> tangentials =
      edges - n_normalized * Eigen::Matrix<double, 1, NumEdges>::Ones();
  EXPECT_TRUE(CompareMatrices(tangentials.colwise().norm(),
                              Eigen::Matrix<double, 1, NumEdges>::Constant(mu),
                              tol));
}

GTEST_TEST(GenerateLinearizedFrictionConeEdgesTest, Test) {
  CheckFrictionConeEdges<4>(Eigen::Vector3d::UnitZ(), 0.5);
  CheckFrictionConeEdges<8>(Eigen::Vector3d::UnitX(), 0.1);
  CheckFrictionConeEdges<8>(Eigen::Vector3d::UnitY(), 1.2);
  CheckFrictionConeEdges<7>(Eigen::Vector3d(0.1, 0.3, 1.2).normalized(), 1.2);
}
}  // namespace
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
