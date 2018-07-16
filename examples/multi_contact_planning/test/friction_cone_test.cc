#include "drake/examples/multi_contact_planning/friction_cone.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace examples {
namespace multi_contact_planning {
void CheckLinearizedFrictionConeConstructor(
    int num_edges, const Eigen::Ref<const Eigen::Vector3d>& unit_normal,
    double mu) {
  const LinearizedFrictionCone cone(num_edges, unit_normal, mu);
  EXPECT_EQ(cone.num_edges(), num_edges);
  EXPECT_EQ(cone.edges().cols(), num_edges);
  EXPECT_TRUE(
      CompareMatrices(cone.unit_normal(), unit_normal.normalized(), 1E-12));

  EXPECT_TRUE(CompareMatrices(cone.edges().rowwise().sum(),
                              unit_normal * num_edges, 1E-14));
  EXPECT_TRUE(CompareMatrices(
      cone.edges().colwise().norm(),
      Eigen::RowVectorXd::Constant(num_edges, std::sqrt(1 + mu * mu)), 1E-12));
  EXPECT_TRUE(CompareMatrices(
      ((unit_normal.transpose() * cone.edges()).array() /
       cone.edges().colwise().norm().array())
          .matrix(),
      Eigen::RowVectorXd::Constant(num_edges, 1 / std::sqrt(1 + mu * mu)),
      1E-12));
}

GTEST_TEST(LinearizedFrictionCone, Test1) {
  // Test constructor.
  CheckLinearizedFrictionConeConstructor(6, Eigen::Vector3d::UnitZ(), 0.5);
  CheckLinearizedFrictionConeConstructor(6, Eigen::Vector3d::UnitX(), 0.5);
  CheckLinearizedFrictionConeConstructor(6, Eigen::Vector3d::UnitY(), 0.5);
  CheckLinearizedFrictionConeConstructor(6, -Eigen::Vector3d::UnitX(), 0.5);
  CheckLinearizedFrictionConeConstructor(6, -Eigen::Vector3d::UnitY(), 0.5);
  CheckLinearizedFrictionConeConstructor(6, -Eigen::Vector3d::UnitZ(), 0.5);
  CheckLinearizedFrictionConeConstructor(
      6, Eigen::Vector3d(1.0 / 3, 2.0 / 3, 2.0 / 3), 0.5);
  CheckLinearizedFrictionConeConstructor(
      6, Eigen::Vector3d(1.0 / 3, -2.0 / 3, 2.0 / 3), 0.5);
}
}  // namespace multi_contact_planning
}  // namespace examples
}  // namespace drake
