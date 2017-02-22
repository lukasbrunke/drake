#include "drake/examples/grasping/forceClosure/dev/force_closure_util.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_matrix_compare.h"

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
namespace test {
GTEST_TEST(ForceClosureUtilTest, InnerPolytope7VerticesTest) {
  Eigen::Matrix<double, 6, 7> W = GenerateWrenchPolytopeInnerSphere7Vertices();
  Eigen::Matrix<double, 7, 7> Z = -1.0 / 6.0 * Eigen::Matrix<double, 7, 7>::Ones();
  for (int i = 0; i < 7; ++i) {
    Z(i, i) = 1.0;
  }
  EXPECT_TRUE(CompareMatrices(W.transpose() * W, Z, 1e-10, MatrixCompareType::absolute));
}

GTEST_TEST(ForceClosureUtilTest, InnerPolytope12VerticesTest) {
  Eigen::Matrix<double, 6, 12> W = GenerateWrenchPolytopeInnerSphere12Vertices();
  for (int i = 0; i < 6; ++i) {
    Eigen::Matrix<double, 6, 1> v_expected;
    v_expected.setZero();
    v_expected(i) = 1;
    EXPECT_TRUE(CompareMatrices(W.col(2 * i), v_expected));
    v_expected(i) = -1;
    EXPECT_TRUE(CompareMatrices(W.col(2 * i + 1), v_expected));
  }
}
}  // namespace test
}  // namespace forceClosure
}  // namespace grasping
}  // namespace examples
}  // namespace drake