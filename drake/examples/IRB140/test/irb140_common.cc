#include "drake/examples/IRB140/test/irb140_common.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_matrix_compare.h"
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
}  // namespace IRB140
}  // namespace examples
}  // namespace drake
