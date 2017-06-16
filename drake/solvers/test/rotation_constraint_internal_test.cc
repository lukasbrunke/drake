#include "drake/solvers/rotation_constraint.h"
#include "drake/solvers/rotation_constraint_internal.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_matrix_compare.h"

using Eigen::Vector3d
namespace drake {
namespace solvers {
namespace {
void CompareIntersectionResults(std::vector<Vector3d> desired,
                                std::vector<Vector3d> actual) {
  EXPECT_EQ(desired.size(), actual.size());
  Eigen::Matrix<bool, Eigen::Dynamic, 1> used =
      Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(desired.size(), false);
  double tol = 1e-8;
  for (int i = 0; i < static_cast<int>(desired.size()); i++) {
    // need not be in the same order.
    bool found_match = false;
    for (int j = 0; j < static_cast<int>(desired.size()); j++) {
      if (used(j)) continue;
      if ((desired[i] - actual[j]).lpNorm<2>() < tol) {
        used(j) = true;
        found_match = true;
        continue;
      }
    }
    EXPECT_TRUE(found_match);
  }
}

// For 2 binary variable per half axis, we know it cuts the first orthant into
// 7 regions. 3 of these regions have 4 co-planar vertices; 3 of these regions
// have 4 non-coplanar vertices, and one region has 3 vertices.
GTEST_TEST(RotationTest, TestAreAllVerticesCoPlanar) {
  Eigen::Vector3d n;
  double d;

  // 4 co-planar vertices. Due to symmetry, we only test one out of the three
  // regions.
  Eigen::Vector3d bmin(0.5, 0.5, 0);
  Eigen::Vector3d bmax(1, 1, 0.5);
  std::vector<Eigen::Vector3d> pts =
      internal::ComputeBoxEdgesAndSphereIntersection(bmin, bmax);
  EXPECT_TRUE(internal::AreAllVerticesCoPlanar(pts, &n, &d));
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(n.norm(), 1, 1E-10);
    EXPECT_NEAR(n.dot(pts[i]), d, 1E-10);
    EXPECT_TRUE((n.array() > 0).all());
  }

  // 4 non co-planar vertices. Due to symmetry, we only test one out of the
  // three regions.
  bmin << 0.5, 0, 0;
  bmax << 1, 0.5, 0.5;
  pts = internal::ComputeBoxEdgesAndSphereIntersection(bmin, bmax);
  EXPECT_FALSE(internal::AreAllVerticesCoPlanar(pts, &n, &d));
  EXPECT_TRUE(CompareMatrices(n, Eigen::Vector3d::Zero()));
  EXPECT_EQ(d, 0);

  // 3 vertices
  bmin << 0.5, 0.5, 0.5;
  bmax << 1, 1, 1;
  pts = internal::ComputeBoxEdgesAndSphereIntersection(bmin, bmax);
  EXPECT_TRUE(internal::AreAllVerticesCoPlanar(pts, &n, &d));
  EXPECT_TRUE(CompareMatrices(n, Eigen::Vector3d::Constant(1.0 / std::sqrt(3)),
                              1E-10, MatrixCompareType::absolute));
  EXPECT_NEAR(pts[0].dot(n), d, 1E-10);
}
}
}  // namespace solvers
}  // namespace drake
