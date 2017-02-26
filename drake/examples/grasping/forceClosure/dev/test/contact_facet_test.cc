#include "drake/examples/grasping/forceClosure/dev/contact_facet.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_matrix_compare.h"

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
namespace test {
GTEST_TEST(ContactFacetTest, testFrictionCone) {
  Eigen::Matrix3d vert;
  // clang-format off
  vert << 0, 1, 0,
          1, 0, 0,
          0, 0, 0;
  // clang-format on
  Eigen::Vector3d normal(0, 0, 1);

  // Test the friction cone edges when the facet has different orientation.
  std::vector<Eigen::AngleAxisd> rotate;
  rotate.push_back(Eigen::AngleAxisd(0, Eigen::Vector3d(0, 0, 1)));
  rotate.push_back(Eigen::AngleAxisd(M_PI/3, Eigen::Vector3d(0, 1, 0)));
  rotate.push_back(Eigen::AngleAxisd(M_PI/3, Eigen::Vector3d(1, 0, 0)));
  for (const auto& r : rotate) {
    auto rotate_matrix = r.toRotationMatrix();
    ContactFacet f(rotate_matrix * vert, rotate_matrix * normal);

    EXPECT_EQ(f.num_vertices(), 3);

    const auto& f_normal = f.facet_normal();
    EXPECT_TRUE(CompareMatrices(f_normal, rotate_matrix * normal, 1E-12, MatrixCompareType::absolute));

    // For a single edge with friction coefficient being 0, the edge is the facet normal.
    EXPECT_TRUE(CompareMatrices(f.LinearizedFrictionConeEdges<1>(0), f_normal, 1E-12, MatrixCompareType::absolute));

    // For 4 edges case.
    const auto& edges = f.LinearizedFrictionConeEdges<4>(1);
    EXPECT_EQ(edges.cols(), 4);

    EXPECT_TRUE(CompareMatrices(edges.rowwise().sum(), 4 * f_normal, 1E-12, MatrixCompareType::absolute));
    
    // Check the angle between the edges and the facet normal, the cosine of the angle should be 1/sqrt(2).
    EXPECT_TRUE(CompareMatrices(
        ((f_normal.transpose() * edges).array() / (edges.colwise().norm().array())).matrix(),
        Eigen::RowVector4d::Constant(1 / sqrt(2)), 1E-12, MatrixCompareType::absolute));
  }

}
}  // namespace test
}  // namespace forceClosure
}  // namespace grasping
}  // namespace examples
}  // namespace test