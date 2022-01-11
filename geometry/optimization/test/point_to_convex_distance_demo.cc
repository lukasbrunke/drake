#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/optimization/test_utilities.h"

namespace drake {
namespace geometry {
namespace optimization {
using drake::math::RigidTransformd;
using Eigen::Vector3d;
GTEST_TEST(Test, foo) {
  const RigidTransformd X_WG(math::RollPitchYawd(.1, .2, 3),
                             Vector3d(-4.0, -5.0, -6.0));
  auto [scene_graph, geom_id] = internal::MakeSceneGraphWithShape(
      Convex(FindResourceOrThrow("drake/geometry/test/octahedron.obj")), X_WG);
  auto context = scene_graph->CreateDefaultContext();
  auto query =
      scene_graph->get_query_output_port().Eval<QueryObject<double>>(*context);

  const Vector3d in1_G{0, 0, 0.}, in2_G{0.1, 0.2, -0.5}, out_G{0.51, .9, 0.4};
  const Vector3d in1_W = X_WG * in1_G;

  const auto query_results = query.ComputeSignedDistanceToPoint(in1_W);
  EXPECT_GT(query_results.size(), 0);
}
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
