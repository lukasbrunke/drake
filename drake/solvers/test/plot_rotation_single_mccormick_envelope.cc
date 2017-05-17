#include "drake/solvers/test/rotation_constraint_visualization.h"

#include "drake/common/call_matlab.h"
#include "drake/solvers/rotation_constraint.h"
#include "drake/solvers/rotation_constraint_internal.h"

namespace drake {
namespace solvers {
namespace {

void DrawLineBetweenPoints(const Eigen::Ref<const Eigen::Vector3d>& pt1,
                           const Eigen::Ref<const Eigen::Vector3d>& pt2) {
  using common::CallMatlab;
  auto h = CallMatlab(1, "plot3", Eigen::Vector2d(pt1(0), pt2(0)), Eigen::Vector2d(pt1(1), pt2(1)), Eigen::Vector2d(pt1(2), pt2(2)));
  CallMatlab("set", h[0], "Color", Eigen::Vector3d(0.8, 0.1, 0.1));
}

// Draw one mccormick envelope for 2 binary variables per half axis case.
// Draw the convex hull of the intersection region, between the surface of the
// sphere, and the box [0, 0.5, 0] <= x <= [0.5, 1, 0.5]
void DrawSingleMcCormickEnvelopes() {
  using common::CallMatlab;
  Eigen::Vector3d bmin(0, 0.5, 0);
  Eigen::Vector3d bmax(0.5, 1, 0.5);
  DrawBoxSphereIntersection(bmin, bmax);
  auto intersection_pts = internal::ComputeBoxEdgesAndSphereIntersection(bmin, bmax);
  DRAKE_ASSERT(intersection_pts.size() == 4);
  // The four points are
  // [0   1         0]
  // [0.5 sqrt(3)/2 0]
  // [0   sqrt(3)/2 0.5]
  // [0.5 sqrt(2)/2 0.5]
  DrawLineBetweenPoints(intersection_pts[0], intersection_pts[1]);
  DrawLineBetweenPoints(intersection_pts[0], intersection_pts[2]);
  DrawLineBetweenPoints(intersection_pts[1], intersection_pts[3]);
  DrawLineBetweenPoints(intersection_pts[2], intersection_pts[3]);

  // Draw the sphere surface
  const int kNumSurfPts = 11;
  Eigen::Matrix<double, kNumSurfPts, kNumSurfPts> surf_X, surf_Y, surf_Z;
  for (int i = 0; i < kNumSurfPts; ++i) {
    surf_Y.col(i) = Eigen::Matrix<double, kNumSurfPts, 1>::LinSpaced(std::sqrt(2) / 2, 1);
  }
  for (int i = 0; i < kNumSurfPts; ++i) {
    surf_X.row(i) = Eigen::Matrix<double, 1, kNumSurfPts>::LinSpaced(0, std::min(std::sqrt(1 - surf_Y(i, 0) * surf_Y(i, 0)), 0.5));
    for (int j = 0; j < kNumSurfPts; ++j) {
      double z_square = 1.0 - surf_X(i, j) * surf_X(i, j) - surf_Y(i, j) * surf_Y(i,j);
      if (z_square > 0) {
        surf_Z(i, j) = std::min(std::sqrt(z_square), 0.5);
      } else {
        surf_Z(i, j) = 0;
      }
    }
  }
  auto h_sphere = CallMatlab(1, "surf", surf_X, surf_Y, surf_Z);
  Eigen::Vector3d sphere_color(0, 0.5, 0.5);
  CallMatlab("set", h_sphere[0], "FaceColor", sphere_color);
  CallMatlab("set", h_sphere[0], "FaceAlpha", 0.2);
  CallMatlab("set", h_sphere[0], "EdgeColor", sphere_color);
  CallMatlab("set", h_sphere[0], "LineStyle", "none");
}

void DoMain() {
  using common::CallMatlab;

  CallMatlab("figure", 1);
  CallMatlab("clf");
  CallMatlab("hold", "on");
  CallMatlab("axis", "equal");
  //DrawSphere();
  DrawSingleMcCormickEnvelopes();

}
}  // namespace
}  // namespace solvers
}  // namespace drake

int main() {
  drake::solvers::DoMain();
  return 0;
}