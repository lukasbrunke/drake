#include "drake/solvers/test/rotation_constraint_visualization.h"

#include "drake/common/call_matlab.h"
#include "drake/solvers/rotation_constraint.h"
#include "drake/solvers/rotation_constraint_internal.h"

namespace drake {
namespace solvers {
namespace {
using common::MatlabRemoteVariable;
using common::CallMatlab;
MatlabRemoteVariable DrawLineBetweenPoints(const Eigen::Ref<const Eigen::Vector3d>& pt1,
                           const Eigen::Ref<const Eigen::Vector3d>& pt2) {
  using common::CallMatlab;
  auto h = CallMatlab(1, "plot3", Eigen::Vector2d(pt1(0), pt2(0)), Eigen::Vector2d(pt1(1), pt2(1)), Eigen::Vector2d(pt1(2), pt2(2)));
  CallMatlab("set", h[0], "Color", Eigen::Vector3d(0.8, 0.1, 0.1));
  return h[0];
}

MatlabRemoteVariable DrawTriangle(const Eigen::Vector3d& pt1, const Eigen::Vector3d& pt2, const Eigen::Vector3d& pt3, const Eigen::RowVector3d& color) {
  auto h = CallMatlab(1, "fill3", Eigen::RowVector3d(pt1(0), pt2(0), pt3(0)),
             Eigen::RowVector3d(pt1(1), pt2(1), pt3(1)),
             Eigen::RowVector3d(pt1(2), pt2(2), pt3(2)), color);
  CallMatlab("set", h[0], "FaceAlpha", 0.2);
  CallMatlab("set", h[0], "EdgeColor", color);
  CallMatlab("set", h[0], "LineStyle", "none");
  return h[0];
}

MatlabRemoteVariable DrawArcLineArea(const std::vector<Eigen::Vector3d>& arc_pts, const Eigen::RowVector3d& color) {
  int num_pts = arc_pts.size();
  Eigen::Matrix3Xd pts_x(3, num_pts - 2);
  Eigen::Matrix3Xd pts_y(3, num_pts - 2);
  Eigen::Matrix3Xd pts_z(3, num_pts - 2);
  for (int i = 0; i < num_pts - 2; ++i) {
    pts_x.col(i) << arc_pts[i](0), arc_pts[i + 1](0), arc_pts[num_pts - 1](0);
    pts_y.col(i) << arc_pts[i](1), arc_pts[i + 1](1), arc_pts[num_pts - 1](1);
    pts_z.col(i) << arc_pts[i](2), arc_pts[i + 1](2), arc_pts[num_pts - 1](2);
  }
  auto h = CallMatlab(1, "fill3", pts_x, pts_y, pts_z, color);
  CallMatlab("set", h[0], "FaceAlpha", 0.2);
  CallMatlab("set", h[0], "EdgeColor", color);
  CallMatlab("set", h[0], "LineStyle", "none");
  return h[0];
}

void DrawTriangleNormal(const Eigen::Vector3d& pt1, const Eigen::Vector3d& pt2, const Eigen::Vector3d& pt3, const Eigen::Vector3d& normal_vec, const std::string& name) {
  Eigen::Vector3d plane_center = (pt1 + pt2 + pt3) / 3;
  //auto h_quiver = CallMatlab(1, "quiver3", plane_center(0), plane_center(1), plane_center(2), normal_vec(0), normal_vec(1), normal_vec(2), 0);
  //CallMatlab("set", h_quiver[0], "Color", Eigen::RowVector3d(0, 0, 0));
  //CallMatlab("set", h_quiver[0], "MaxHeadSize", 0.35);
  //CallMatlab("set", h_quiver[0], "LineWidth", 1);
  //CallMatlab("set", h_quiver[0], "ShowArrowHead", "on");
  Eigen::RowVector2d arrow_x(plane_center(0), plane_center(0) + normal_vec(0));
  Eigen::RowVector2d arrow_y(plane_center(1), plane_center(1) + normal_vec(1));
  Eigen::RowVector2d arrow_z(plane_center(2), plane_center(2) + normal_vec(2));
  auto h_arrow = CallMatlab(1, "arrow3d", arrow_x, arrow_y, arrow_z, 0.8, 0.005, 0.01, Eigen::RowVector3d(0, 0, 0));
  Eigen::Vector3d text_pos = plane_center + normal_vec * 1.2;
  auto h_text = CallMatlab(1, "text", text_pos(0), text_pos(1), text_pos(2), name);
  CallMatlab("set", h_text[0], "FontSize", 12);
}

void DrawSurfacePatch(const std::vector<Eigen::Vector3d>& intersection_pts, const Eigen::RowVector3d& sphere_color) {
  // Draw the sphere surface
  const int kNumSurfPts = 21;
  Eigen::Matrix<double, kNumSurfPts, kNumSurfPts> surf_X, surf_Y, surf_Z;
  for (int i = 0; i < kNumSurfPts; ++i) {
    surf_Y.col(i) = Eigen::Matrix<double, kNumSurfPts, 1>::LinSpaced(std::sqrt(2) / 2, 1);
  }
  for (int i = 0; i < kNumSurfPts; ++i) {
    surf_X.row(i) = Eigen::Matrix<double, 1, kNumSurfPts>::LinSpaced(0, std::sqrt(1 - surf_Y(i, 0) * surf_Y(i, 0)));
    for (int j = 0; j < kNumSurfPts; ++j) {
      double z_square = 1.0 - surf_X(i, j) * surf_X(i, j) - surf_Y(i, j) * surf_Y(i,j);
      if (z_square > 0) {
        surf_Z(i, j) = std::sqrt(z_square);
      } else {
        surf_Z(i, j) = 0;
      }
      if (surf_Z(i, j) > 0.5) {
        surf_Y(i, j) = std::sqrt(1 - surf_X(i, j) * surf_X(i, j) - 0.25);
        surf_Z(i, j) = 0.5;
      }
      if (surf_X(i, j) > 0.5) {
        surf_X(i, j) = 0.5;
        surf_Y(i, j) = std::sqrt(1 - surf_Z(i, j) * surf_Z(i, j) - 0.25);
      }
    }
  }
  auto h_sphere = CallMatlab(1, "surf", surf_X, surf_Y, surf_Z);
  CallMatlab("set", h_sphere[0], "FaceColor", sphere_color);
  CallMatlab("set", h_sphere[0], "FaceAlpha", 0.2);
  CallMatlab("set", h_sphere[0], "EdgeColor", sphere_color);
  CallMatlab("set", h_sphere[0], "LineStyle", "none");
}


// Draw one mccormick envelope for 2 binary variables per half axis case.
// Draw the convex hull of the intersection region, between the surface of the
// sphere, and the box [0, 0.5, 0] <= x <= [0.5, 1, 0.5]
void DrawSingleMcCormickEnvelopes(const Eigen::Vector3d& bmin, const Eigen::Vector3d& bmax, const Eigen::RowVector3d& sphere_color) {
  using common::CallMatlab;
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

  // Draw the patch on the surface
  DrawSurfacePatch(intersection_pts, sphere_color);

  // The two inner facets are triangles with points 0, 1, 3
  // and with points 0, 2, 3
  Eigen::Vector3d n1, n2;
  double d1, d2;
  internal::ComputeTriangleOutwardNormal(intersection_pts[0], intersection_pts[2], intersection_pts[3], &n1, &d1);
  std::cout << "n' * x >= d: " << n1.transpose() << "* x >= " << d1 << std::endl;
  internal::ComputeTriangleOutwardNormal(intersection_pts[0], intersection_pts[1], intersection_pts[3], &n2, &d2);
  std::cout << "n' * x >= d: " << n2.transpose() << "* x >= " << d2 << std::endl;
  DrawLineBetweenPoints(intersection_pts[0], intersection_pts[3]);

  // Draw the inner facets
  Eigen::RowVector3d plane_color = sphere_color;
  DrawTriangle(intersection_pts[0], intersection_pts[1], intersection_pts[3], plane_color);
  DrawTriangle(intersection_pts[0], intersection_pts[2], intersection_pts[3], plane_color);

  // Draw the normal vector
  DrawTriangleNormal(intersection_pts[0], intersection_pts[2], intersection_pts[3], n1 / 5, "n_1");
  DrawTriangleNormal(intersection_pts[0], intersection_pts[1], intersection_pts[3], n2 / 5, "n_2");

  // Draw the boundary regions as part of the axis-aligned planes, with the arc
  // being one boundary of the region.
  int num_arc_pts = 20;
  std::vector<Eigen::Vector3d> arc_pts(num_arc_pts);
  // The arc between [0 1 0] and [0.5 sqrt(3)/2 0]
  for (int i = 0; i < num_arc_pts; ++i) {
    arc_pts[i](0) = 0.5 / (num_arc_pts - 1) * static_cast<double>(i);
    arc_pts[i](1) = std::sqrt(1 - arc_pts[i](0) * arc_pts[i](0));
    arc_pts[i](2) = 0;
  }
  DrawArcLineArea(arc_pts, plane_color);

  // The arc between [0.5 sqrt(3)/2 0] and [0.5 sqrt(2)/2 0.5]
  for (int i = 0; i < num_arc_pts; ++i) {
    arc_pts[i](0) = 0.5;
    arc_pts[i](2) = 0.5 / (num_arc_pts - 1) * static_cast<double>(i);
    arc_pts[i](1) = std::sqrt(1 - arc_pts[i](2) * arc_pts[i](2) - 0.25);
  }
  DrawArcLineArea(arc_pts, plane_color);

  // The arc between [0.5 sqrt(2)/2 0.5] and [0 sqrt(3)/2 0.5]
  for (int i = 0; i < num_arc_pts; ++i) {
    arc_pts[i](2) = 0.5;
    arc_pts[i](0) = 0.5 / (num_arc_pts - 1) * static_cast<double>(i);
    arc_pts[i](1) = std::sqrt(1 - arc_pts[i](0) * arc_pts[i](0) - 0.25);
  }
  DrawArcLineArea(arc_pts, plane_color);

  // The arc between [0 sqrt(3)/2 0.5] and [0 1 0]
  for (int i = 0; i < num_arc_pts; ++i) {
    arc_pts[i](0) = 0;
    arc_pts[i](2) = 0.5 / (num_arc_pts - 1) * static_cast<double>(i);
    arc_pts[i](1) = std::sqrt(1 - arc_pts[i](2) * arc_pts[i](2));
  }
  DrawArcLineArea(arc_pts, plane_color);
}

void DrawAllMcCormickEnvelopes(int num_bins) {
  for (int i = 0; i < num_bins; ++i) {
    for (int j = 0; j < num_bins; ++j) {
      for (int k = 0; k < num_bins; ++k) {
        Eigen::Vector3d bmin(static_cast<double>(i) / num_bins,
                             static_cast<double>(j) / num_bins,
                             static_cast<double>(k) / num_bins);
        Eigen::Vector3d bmax(static_cast<double>(i + 1) / num_bins,
                             static_cast<double>(j + 1) / num_bins,
                             static_cast<double>(k + 1) / num_bins);
        if (bmin.norm() <= 1 && bmax.norm() >= 1) {
          DrawBoxSphereIntersection(bmin, bmax);
        }
      }
    }
  }
}

void DoMain() {
  using common::CallMatlab;

  Eigen::Vector3d bmin(0, 0.5, 0);
  Eigen::Vector3d bmax(0.5, 1, 0.5);
  auto intersection_pts = internal::ComputeBoxEdgesAndSphereIntersection(bmin, bmax);
  CallMatlab("figure", 1);
  CallMatlab("clf");
  CallMatlab("hold", "on");
  CallMatlab("axis", "equal");
  DrawSphere();
  DrawAllMcCormickEnvelopes(2);
  Eigen::RowVector3d patch_color(0.6, 0.2, 0.2);
  DrawSurfacePatch(intersection_pts, patch_color);
  CallMatlab("xlabel", "x");
  CallMatlab("ylabel", "y");
  CallMatlab("zlabel", "z");
  CallMatlab("view", 145, 30);


  CallMatlab("figure", 2);
  CallMatlab("clf");
  CallMatlab("hold", "on");
  CallMatlab("axis", "equal");
  CallMatlab("xlabel", "x");
  CallMatlab("ylabel", "y");
  CallMatlab("zlabel", "z");
  //DrawSphere();
  DrawSingleMcCormickEnvelopes(bmin, bmax, patch_color);
  //CallMatlab("view", 93, 18);
  CallMatlab("view",-117, 27);
}
}  // namespace
}  // namespace solvers
}  // namespace drake

int main() {
  drake::solvers::DoMain();
  return 0;
}