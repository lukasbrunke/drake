/* clang-format off */
#include "drake/solvers/test/rotation_constraint_visualization.h"
/* clang-format on */

#include "drake/common/call_matlab.h"
#include "drake/solvers/rotation_constraint.h"
#include "drake/solvers/rotation_constraint_internal.h"

// Plot the McCormick Envelope on the unit sphere, to help developers to
// visualize the McCormick Envelope relaxation.

namespace drake {
namespace solvers {
namespace {
void DrawCuttingPlane(double intersection, int axis) {
  using common::CallMatlab;
  Eigen::Matrix<double, 4, 3> plane_vert;
  int axis1 = (axis + 1) % 3;
  int axis2 = (axis + 2) % 3;
  plane_vert.col(axis) = Eigen::Vector4d::Constant(intersection);
  plane_vert.col(axis1) << -1, -1, 1, 1;
  plane_vert.col(axis2) << -1, 1, 1, -1;
  auto h = CallMatlab(1, "fill3", plane_vert.col(0), plane_vert.col(1), plane_vert.col(2), Eigen::RowVector3d(0.5, 0.2, 0.3));
  CallMatlab("set", h[0], "FaceAlpha", 0.2);
  //CallMatlab("set", h[0], "LineStyle", "none");
}

// This draws all the McCormick Envelope in the first orthant (+++).
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
          // DrawBox(bmin, bmax);
        }
      }
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    for (int i = 0; i < num_bins; ++i) {
      double intersection = static_cast<double>(i) / num_bins;
      DrawCuttingPlane(intersection, axis);
    }
  }
}

void DoMain() {
  using common::CallMatlab;
  for (int num_bins = 1; num_bins <= 3; ++num_bins) {
    CallMatlab("figure", num_bins);
    CallMatlab("clf");
    CallMatlab("hold", "on");
    CallMatlab("axis", "equal");
    DrawSphere();
    DrawAllMcCormickEnvelopes(num_bins);
    CallMatlab("xlabel", "x");
    CallMatlab("ylabel", "y");
    CallMatlab("zlabel", "z");
    CallMatlab("view", 145, 25);
    //std::string file_name = "/home/hongkai/research/ISRR2017/figure/sphere_" + std::to_string(num_bins) + "_bins_w_box";
    //CallMatlab("print", file_name, "-dsvg");
  }
}
}  // namespace
}  // namespace solvers
}  // namespace drake

int main() {
  drake::solvers::DoMain();
  return 0;
}
