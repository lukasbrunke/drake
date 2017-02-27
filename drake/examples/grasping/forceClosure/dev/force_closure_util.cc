#include "drake/examples/grasping/forceClosure/dev/force_closure_util.h"

#include <iostream>
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullFacetList.h"

#include <Eigen/Dense>

#include "drake/common/drake_assert.h"

using Eigen::Matrix;
using orgQhull::Qhull;
using orgQhull::QhullHyperplane;

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
Matrix<double, 6, 7> GenerateWrenchPolytopeInnerSphere7Vertices() {
  using Matrix7d = Eigen::Matrix<double, 7, 7>;
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using RowVector7d = Eigen::Matrix<double, 1, 7>;
  Matrix7d Z = Matrix7d::Identity() - Vector7d::Ones() * RowVector7d::Ones();
  Eigen::SelfAdjointEigenSolver<Matrix7d> eigensolver(Z);
  const auto& Z_eigen_value = eigensolver.eigenvalues();
  Eigen::Matrix<double, 7, 7> W = std::sqrt(7.0/6.0) * eigensolver.eigenvectors();
  for (int i = 1; i < 7; ++i) {
    W.col(i) *= sqrt(Z_eigen_value(i));
  }
  return W.block<7, 6>(0, 1).transpose();
};

Matrix<double, 6, 12> GenerateWrenchPolytopeInnerSphere12Vertices() {
  Matrix<double, 6, 12> W;
  W.setZero();
  for (int i = 0; i < 6; ++i) {
    W(i, 2*i)  = 1;
    W(i, 2 * i + 1) = -1;
  }
  return W;
};

double ForceClosureQ1metricLinearizedFrictionCone(const Eigen::Matrix3Xd& contact_pts, const std::vector<Eigen::Matrix3Xd>& friction_edges, const Eigen::Matrix<double, 6, 6>& Q) {
  int num_wrenches = 0;
  DRAKE_DEMAND(contact_pts.cols() == static_cast<int>(friction_edges.size()));
  for (int i = 0; i < static_cast<int>(friction_edges.size()); ++i) {
    num_wrenches += friction_edges[i].cols();
  }
  Matrix<double, 6, Eigen::Dynamic> wrenches(6, num_wrenches);
  int wrench_count = 0;
  for (int i = 0; i < contact_pts.cols(); ++i) {
    wrenches.block(0, wrench_count, 3, friction_edges[i].cols()) = friction_edges[i];
    for (int j = 0; j < friction_edges[i].cols(); ++j) {
      wrenches.block(3, wrench_count + j, 3, 1) = contact_pts.col(i).cross(friction_edges[i].col(j));
    }
    wrench_count += friction_edges[i].cols();
  }
  Qhull cws_qhull("", 6, num_wrenches, wrenches.data(), "");
  double q1_metric = std::numeric_limits<double>::infinity();
  //Eigen::Matrix<double, 6, 6> Qw_inv = Q.inverse();
  for (const auto& f : cws_qhull.facetList()) {
    const QhullHyperplane& h = f.hyperplane();
    std::cout<<"h" << h <<std::endl;
    q1_metric = std::min(q1_metric, h.offset());
  }
  return 0;
}
}  // namespace forceClosure
}  // namespace grasping
}  // namespace examples
}  // namespace drake