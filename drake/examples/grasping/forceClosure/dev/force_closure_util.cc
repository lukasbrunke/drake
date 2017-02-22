#include "drake/examples/grasping/forceClosure/dev/force_closure_util.h"

#include <Eigen/Dense>

#include "libqhullcpp/Qhull.h"

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
Eigen::Matrix<double, 6, 7> GenerateWrenchPolytopeInnerSphere7Vertices() {
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

Eigen::Matrix<double, 6, 12> GenerateWrenchPolytopeInnerSphere12Vertices() {
  Eigen::Matrix<double, 6, 12> W;
  W.setZero();
  for (int i = 0; i < 6; ++i) {
    W(i, 2*i)  = 1;
    W(i, 2 * i + 1) = -1;
  }
  return W;
};
}  // namespace forceClosure
}  // namespace grasping
}  // namespace examples
}  // namespace drake