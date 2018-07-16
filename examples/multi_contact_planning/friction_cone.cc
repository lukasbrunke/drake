#include "drake/examples/multi_contact_planning/friction_cone.h"

#include "drake/common/drake_assert.h"

namespace drake {
namespace examples {
namespace multi_contact_planning {
LinearizedFrictionCone::LinearizedFrictionCone(
    int num_edges, const Eigen::Ref<const Eigen::Vector3d>& unit_normal,
    double mu)
    : num_edges_(num_edges) {
  DRAKE_ASSERT(num_edges >= 3);
  DRAKE_ASSERT(mu >= 0);
  // First construct a linearized friction cone, whose normal vector is [0;0;1],
  // and then rotate the friction to align the normal vector with unit_normal.
  const Eigen::RowVectorXd theta =
      Eigen::RowVectorXd::LinSpaced(num_edges + 1, 0, 2 * M_PI);
  const Eigen::RowVectorXd cos_theta =
      theta.head(num_edges).array().cos().matrix();
  const Eigen::RowVectorXd sin_theta =
      theta.head(num_edges).array().sin().matrix();
  edges_.resize(3, num_edges);
  edges_.row(0) = cos_theta * mu;
  edges_.row(1) = sin_theta * mu;
  edges_.row(2) = Eigen::RowVectorXd::Ones(num_edges_);
  // Now rotate edges_ such that the normal alligns with unit_normal.
  const double unit_normal_len = unit_normal.norm();
  DRAKE_ASSERT(unit_normal_len > 10 * std::numeric_limits<double>::epsilon());
  const Eigen::Vector3d normal = unit_normal / unit_normal_len;
  unit_normal_ = normal;
  Eigen::Matrix3d R;
  // R * [0;0;1] = normal => R.col(2) = normal.
  R.col(2) = normal;
  // Arbitrarily pick v = UniX(), if UnitX() aligns with normal, then pick v =
  // UniY();
  Eigen::Vector3d v = Eigen::Vector3d::UnitX();
  Eigen::Vector3d u = normal.cross(v);
  if (u.norm() < 0.01) {
    v = Eigen::Vector3d::UnitY();
    u = normal.cross(v);
  }
  // Now u is perpendicular to normal.
  R.col(0) = u / u.norm();
  R.col(1) = R.col(2).cross(R.col(0));
  edges_ = R * edges_;
}
}  // namespace multi_contact_planning
}  // namespace examples
}  // namespace drake
