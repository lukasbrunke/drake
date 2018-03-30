#pragma once

#include <Eigen/Core>

#include "drake/common/drake_copyable.h"

namespace drake {
namespace manipulation {
namespace planner {
class BodyContactPoint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BodyContactPoint)

  /**
   * @param p_BQ The position of the contact point Q in the body frame B.
   * @param n_B. The normal vector of the friction cone, pointing towards the
   * object, expressed in the body frame B.
   * @param e_B. The edges of the linearized friction cone at point Q, pointing
   * towards the object, expressed in the object body frame B.
   */
  BodyContactPoint(const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
                   const Eigen::Ref<const Eigen::Vector3d>& n_B,
                   const Eigen::Ref<const Eigen::Matrix3Xd>& e_B)
      : p_BQ_{p_BQ}, n_B_{n_B.normalized()}, e_B_{e_B} {}

  ~BodyContactPoint() = default;

  const Eigen::Vector3d& p_BQ() const { return p_BQ_; }

  const Eigen::Vector3d n_B() const { return n_B_; }

  const Eigen::Vector3d e_B() const { return e_B_; }

 private:
  const Eigen::Vector3d p_BQ_;
  const Eigen::Vector3d n_B_;
  const Eigen::Matrix3Xd e_B_;
};
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
