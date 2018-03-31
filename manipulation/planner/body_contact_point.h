#pragma once

#include <Eigen/Core>

#include "drake/common/drake_copyable.h"

namespace drake {
namespace manipulation {
namespace planner {
class BodyContactPoint {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BodyContactPoint)

  /**
   * @param p_BQ The position of the contact point Q in the body frame B.
   * @param n_B. The normal vector of the friction cone, pointing towards the
   * object, expressed in the body frame B.
   * @param e_B. The edges of the linearized friction cone at point Q, pointing
   * towards the object, expressed in the object body frame B. The edges stored
   * in this class will be normalized to have unit length.
   */
  BodyContactPoint(const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
                   const Eigen::Ref<const Eigen::Matrix3Xd>& e_B)
      : p_BQ_{p_BQ},
        e_B_{(e_B.array() /
              ((Eigen::Vector3d::Ones() * e_B.colwise().norm()).array()))
                 .matrix()} {}

  ~BodyContactPoint() = default;

  const Eigen::Vector3d& p_BQ() const { return p_BQ_; }

  /** Getter for the edges, pointing to the object. Each edge has a unit
   * length */
  const Eigen::Matrix3Xd& e_B() const { return e_B_; }

 private:
  Eigen::Vector3d p_BQ_;
  Eigen::Vector3d n_B_;
  Eigen::Matrix3Xd e_B_;
};
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
