#pragma once

#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace IRB140 {
bool CompareIsometry3d(const Eigen::Isometry3d& X1, const Eigen::Isometry3d& X2, double tol = 1E-10);

std::unique_ptr<RigidBodyTreed> ConstructIRB140();

void VisualizePosture(const RigidBodyTreed& tree, const Eigen::Ref<const Eigen::VectorXd>& q);
}  // namespace IRB140
}  // namespace examples
}  // namespace drake
