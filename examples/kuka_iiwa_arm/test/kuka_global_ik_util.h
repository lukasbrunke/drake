#pragma once
#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
void AddBoxToTree(RigidBodyTreed* tree,
                  const Eigen::Ref<const Eigen::Vector3d>& box_size,
                  const Eigen::Isometry3d& box_pose, const std::string& name,
                  const Eigen::Vector4d& color = Eigen::Vector4d(0.3, 0.4, 0.5,
                                                                 0.5));

Eigen::Matrix<double, 3, 8> BoxVertices(
    const Eigen::Ref<const Eigen::Vector3d>& box_size,
    const Eigen::Isometry3d& box_pose);

void AddSphereToBody(RigidBodyTreed* tree, int link_idx,
                     const Eigen::Vector3d& pt, const std::string& name,
                     double radius = 0.01);

std::unique_ptr<RigidBodyTreed> ConstructKuka();
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
