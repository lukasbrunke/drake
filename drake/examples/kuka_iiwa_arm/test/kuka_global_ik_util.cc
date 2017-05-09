#include "drake/examples/kuka_iiwa_arm/test/kuka_global_ik_util.h"

#include "drake/multibody/shapes/geometry.h"
#include "drake/multibody/joints/fixed_joint.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
Eigen::Matrix<double, 3, 8> AddBoxToTree(RigidBodyTreed* tree, const Eigen::Ref<const Eigen::Vector3d>& box_size, const Eigen::Isometry3d& box_pose, const std::string& name) {
  auto body = std::make_unique<RigidBody<double>>();
  body->set_name(name);
  body->set_mass(1.0);
  body->set_spatial_inertia(Matrix6<double>::Identity());

  const DrakeShapes::Box shape(box_size);
  const Eigen::Vector4d material(0.3, 0.4, 0.5, 0.5);
  const DrakeShapes::VisualElement visual_element(shape, Eigen::Isometry3d::Identity(), material);
  body->AddVisualElement(visual_element);

  auto joint = std::make_unique<FixedJoint>(name + "joint", box_pose);
  body->add_joint(&tree->world(), std::move(joint));

  tree->bodies.push_back(std::move(body));
  Eigen::Matrix<double, 3, 8> box_vertices;
  box_vertices.row(0) << 1, 1, 1, 1, -1, -1, -1, -1;
  box_vertices.row(1) << 1, 1, -1, -1, 1, 1, -1, -1;
  box_vertices.row(2) << 1, -1, 1, -1, 1, -1, 1, -1;
  for (int i = 0; i < 3; ++i) {
    box_vertices.row(i) *= box_size(i) / 2;
  }
  box_vertices = box_pose.linear() * box_vertices;
  for (int i = 0; i < 8; ++i) {
    box_vertices.col(i) += box_pose.translation();
  }
  return box_vertices;
}
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake