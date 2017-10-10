#pragma once

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
struct ContactFacet {
  int num_vertices_;
  Eigen::Matrix3Xd vertices_;
  Eigen::Matrix3Xd cone_edges_;
};

class MultiContactTimeOptimalPlanner : public solvers::MathematicalProgram {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MultiContactTimeOptimalPlanner)

  MultiContactTimeOptimalPlanner(double mass, const Eigen::Ref<const Eigen::Matrix3d>& inertia, const std::vector<ContactFacet>& contact_facet, int nT);

  void SetBoxPoseSequence(const std::vector<Eigen::Isometry3d>& box_pose);

 private:
  double m_;
  Eigen::Matrix3d I_B_; // Inertia matrix about the body center of mass,
                        // measured and expressed in the body frame.
  std::vector<ContactFacet> contact_facets_;
  Eigen::Vector3d gravity_;
  int nT_;
  solvers::VectorXDecisionVariable theta_;
};
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
