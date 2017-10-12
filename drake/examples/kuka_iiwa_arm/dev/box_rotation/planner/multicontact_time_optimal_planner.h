#pragma once

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
class ContactFacet {
 public:
  ContactFacet(const Eigen::Ref<const Eigen::Matrix3Xd>& vertices,
               const Eigen::Ref<const Eigen::Matrix3Xd>& friction_cone_edges);

  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> WrenchConeEdges() const;

  int num_vertices() const { return vertices_.cols(); }

  int num_friction_cone_edges() const { return friction_cone_edges_.cols(); }

  const Eigen::Matrix3Xd& vertices() const { return vertices_; }

 private:
  Eigen::Matrix3Xd vertices_;
  Eigen::Matrix3Xd friction_cone_edges_;
};

class MultiContactTimeOptimalPlanner : public solvers::MathematicalProgram {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MultiContactTimeOptimalPlanner)

  /**
   * @param mass The mass of the polytope to be manipulated.
   * @param inertia The inertia matrix of the polytope to be manipulated.
   * @param contact_facets The contact facets on the polytope.
   * @param nT The number of time points.
   * @param num_arms The number of arms to manipulate the polytope.
   */
  MultiContactTimeOptimalPlanner(
      double mass, const Eigen::Ref<const Eigen::Matrix3d>& inertia,
      const std::vector<ContactFacet>& contact_facets, int nT, int num_arms);

  void SetObjectPoseSequence(const std::vector<Eigen::Isometry3d>& object_pose);

 protected:
  symbolic::Expression s_ddot(int i) const {
    return (theta_(i + 1) - theta_(i)) / (2 * (s_(i + 1) - s_(i)));
  }

  // Given x' (∂x/∂s) and x'' (∂x²/∂²s), compute the acceleration of x.
  Eigen::Matrix<symbolic::Expression, 3, 1> x_accel(
      int i, const Eigen::Ref<const Eigen::Vector3d>& x_prime,
      const Eigen::Ref<const Eigen::Vector3d>& x_double_prime) const;

  // Given CoM position along the path, compute r' (∂r/∂s) and r'' (∂r²/∂²s)
  std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd> com_path_prime(const Eigen::Ref<const Eigen::Matrix3Xd>& com_path) const;

  // Given orientation along the path, compute ω_bar and ω_bar'
  std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd> angular_path_prime(const std::vector<Eigen::Matrix3d>& orient_path) const;

 private:
  double m_;
  Eigen::Matrix3d I_B_;  // Inertia matrix about the body center of mass,
                         // measured and expressed in the body frame.
  std::vector<ContactFacet> contact_facets_;
  Eigen::Vector3d gravity_;
  int nT_;
  int num_arms_;
  // s is the parameter along the path.
  Eigen::VectorXd s_;
  // θ is the slack variable, θ = ṡ², where ṡ is the time derivative of s
  solvers::VectorXDecisionVariable theta_;
  // λ is the same length as contact_facets_. λ[i] is of size num_weights x nT_;
  // where num_weights is the number of contact cone edges on that contact
  // facet.
  std::vector<solvers::MatrixXDecisionVariable> lambda_;
  // B_ is a num_facets x nT matrix. B(i, j) = 1 if the i'th facet is active
  // at j'th time point.
  solvers::MatrixXDecisionVariable B_;
};
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
