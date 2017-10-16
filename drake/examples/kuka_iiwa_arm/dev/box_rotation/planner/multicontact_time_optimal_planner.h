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

  std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> CalcWrenchConeEdges()
      const;

  int NumVertices() const { return vertices_.cols(); }

  int NumFrictionConeEdges() const { return friction_cone_edges_.cols(); }

  const Eigen::Matrix3Xd& vertices() const { return vertices_; }

  const Eigen::Matrix3Xd& friction_cone_edges() const {return friction_cone_edges_;}

 private:
  Eigen::Matrix3Xd vertices_;
  Eigen::Matrix3Xd friction_cone_edges_;
};

// forward declaration
class MultiContactTimeOptimalPlannerTest;

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

  const solvers::VectorXDecisionVariable& t_bar() const {return t_bar_;}

  /**
   * Add an upper bound for a segment's time interval. Notice due to our
   * parameterization, the constraint dt <= b would be a non-convex constraint.
   * Here we impose a sufficient condition for dt <= b, such that this
   * sufficient condition is convex.
   * @param interval_index interval_index should be an integer between 0 and
   * nT_ - 2
   * @param dt_lower_bound The upper bound for the time interval
   */
  void AddTimeIntervalLowerBound(int interval_index, double dt_lower_bound);

  solvers::VectorXDecisionVariable z_;
 protected:
  friend MultiContactTimeOptimalPlannerTest;
  const solvers::VectorXDecisionVariable& theta() const {return theta_;}

  const Eigen::VectorXd& s() const {return s_;}

  symbolic::Expression s_ddot(int i) const {
    if (i < nT_ - 1) {
      return (theta_(i + 1) - theta_(i)) / (2 * (s_(i + 1) - s_(i)));
    } else {
      return (theta_(i) - theta_(i - 1)) / (2 * (s_(i) - s_(i - 1)));
    }
  }

  // Given x' (∂x/∂s) and x'' (∂x²/∂²s), compute the acceleration of x.
  Eigen::Matrix<symbolic::Expression, 3, 1> VecAccel(
      int i, const Eigen::Ref<const Eigen::Vector3d>& x_prime,
      const Eigen::Ref<const Eigen::Vector3d>& x_double_prime) const;

  // Given CoM position along the path, compute r' (∂r/∂s) and r'' (∂r²/∂²s)
  std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd> ComPathPrime(
      const Eigen::Ref<const Eigen::Matrix3Xd>& com_path) const;

  // Given orientation along the path, compute ω_bar and ω_bar'
  std::pair<Eigen::Matrix3Xd, Eigen::Matrix3Xd> AngularPathPrime(
      const std::vector<Eigen::Matrix3d>& orient_path) const;

  Eigen::Matrix<symbolic::Expression, 6, 1> ContactFacetWrench(
      int facet_index, int time_index) const;

  void AddTimeIntervalBoundVariables();

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
  // t_bar_ is a nT - 1 x 1 vector. t_bar[i] is an upper bound on the duration
  // of the i'th segment.
  solvers::VectorXDecisionVariable t_bar_;
};
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
