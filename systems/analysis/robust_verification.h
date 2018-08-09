#pragma once

#include "drake/common/symbolic.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/system.h"

/**
 * We want to verify the invariant set {x | V(x) ≤ ρ } for a control affine
 * system ẋ = f(x, u). We assume that this system is stabilized by a linear
 * controller u = K(x+xₑ) + k₀, where xₑ is the unknown but bounded state
 * estimation error. We assume that the state estimation error xₑ is within a
 * polytope ConvexHull(xₑ¹, xₑ², ..., xₑⁿ).
 * The invariant set condition is
 * V=ρ ⇒ V̇ ≤ 0
 * we can formulate this implication as the following algebraic constraint
 * -∂V/∂x * f(x,K(x+xₑⁱ)+k₀) -  lᵢ(x)(ρ - V) is SOS for all i = 1, ..., n
 */
namespace drake {
namespace systems {
namespace analysis {
class RobustInvariantSetVerfication {
 public:
  /**
   * Constructor
   * @param system The control-affine system. @throws a runtime error if the
   * system is not control affine. `system` should remain valid for the life
   * time of this newly constructed object.
   * @param K The linear coefficient of the control law.
   * @param k0 The constant term of the control law.
   * @param x_err_vertices x_err_vertices.col(i) is the i'th vertex of the
   * polytope bounding the state estimation error x_err.
   */
  RobustInvariantSetVerfication(
      const System<symbolic::Expression>& system,
      const Eigen::Ref<const Eigen::MatrixXd>& K,
      const Eigen::Ref<const Eigen::VectorXd>& k0,
      const Eigen::Ref<const Eigen::MatrixXd>& x_err_vertices);

 private:
  const System<symbolic::Expression>* system_;

  solvers::VectorXIndeterminate x_;
  // The control law is u = K_(x+x_err) + k0_
  Eigen::MatrixXd K_;
  Eigen::VectorXd k0_;
  Eigen::MatrixXd x_err_vertices_;
};
}  // namespace analysis
}  // namespace systems
}  // namespace drake
