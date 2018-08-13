#pragma once

#include <memory>

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
 * We also want the invariant set to include certain states x₁, ... xₘ, so we
 * impose the constraint V(xᵢ) ≤ ρ for all i = 1, ..., m
 * And our goal is to minimize ρ.
 *
 * With a fixed V, we use a bilinear alternation to minimize ρ.
 * ρ step:
 *   Fix lᵢ(x), min ρ
 *   s.t -∂V/∂x * f(x,K(x+xₑⁱ)+k₀) -  lᵢ(x)(ρ - V) is SOS for all i = 1, ..., n
 *       V(xᵢ) ≤ ρ for all i = 1, ..., m
 * l step:
 *   Fix ρ,  max ε
 *   s.t -∂V/∂x * f(x,K(x+xₑⁱ)+k₀) -  lᵢ(x)(ρ - V) - εxᵀx is SOS for all i
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
   * @param l_degree The degree of the polynomial lᵢ(x).
   */
  RobustInvariantSetVerfication(
      const System<symbolic::Expression>& system,
      const Eigen::Ref<const Eigen::MatrixXd>& K,
      const Eigen::Ref<const Eigen::VectorXd>& k0,
      const Eigen::Ref<const Eigen::MatrixXd>& x_err_vertices, int l_degree);

  /**
   * Constructs a SOS problem
   *   Fix ρ,  max ε
   *   s.t -∂V/∂x * f(x,K(x+xₑⁱ)+k₀) -  lᵢ(x)(ρ - V) - εxᵀx is SOS for all i
   * @param V The Lyapunov candidate.
   * @param x_val In most cases the dynamics f(x) is not a polynomial of x. We
   * thus need to do taylor expansion of f around the x_val.
   * @param taylor_order The order for the taylor expansion on f(x)
   * @param rho_value The fixed value of ρ.
   * @pre The size of x_val matches with the number of states in the system.
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructLagrangianStep(
      const symbolic::Polynomial& V,
      const Eigen::Ref<const Eigen::VectorXd>& x_val, int taylor_order,
      double rho_value) const;

  const std::vector<symbolic::Polynomial>& l_polynomials() const {
    return l_polynomials_;
  }

  const solvers::VectorXIndeterminate& x() const { return x_; }

  friend class RobustVerificationTester;

 private:
  void ConstructSymbolicEnvironment(
      const Eigen::Ref<
          const Eigen::Matrix<symbolic::Variable, Eigen::Dynamic, 1>>& x_var,
      const Eigen::Ref<const Eigen::VectorXd>& x_val,
      symbolic::Environment* env) const;

  // Calculate Vdot = ∂V/∂x * f(x,K(x+xₑⁱ)+k₀).
  // @param[out] Vdot A x_err_vertices.col() x 1 vector. Vdot[i] is the Vdot
  // for xₑⁱ.
  void CalcVdot(const symbolic::Polynomial& V,
                const Eigen::Ref<const Eigen::VectorXd>& x_val,
                int taylor_order, VectorX<symbolic::Polynomial>* Vdot) const;

  const System<symbolic::Expression>* system_;

  solvers::VectorXIndeterminate x_;
  symbolic::Variables x_set_;
  // The control law is u = K_(x+x_err) + k0_
  Eigen::MatrixXd K_;
  Eigen::VectorXd k0_;
  Eigen::MatrixXd x_err_vertices_;
  std::vector<symbolic::Polynomial> l_polynomials_;
  std::vector<VectorX<symbolic::Variable>> l_polynomials_coeffs_;
};
}  // namespace analysis
}  // namespace systems
}  // namespace drake
