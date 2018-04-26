#pragma once

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
/**
 * Solves the Linear Complementarity Problem (LCP) as a Mixed-integer Linear
 * Programming (MILP) problem. An LCP has the following form
 * ```
 * 0≤ w ⊥ z ≥ 0
 * w = M * z + q
 * ```
 * where w, z ∈ ℝⁿ are unknown variables. M ∈ ℝⁿˣⁿ, q ∈ ℝⁿ are given
 * matrix/vector.
 *
 * In order to solve this problem using MILP, we consider to introduce binary
 * variables b ∈ {0, 1}ⁿ, where b(i) = 1 => z(i) = 0, b(i) = 0 => w(i) = 0.
 * Suppose that we know that w and z are upper bounded as
 * z(i) ≤ zₘₐₓ(i), w(i) ≤ wₘₐₓ(i)
 * we can then formulate the following MILP
 * ```
 * min 0
 * s.t 0 ≤ z(i) ≤ zₘₐₓ(i) * (1 - b(i))
 *     0 ≤ w(i) ≤ wₘₐₓ(i) *  b(i)
 *     w = M * z + q
 * ```
 * @note The upper bound value zₘₐₓ, wₘₐₓ are specified by the user. They should
 * be large enough to be greater than any possible value of z and w
 * respectively, but not too large. A tight estimation of zₘₐₓ and wₘₐₓ will
 * significantly speed up the MILP solver.
 */
class MixedIntegerLinearProgramLCP {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MixedIntegerLinearProgramLCP)

  MixedIntegerLinearProgramLCP(const Eigen::Ref<const Eigen::VectorXd>& q,
                               const Eigen::Ref<const Eigen::MatrixXd>& M,
                               const Eigen::Ref<const Eigen::ArrayXd>& z_max,
                               const Eigen::Ref<const Eigen::ArrayXd>& w_max);

  ~MixedIntegerLinearProgramLCP() = default;

  const VectorXDecisionVariable& w() const { return w_; }

  const VectorXDecisionVariable& z() const { return z_; }

  const VectorXDecisionVariable& b() const { return b_; }

  const MathematicalProgram& prog() const { return *prog_; }

  MathematicalProgram* get_mutable_prog() { return prog_.get(); }

  /**
   * Given the assignment of the binary variables b_val, solve the following 
   * linear equations to polish the solution.
   * w = q + M * z
   * w(i) = 0 if b(i) = 0
   * z(i) = 0 if b(i) = 1
   */
  bool PolishSolution(const Eigen::Ref<const Eigen::VectorXd>& b_val, Eigen::VectorXd* w_sol, Eigen::VectorXd* z_sol) const;
 private:
  const int n_;
  const Eigen::VectorXd q_;
  const Eigen::MatrixXd M_;
  const Eigen::ArrayXd z_max_;
  const Eigen::ArrayXd w_max_;
  std::unique_ptr<MathematicalProgram> prog_;
  VectorXDecisionVariable w_;
  VectorXDecisionVariable z_;
  VectorXDecisionVariable b_;
};
}  // namespace solvers
}  // namespace drake
