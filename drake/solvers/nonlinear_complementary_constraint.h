#pragma once

#include "drake/multibody/rigid_body_tree.h"
#include "drake/solvers/binding.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/evaluator_base.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
/**
 * This constraint formulates the nonlinear constraints in the general nonlinear
 * complementary condition
 * <pre>
 *   0 ≤ g(z) ⊥ h(z) ≥ 0    (1)
 * </pre>
 * where g(z) and h(z) are column vectors of the same dimension. The inequality
 * is elementwise.
 * To solve a problem with this nonlinear complementary constraint through
 * nonlinear optimization, we formulate (and relax) it as
 * <pre>
 *   α = g(z)
 *   β = h(z)
 *   α, β ≥ 0
 *   αᵢ * βᵢ ≤ 0
 * </pre>
 * where α, β are additional slack variables.
 * The nonlinear constraints are
 * <pre>
 *  α = g(z)
 *  β = h(z)
 *  αᵢ * βᵢ ≤ 0
 * </pre>
 * For more details on solving nonlinear complementary condition by nonlinear
 * optimization, please refer to
 *   Complementarity constraints as nonlinear equations: Theory and numerical
 *   experience
 *     Sven Leyffer 2006.
 * and
 *   Interior methods for mathematical programs with complementarity constraints
 *     by Sven Leyffer, Gabriel Lopez-Calva and  Jorge Nocedal, 2005.
 */
template <typename G, typename H>
class GeneralNonlinearComplementaryConditionNonlinearEvaluator : EvaluatorBase {
 public:

  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GeneralNonlinearComplementaryConditionNonlinearEvaluator)

  template <typename GG, typename HH>
  GeneralNonlinearComplementaryConditionNonlinearEvaluator(GG&& g, HH&& h)
      : EvaluatorBase(detail::FunctionTraits<G>::numOutputs(g) * 2 + 1,
  detail::FunctionTraits<G>::numInputs(g) + 2 * detail::FunctionTraits<G>::numOutputs(g)),
        g_(std::forward<GG>(g)),
        h_(std::forward<HH>(h)) {
    DRAKE_ASSERT(detail::FunctionTraits<G>::numOutputs(g) == detail::FunctionTraits<H>::numOutputs(h));
    DRAKE_ASSERT(detail::FunctionTraits<G>::numInputs)
  }

 private:
  void DoEval(const)

 private:
  const G g_;
  const H h_;
};
}  // namespace solvers
}  // namespace drake