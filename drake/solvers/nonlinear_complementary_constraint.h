#pragma once

#include "drake/solvers/evaluator_base.h"

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
 *   αᵀ * β ≤ 0
 * </pre>
 * where α, β are additional slack variables.
 * The nonlinear constraints are
 * <pre>
 *  α = g(z)
 *  β = h(z)
 *  αᵀ * β ≤ 0
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
        h_(std::forward<HH>(h)),
        num_complementary_{detail::FunctionTraits<G>::numOutputs(g)},
        z_size_{detail::FunctionTraits<G>::numInputs(g)} {
    DRAKE_ASSERT(detail::FunctionTraits<G>::numOutputs(g) == detail::FunctionTraits<H>::numOutputs(h));
    DRAKE_ASSERT(detail::FunctionTraits<G>::numInputs)
  }

  int num_complementary() const {return num_complementary_;}

 private:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd& y) const override {
    // y = [g(z) - α]
    //     [h(z) - β]
    //     [αᵀ * β  ]
    // x = [z; α; β]
    y.resize(num_outputs());
    // g(z) - α
    detail::FunctionTraits<G>::eval(g_, x.topRows(z_size_), y.topRows(num_complementary_));
    y.topRows(num_complementary_) -= x.middleRows(z_size_, num_complementary_);
    // h(z) - β
    detail::FunctionTraits<H>::eval(h_, x.topRows(z_size_), y.middleRows(num_complementary_, num_complementary_));
    y.middleRows(num_complementary_, num_complementary_) -= x.bottomRows(num_complementary_);
    // αᵀ * β
    y(num_outputs() - 1) = x.middleRows(z_size_, num_complementary_).dot(x.bottomRows(num_complementary_));
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd& y) const override {
    // y = [g(z) - α]
    //     [h(z) - β]
    //     [αᵀ * β  ]
    // x = [z; α; β]
    y.resize(num_outputs());
    // g(z) - α
    detail::FunctionTraits<G>::eval(g_, x.topRows(z_size_), y.topRows(num_complementary_));
    y.topRows(num_complementary_) -= x.middleRows(z_size_, num_complementary_);
    // h(z) - β
    detail::FunctionTraits<H>::eval(h_, x.topRows(z_size_), y.middleRows(num_complementary_, num_complementary_));
    y.middleRows(num_complementary_, num_complementary_) -= x.bottomRows(num_complementary_);
    // αᵀ * β
    y(num_outputs() - 1) = x.middleRows(z_size_, num_complementary_).dot(x.bottomRows(num_complementary_));
  }

 private:
  const G g_;
  const H h_;
  const int num_complementary_;
  const int z_size_;
};
}  // namespace solvers
}  // namespace drake