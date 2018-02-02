#include "drake/systems/trajectory_optimization/contact_implicit_direct_transcription.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace systems {
namespace trajectory_optimization {

// This evaluator computes the generalized constraint force Jᵀλ.
GeneralizedConstraintForceEvaluator::GeneralizedConstraintForceEvaluator(
    const RigidBodyTree<double>& tree, int num_lambda,
    std::shared_ptr<KinematicsCacheWithVHelper<AutoDiffXd>> kinematics_helper)
    : EvaluatorBase(tree.get_num_velocities(),
                    tree.get_num_positions() + num_lambda,
                    "generalized constraint force"),
      tree_{&tree},
      num_lambda_(num_lambda),
      kinematics_helper_{kinematics_helper} {}

void GeneralizedConstraintForceEvaluator::DoEval(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    Eigen::VectorXd& y) const {
  AutoDiffVecXd y_t;
  Eval(math::initializeAutoDiff(x), y_t);
  y = math::autoDiffToValueMatrix(y_t);
}

void GeneralizedConstraintForceEvaluator::DoEval(
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd& y) const {
  // x contains q and λ
  DRAKE_ASSERT(x.rows() == tree_->get_num_positions() + num_lambda_);
  const auto q = x.head(tree_->get_num_positions());
  const auto lambda = x.tail(num_lambda_);

  auto kinsol = kinematics_helper_->UpdateKinematics(q);
  // TODO: the user need to determine which constraint will be used here. Also
  // for each constraint, the user needs to compute the Jacobian for that
  // constraint, and multiply Jacobian with the corresponding lambda.
  // Here since the tree contains four bar linkages,
  // tree_->positionConstraints
  // compute the violation of the four-bar linkage constraint.
  const auto J_position_constraint =
      tree_->positionConstraintsJacobian(kinsol, false);
  const int num_position_constraint_lambda = tree_->getNumPositionConstraints();
  const auto position_constraint_lambda =
      lambda.head(num_position_constraint_lambda);
  y = J_position_constraint.transpose() * position_constraint_lambda;
  // If there are more constraint, such as foot above the ground, then you
  // should compute the Jacobian of the foot toe, multiply the transpose of
  // this Jacobian with the ground contact force, and add the product to y.
}

/**
 * Implements the constraint for the backward Euler integration
 * <pre>
 * qᵣ - qₗ = q̇ᵣ*h
 * Mᵣ(vᵣ - vₗ) = (B*uᵣ + Jᵣᵀ*λᵣ -c(qᵣ, vᵣ))h
 * </pre>
 * where
 * qᵣ: The generalized position on the right knot.
 * qₗ: The generalized position on the left knot.
 * vᵣ: The generalized velocity on the right knot.
 * vₗ: The generalized velocity on the left knot.
 * uᵣ: The actuator input on the right knot.
 * Mᵣ: The inertia matrix computed from qᵣ.
 * λᵣ: The constraint force (e.g., contact force, joint limit force, etc) on the
 * right knot.
 * c(qᵣ, vᵣ): The Coriolis, gravity and centripedal force on the right knot.
 * h: The duration between the left and right knot.
 */
class DirectTranscriptionConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DirectTranscriptionConstraint)

  DirectTranscriptionConstraint(
      const RigidBodyTree<double>& tree, int num_lambda,
      std::shared_ptr<KinematicsCacheWithVHelper<AutoDiffXd>>
          kinematics_helper)
      : Constraint(tree.get_num_positions() + tree.get_num_velocities(),
                   1 + 2 * tree.get_num_positions() +
                       2 * tree.get_num_velocities() +
                       tree.get_num_actuators() + num_lambda,
                   Eigen::VectorXd::Zero(tree.get_num_positions() +
                                         tree.get_num_velocities()),
                   Eigen::VectorXd::Zero(tree.get_num_positions() +
                                         tree.get_num_velocities())),
        tree_(&tree),
        num_positions_{tree.get_num_positions()},
        num_velocities_{tree.get_num_velocities()},
        num_actuators_{tree.get_num_actuators()},
        num_lambda_{num_lambda},
        kinematics_helper1_{kinematics_helper},
        constraint_force_evaluator_(tree, num_lambda, kinematics_helper) {
    DRAKE_THROW_UNLESS(num_positions_ == num_velocities_);
  }

  ~DirectTranscriptionConstraint() override = default;

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd& y) const override {
    AutoDiffVecXd y_t;
    Eval(math::initializeAutoDiff(x), y_t);
    y = math::autoDiffToValueMatrix(y_t);
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd& y) const override {
    DRAKE_ASSERT(x.size() == num_vars());

    int x_count = 0;
    // A lambda expression to take num_element entreis from x, in a certain
    // order.
    auto x_segment = [x, &x_count](int num_element) {
      x_count += num_element;
      return x.segment(x_count - num_element, num_element);
    };

    const AutoDiffXd h = x(0);
    x_count++;
    const AutoDiffVecXd q_l = x_segment(num_positions_);
    const AutoDiffVecXd v_l = x_segment(num_velocities_);
    const AutoDiffVecXd q_r = x_segment(num_positions_);
    const AutoDiffVecXd v_r = x_segment(num_velocities_);
    const AutoDiffVecXd u_r = x_segment(num_actuators_);
    const AutoDiffVecXd lambda_r = x_segment(num_lambda_);

    auto kinsol = kinematics_helper1_->UpdateKinematics(q_r, v_r);

    y.resize(num_constraints());

    // By using backward Euler integration, the constraint is
    // qᵣ - qₗ = q̇ᵣ*h
    // Mᵣ(vᵣ - vₗ) = (B*uᵣ + Jᵣᵀ*λᵣ -c(qᵣ, vᵣ))h
    // We assume here q̇ᵣ = vᵣ
    // TODO(hongkai.dai): compute qdot_r from q_r and v_r.
    y.head(num_positions_) = q_r - q_l - v_r * h;

    const auto M = tree_->massMatrix(kinsol);

    // Compute the Coriolis force, centripedal force, etc.
    const typename RigidBodyTree<AutoDiffXd>::BodyToWrenchMap
        no_external_wrenches;
    const auto c = tree_->dynamicsBiasTerm(kinsol, no_external_wrenches);

    // Compute Jᵀλ
    AutoDiffVecXd q_lambda(num_positions_ + num_lambda_);
    q_lambda << q_r, lambda_r;
    AutoDiffVecXd generalized_constraint_force(num_velocities_);
    constraint_force_evaluator_.Eval(q_lambda, generalized_constraint_force);

    y.tail(num_velocities_) =
        M * (v_r - v_l) -
        (tree_->B * u_r + generalized_constraint_force - c) * h;
  }

 private:
  const RigidBodyTree<double>* tree_;
  const int num_positions_;
  const int num_velocities_;
  const int num_actuators_;
  const int num_lambda_;
  // Stores the kinematics cache at the right knot point.
  mutable std::shared_ptr<KinematicsCacheWithVHelper<AutoDiffXd>>
      kinematics_helper1_;
  GeneralizedConstraintForceEvaluator constraint_force_evaluator_;
};
}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace drake
