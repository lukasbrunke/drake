#include "drake/examples/minitaur/contact_implicit_direct_transcription.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace examples {
namespace minitaur {
namespace {

// Stores and updates the kinematics cache for the rigid body tree. For each
// generalized position and velocity, we will compute the kinematics cache,
// containing information such as the pose of each link. It is very likely that
// ther will be multiple constraints, on the same knot point. All these
// constraints will consume the same kinematics information, and thus re-using
// this kinematics cache, without re-computing it.
class KinematicsCacheHelper {
 public:
  explicit KinematicsCacheHelper(const RigidBodyTree<double>& tree)
      : tree_{&tree},
        kinsol_(tree.CreateKinematicsCacheWithType<AutoDiffXd>()) {}

  KinematicsCache<AutoDiffXd>& UpdateKinematics(
      const Eigen::Ref<const AutoDiffVecXd>& q,
      const Eigen::Ref<const AutoDiffVecXd>& v) {
    if (q.size() != last_q_.size() || q != last_q_ ||
        v.size() != last_v_.size() || v != last_v_) {
      last_q_ = q;
      last_v_ = v;
      kinsol_.initialize(q, v);
      tree_->doKinematics(kinsol_);
    }
    return kinsol_;
  }

  KinematicsCache<AutoDiffXd>& UpdateKinematics(
      const Eigen::Ref<const AutoDiffVecXd>& q) {
    if (q.size() != last_q_.size() || q != last_q_) {
      last_q_ = q;
      kinsol_.initialize(q, last_v_);
      tree_->doKinematics(kinsol_);
    }
    return kinsol_;
  }

 private:
  const RigidBodyTree<double>* tree_;
  AutoDiffVecXd last_q_;
  AutoDiffVecXd last_v_;
  KinematicsCache<AutoDiffXd> kinsol_;
};

// This evaluator computes the generalized constraint force Jᵀλ.
class GeneralizedConstraintForceEvaluator : public solvers::EvaluatorBase {
 public:
  GeneralizedConstraintForceEvaluator(
      const RigidBodyTree<double>& tree, int num_lambda,
      std::shared_ptr<KinematicsCacheHelper> kinematics_helper)
      : EvaluatorBase(tree.get_num_velocities(),
                      tree.get_num_positions() + num_lambda,
                      "generalized constraint force"),
        tree_{&tree},
        num_lambda_(num_lambda),
        kinematics_helper_{kinematics_helper} {}

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd& y) const override {
    AutoDiffVecXd y_t;
    Eval(math::initializeAutoDiff(x), y_t);
    y = math::autoDiffToValueMatrix(y_t);
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd& y) const override {
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
    const int num_position_constraint_lambda =
        tree_->getNumPositionConstraints();
    const auto position_constraint_lambda =
        lambda.head(num_position_constraint_lambda);
    y = J_position_constraint.transpose() * position_constraint_lambda;
    // If there are more constraint, such as foot above the ground, then you
    // should compute the Jacobian of the foot toe, multiply the transpose of
    // this Jacobian with the ground contact force, and add the product to y.
  }

 private:
  const RigidBodyTree<double>* tree_;
  const int num_lambda_;
  mutable std::shared_ptr<KinematicsCacheHelper> kinematics_helper_;
};

class DirectTranscriptionConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DirectTranscriptionConstraint)

  DirectTranscriptionConstraint(
      const RigidBodyTree<double>& tree, int num_lambda,
      std::shared_ptr<KinematicsCacheHelper> kinematics_helper)
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
    const AutoDiffVecXd q0 = x_segment(num_positions_);
    const AutoDiffVecXd v0 = x_segment(num_velocities_);
    const AutoDiffVecXd q1 = x_segment(num_positions_);
    const AutoDiffVecXd v1 = x_segment(num_velocities_);
    const AutoDiffVecXd u1 = x_segment(num_actuators_);
    const AutoDiffVecXd lambda1 = x_segment(num_lambda_);

    auto kinsol = kinematics_helper1_->UpdateKinematics(q1, v1);

    y.resize(num_constraints());

    // By using backward Euler integration, the constraint is
    // q1 - q0 - v1 * h = 0
    // M(v1 - v0) = (B * u1 + Jᵀλ1 - c(q, v))h
    y.head(num_positions_) = q1 - q0 - v1 * h;

    const auto M = tree_->massMatrix(kinsol);

    // Compute the Coriolis force, centripedal force, etc.
    const typename RigidBodyTree<AutoDiffXd>::BodyToWrenchMap
        no_external_wrenches;
    const auto c = tree_->dynamicsBiasTerm(kinsol, no_external_wrenches);

    // Compute Jᵀλ1
    AutoDiffVecXd q_lambda(num_positions_ + num_lambda_);
    q_lambda << q1, lambda1;
    AutoDiffVecXd generalized_constraint_force(num_velocities_); 
    constraint_force_evaluator_.Eval(q_lambda, generalized_constraint_force);

    y.tail(num_velocities_) =
        M * (v1 - v0) - (tree_->B * u1 + generalized_constraint_force - c) * h;
  }

 private:
  const RigidBodyTree<double>* tree_;
  const int num_positions_;
  const int num_velocities_;
  const int num_actuators_;
  const int num_lambda_;
  // Stores the kinematics cache at the right knot point.
  mutable std::shared_ptr<KinematicsCacheHelper> kinematics_helper1_;
  GeneralizedConstraintForceEvaluator constraint_force_evaluator_;
};
}  // namespace
}  // namespace minitaur
}  // namespace examples
}  // namespace drake
