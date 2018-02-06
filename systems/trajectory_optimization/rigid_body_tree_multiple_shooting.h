#pragma once

#include <memory>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/trajectory_optimization/multiple_shooting.h"

namespace drake {
namespace systems {
namespace trajectory_optimization {
/**
 * Stores and updates the kinematics cache for the rigid body tree. This cache
 * is used in evaluating the direct transcription constraint
 * <pre>
 * qᵣ - qₗ = q̇ᵣ*h
 * Mᵣ(vᵣ - vₗ) = (B*uᵣ + Jᵣᵀ*λᵣ -c(qᵣ, vᵣ))h
 * </pre>
 * together with the gradient of the constraint w.r.t qₗ, qᵣ, vₗ, vᵣ, uᵣ, λᵣ
 * For each generalized position and velocity, we will compute the kinematics
 * cache, containing information such as the pose of each link. It is very
 * likely that we will need to re-use these information in different functions.
 * To avoid redundant computation, we will store these information in this cache
 * helper, and update the cache every time the generalized position or velocites
 * are changed.
 */
template <typename Scalar>
class KinematicsCacheWithVHelper {
 public:
  explicit KinematicsCacheWithVHelper(const RigidBodyTree<double>& tree)
      : tree_{&tree}, kinsol_(tree.CreateKinematicsCacheWithType<Scalar>()) {
    last_q_.resize(tree.get_num_positions());
    last_v_.resize(tree.get_num_velocities());
  }

  KinematicsCache<Scalar>& UpdateKinematics(
      const Eigen::Ref<const VectorX<Scalar>>& q,
      const Eigen::Ref<const VectorX<Scalar>>& v) {
    if (q.size() != last_q_.size() || q != last_q_ ||
        v.size() != last_v_.size() || v != last_v_) {
      last_q_ = q;
      last_v_ = v;
      kinsol_.initialize(q, v);
      tree_->doKinematics(kinsol_, true);  // compute Jdotv
    }
    return kinsol_;
  }

  KinematicsCache<Scalar>& UpdateKinematics(
      const Eigen::Ref<const VectorX<Scalar>>& q) {
    if (q.size() != last_q_.size() || q != last_q_) {
      last_q_ = q;
      kinsol_.initialize(q, last_v_);
      tree_->doKinematics(kinsol_, true);  // compute Jdotv
    }
    return kinsol_;
  }

 private:
  const RigidBodyTree<double>* tree_;
  VectorX<Scalar> last_q_;
  VectorX<Scalar> last_v_;
  KinematicsCache<Scalar> kinsol_;
};

/** This evaluator computes the generalized constraint force Jᵀλ.
 * Inside this evaluator, we only compute the generalized constraint force
 * coming from the positionConstraint(). RigidBodyTree::positionConstraint()
 * contains the constraint φ(q) = 0, that are ALWAYS active, such as the closed
 * loop constraint (e.g., four-bar linkage).
 * In order to add additional generalized constraint force, the user can
 * derive their evaluator class, and override the DoEval function.
 * TODO(hongkai.dai): add an example of the derived evaluator class, in the
 * minitaur example.
 */
class GeneralizedConstraintForceEvaluator : public solvers::EvaluatorBase {
 public:
  GeneralizedConstraintForceEvaluator(
      const RigidBodyTree<double>& tree, int num_lambda,
      std::shared_ptr<KinematicsCacheWithVHelper<AutoDiffXd>>
          kinematics_helper);

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd& y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd& y) const override;

 protected:
  const RigidBodyTree<double>* tree_;
  const int num_lambda_;
  mutable std::shared_ptr<KinematicsCacheWithVHelper<AutoDiffXd>>
      kinematics_helper_;
};

/**
 * Implements the trajectory optimization for a RigidBodyTree.
 * Trajectory optimization for RigidBodyTree is special, because the dynamics
 * of the tree has some special structures.
 * 1. Since RigidBodyTree has a second order dynamics, its dynamics can be
 * separated as the time derivative on the generalized position, and the time
 * derivative on the generalized velocities.
 * 2. Its generalized acceleration can be affected by the external force, under
 * the term Jᵀλ. We can optimize over λ as decision variables.
 * 3. The kinematics cache can be reused in each knot of the trajectory, so we
 * will store the kinematics cache for each knot.
 */
class RigidBodyTreeMultipleShooting : public MultipleShooting {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RigidBodyTreeMultipleShooting)

  /**
   * Constructor.
   * @param tree The RigidBodyTree whose trajectory will be optimized.
   * @param num_lambda  num_lambda[i] is the length of λ at knot i.
   * @param num_time_samples The total number of knots in the trajectory.
   * @param minimum_timestep The minimum of the time step.
   * @param maximum_timestep The maximum of the time step.
   */
  RigidBodyTreeMultipleShooting(const RigidBodyTree<double>& tree,
                                const std::vector<int>& num_lambda,
                                int num_time_samples, double minimum_timestep,
                                double maximum_timestep);

  PiecewisePolynomialTrajectory ReconstructInputTrajectory() const override;

  PiecewisePolynomialTrajectory ReconstructStateTrajectory() const override;

  const solvers::MatrixXDecisionVariable& GeneralizedPositions() const {
    return q_vars_;
  }

  const solvers::MatrixXDecisionVariable& GeneralizedVelocities() const {
    return v_vars_;
  }

  const std::vector<solvers::VectorXDecisionVariable>& ConstraintForces()
      const {
    return lambda_vars_;
  }

  ~RigidBodyTreeMultipleShooting() override {}

 private:
  void DoAddRunningCost(const symbolic::Expression& e) override;

  // Store system-relevant data for e.g. computing the derivatives during
  // trajectory reconstruction.
  const RigidBodyTree<double>* tree_{nullptr};
  const int num_positions_;
  const int num_velocities_;
  const std::vector<int> num_lambda_;
  std::vector<std::shared_ptr<KinematicsCacheWithVHelper<AutoDiffXd>>>
      kinematics_with_v_helpers_;
  solvers::MatrixXDecisionVariable q_vars_;
  solvers::MatrixXDecisionVariable v_vars_;
  std::vector<solvers::VectorXDecisionVariable> lambda_vars_;
};
}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace drake
