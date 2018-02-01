#pragma once

#include <memory>

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
class DirectTranscriptionKinematicsCacheHelper {
 public:
  explicit DirectTranscriptionKinematicsCacheHelper(
      const RigidBodyTree<double>& tree);

  KinematicsCache<AutoDiffXd>& UpdateKinematics(
      const Eigen::Ref<const AutoDiffVecXd>& q,
      const Eigen::Ref<const AutoDiffVecXd>& v);

  KinematicsCache<AutoDiffXd>& UpdateKinematics(
      const Eigen::Ref<const AutoDiffVecXd>& q);

 private:
  const RigidBodyTree<double>* tree_;
  AutoDiffVecXd last_q_;
  AutoDiffVecXd last_v_;
  KinematicsCache<AutoDiffXd> kinsol_;
};

/** This evaluator computes the generalized constraint force Jᵀλ.
 * Inside this evaluator, we only compute the generalized constraint force
 * coming from the positionConstraint(). RigidBodyTree::positionConstraint()
 * contains the constraint φ(q) = 0, that are ALWAYS active, such as the closed
 * loop constraint (e.g., four-bar linkage).
 * In order to add additional generalized constraint force, the user can 
 * derive their evaluator class, and override the DoEval function. 
 * TODO(hongkai.dai): add an example of the derived evaluator class, in the minitaur example.
 */
class GeneralizedConstraintForceEvaluator : public solvers::EvaluatorBase {
 public:
  GeneralizedConstraintForceEvaluator(
      const RigidBodyTree<double>& tree, int num_lambda,
      std::shared_ptr<DirectTranscriptionKinematicsCacheHelper>
          kinematics_helper);

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd& y) const override;

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd& y) const override;

 private:
  const RigidBodyTree<double>* tree_;
  const int num_lambda_;
  mutable std::shared_ptr<DirectTranscriptionKinematicsCacheHelper>
      kinematics_helper_;
};

/**
 * Implements the contact implicit trajectory optimization from the paper
 * A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact
 * by Michael Posa, Cecilia Cantu and Russ Tedrake
 */
class ContactImplicitDirectTranscription : public MultipleShooting {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactImplicitDirectTranscription)

  ContactImplicitDirectTranscription(const RigidBodyTree<double>& tree,
                                     int num_time_samples,
                                     double minimum_timestep,
                                     double maximum_timestep);

  ~ContactImplicitDirectTranscription() override {}

 private:
  void DoAddRunningCost(const symbolic::Expression& e) override;

  // Store system-relevant data for e.g. computing the derivatives during
  // trajectory reconstruction.
  const RigidBodyTree<double>* tree_{nullptr};
};
}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace drake
