#pragma once

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "drake/common/sorted_pair.h"
#include "drake/multibody/optimization/contact_wrench_evaluator.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/trajectory_optimization/multiple_shooting.h"

namespace drake {
namespace multibody {
/**
 * Solves a contact implicit trajectory optimization problem to find the
 * trajectory and contact forces of a MultibodyPlant. We formulate this
 * trajectory optimization as a nonlinear optimization problem with
 * complementarity constraints. The formulation is similar to the one in "A
 * Direct Method for Trajectory Optimization of Rigid Bodies Through Contact" by
 * Michael Posa, Cecilia Cantu and Russ Tedrake, International Journal of
 * Robotics Research, 2014. We change the formulation slightly, so that the
 * default is __nonlinear__ Coulomb friction cone instead of a linearized
 * Coulomb friction cone as in the paper.
 */
class ContactImplicitTrajectoryOptimization
    : public systems::trajectory_optimization::MultipleShooting {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactImplicitTrajectoryOptimization)

  /**
   * @param plant The MultibodyPlant for which the trajectory is optimized.
   * @plant must outlive this ContactImplicitTrajectoryOptimization object.
   * @param contexts The length of @p contexts is num_time_samples. contexts[i]
   * is the context for the i'th knot. contexts[i] must outlive this
   * ContactImplicitTrajectoryOptimization object.
   * @param minimum_timestep Minimum spacing between sample times.
   * @param maximum_timestep Maximum spacing between sample times.
   */
  ContactImplicitTrajectoryOptimization(
      const MultibodyPlant<AutoDiffXd>* plant,
      const std::vector<systems::Context<AutoDiffXd>*>& contexts,
      int num_time_samples, double minimum_timestep, double maximum_timestep);

  /**
   * Ignores the contacts between a pair of geometries at a given time
   * sample. Namely we don't ignore the distance that these two geometries
   * shouldn't penetrate, and we ignore the possible contact force between these
   * pair of geometries.
   */
  void SetIgnoredContacts(
      int t_index, const std::unordered_set<SortedPair<geometry::GeometryId>>&
                       ignored_contacts);

  void SetExplicitInContactPairs(
      int t_index, const std::unordered_set<SortedPair<geometry::GeometryId>>&
                       explicit_contacts);

  /**
   * Calling FinalizeContactPairs() indicates that the user has set up all the
   * pairs of possible contacts. The user should not call
   * SetIgnoredImplicitContacts() or SetExplicitInContactPairs() after calling
   * FinalizeContactPairs().
   * Inside FinalizeContactPairs(), it checks each pair of geometries in the
   * SceneGraph, if the contact between the pair is not ignored, or the user
   * hasn't explicit the contact between the pair, then we add the implicit
   * contact constraints between the pair (0 ≤ φ(q) ⊥ fₙ ≥ 0)
   */
  void FinalizeContactPairs(double complementarity_tolerance);

  /**
   * Whether we consider the contact force between a pair of geometries.
   */
  enum class ContactType {
    kIgnore,  ///< No contact force between the pair of geometries, for example,
              ///< this pair of contact is ignored.
    kExplicit,  ///< Explicitly demands contact force between the pair of
                ///< geometries.
    kImplicit,  ///< Implicit contact between the pair of geometries, we will
                ///< impose the complementarity constraint 0 ≤ φ(q) ⊥ fₙ ≥ 0.
  };

 private:
  const MultibodyPlant<AutoDiffXd>& plant_;
  // contexts_[i] is the context for the i'th knot (time t[i]).
  std::vector<systems::Context<AutoDiffXd>*> contexts_;

  // contact_wrench_evaluators_and_lambda_ is a vector of length
  // num_time_samples, contact_wrench_evaluators_and_lambda_[i] contains all the
  // contact wrench evaluators and their corresponding λ at the i'th knot point.
  std::vector<std::unordered_map<std::shared_ptr<ContactWrenchEvaluator>,
                                 VectorX<symbolic::Variable>>>
      contact_wrench_evaluators_and_lambda_;

  bool contact_pairs_finalized_{false};

  std::vector<std::map<SortedPair<geometry::GeometryId>, ContactType>>
      contact_pairs_;
};
}  // namespace multibody
}  // namespace drake
