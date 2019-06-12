#include "drake/multibody/optimization/contact_implicit_trajectory_optimization.h"

#include <utility>
#include <vector>

#include "drake/multibody/inverse_kinematics/kinematic_constraint_utilities.h"
#include "drake/multibody/optimization/static_friction_cone_complementarity_constraint.h"
#include "drake/multibody/optimization/static_friction_cone_constraint.h"

using drake::multibody::internal::RefFromPtrOrThrow;
using drake::multibody::internal::UpdateContextConfiguration;

namespace drake {
namespace multibody {
ContactImplicitTrajectoryOptimization::ContactImplicitTrajectoryOptimization(
    const MultibodyPlant<AutoDiffXd>* plant,
    const std::vector<systems::Context<AutoDiffXd>*>& contexts,
    int num_time_samples, double minimum_timestep, double maximum_timestep)
    : systems::trajectory_optimization::MultipleShooting(
          plant->num_actuated_dofs(),
          plant->num_positions() + plant->num_velocities(), num_time_samples,
          minimum_timestep, maximum_timestep),
      plant_{*plant},
      contexts_{contexts},
      contact_wrench_evaluators_and_lambda_(num_time_samples),
      contact_pairs_(num_time_samples) {
  DRAKE_DEMAND(static_cast<int>(contexts_.size()) == num_time_samples);
  const auto& query_port = plant_.get_geometry_query_input_port();
  for (int i = 0; i < num_time_samples; ++i) {
    if (!query_port.HasValue(*contexts_[i])) {
      throw std::invalid_argument(
          "ContactImplicitTrajectoryOptimization: Cannot get a valid "
          "geometry::QueryObject for contexts[" +
          std::to_string(i) +
          "]. Please refer to AddMultibodyPlantSceneGraph.");
    }
    const auto& query_object =
        query_port.Eval<geometry::QueryObject<AutoDiffXd>>(*contexts_[i]);
    const geometry::SceneGraphInspector<AutoDiffXd>& inspector =
        query_object.inspector();
    const std::set<std::pair<geometry::GeometryId, geometry::GeometryId>>
        collision_candidate_pairs = inspector.GetCollisionCandidates();
    // contact_pairs_[i].reserve(collision_candidate_pairs.size());
    for (const auto& collision_candidate_pair : collision_candidate_pairs) {
      // Initialize each contact pair to be implicit contact.
      contact_pairs_[i].emplace_hint(
          contact_pairs_[i].end(),
          SortedPair<geometry::GeometryId>(collision_candidate_pair.first,
                                           collision_candidate_pair.second),
          ContactType::kImplicit);
    }
  }
}

void ContactImplicitTrajectoryOptimization::SetIgnoredContacts(
    int t_index, const std::unordered_set<SortedPair<geometry::GeometryId>>&
                     ignored_contacts) {
  if (contact_pairs_finalized_) {
    throw std::runtime_error(
        "ContactImplicitTrajectoryOptimization::SetIgnoredContacts(): cannot "
        "call this function after calling "
        "ContactImplicitTrajectoryOptimization::FinalizeContactPairs().");
  }
  DRAKE_DEMAND(t_index >= 0 && t_index < N());
  for (const auto& ignored_contact : ignored_contacts) {
    auto it = contact_pairs_[t_index].find(ignored_contact);
    it->second = ContactType::kIgnore;
  }
}

void ContactImplicitTrajectoryOptimization::SetExplicitInContactPairs(
    int t_index, const std::unordered_set<SortedPair<geometry::GeometryId>>&
                     explicit_contacts) {
  if (contact_pairs_finalized_) {
    throw std::runtime_error(
        "ContactImplicitTrajectoryOptimization::SetExplicitInContacts(): "
        "cannot call this function after calling "
        "ContactImplicitTrajectoryOptimization::FinalizeContactPairs().");
  }
  DRAKE_DEMAND(t_index >= 0 && t_index < N());
  for (const auto& explicit_contact : explicit_contacts) {
    auto it = contact_pairs_[t_index].find(explicit_contact);
    it->second = ContactType::kExplicit;
  }
}

void ContactImplicitTrajectoryOptimization::FinalizeContactPairs(
    double complementarity_tolerance) {
  // Add the decision variables for the contact forces.
  for (int i = 0; i < N(); ++i) {
    contact_wrench_evaluators_and_lambda_[i].reserve(contact_pairs_[i].size());
    for (const auto& contact_pair_type : contact_pairs_[i]) {
      if (contact_pair_type.second == ContactType::kExplicit ||
          contact_pair_type.second == ContactType::kImplicit) {
        auto lambda = NewContinuousVariables<3>();
        auto contact_wrench_evaluator =
            std::make_shared<ContactWrenchFromForceInWorldFrameEvaluator>(
                &plant_, contexts_[i], contact_pair_type.first);
        contact_wrench_evaluators_and_lambda_[i].emplace_hint(
            contact_wrench_evaluators_and_lambda_[i].end(),
            contact_wrench_evaluator, lambda);
        if (contact_pair_type.second == ContactType::kExplicit) {
          // Add the constraint that the contact force is within the friction
          // cone, and the two geometries are in contact.
          // First add the friction cone constraint.
          AddConstraint(std::make_shared<StaticFrictionConeConstraint>(
                            contact_wrench_evaluator.get()),
                        lambda);
          // Now add the constraint that the two geometries are in contact.
        } else if (contact_pair_type.second == ContactType::kImplicit) {
          // Add the complementarity constraint 0 ≤ φ(q) ⊥ fₙ ≥ 0.
          AddStaticFrictionConeComplementarityConstraint(
              contact_wrench_evaluator.get(), complementarity_tolerance,
              state(i).head(plant_.num_positions()), lambda, this);
        }
      }
    }
  }
}

}  // namespace multibody
}  // namespace drake
