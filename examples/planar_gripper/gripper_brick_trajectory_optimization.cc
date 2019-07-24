#include "drake/examples/planar_gripper/gripper_brick_trajectory_optimization.h"

#include "drake/examples/planar_gripper/brick_static_equilibrium_constraint.h"
#include "drake/examples/planar_gripper/gripper_brick_planning_utils.h"
#include "drake/multibody/inverse_kinematics/kinematic_constraint_utilities.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GripperBrickTrajectoryOptimization::GripperBrickTrajectoryOptimization(
    const GripperBrickSystem<double>* const gripper_brick, int nT,
    const std::unordered_map<Finger, BrickFace>& initial_contact,
    const std::vector<FingerTransition>& finger_transitions,
    const Options& options)
    : gripper_brick_{gripper_brick},
      nT_{nT},
      prog_{std::make_unique<solvers::MathematicalProgram>()},
      q_vars_{prog_->NewContinuousVariables(
          gripper_brick_->plant().num_positions(), nT_)},
      brick_v_y_vars_{prog_->NewContinuousVariables(nT_)},
      brick_v_z_vars_{prog_->NewContinuousVariables(nT_)},
      brick_omega_x_vars_{prog_->NewContinuousVariables(nT_)},
      f_FB_B_(nT_),
      plant_mutable_contexts_(nT_),
      dt_(prog_->NewContinuousVariables(nT_ - 1)) {
  // Create the contexts used at each knot.
  diagram_contexts_.reserve(nT_);
  for (int i = 0; i < nT_; ++i) {
    diagram_contexts_.push_back(
        gripper_brick->diagram().CreateDefaultContext());
    plant_mutable_contexts_[i] =
        &(gripper_brick_->diagram().GetMutableSubsystemContext(
            gripper_brick_->plant(), diagram_contexts_[i].get()));
  }
  AssignVariableForContactForces(initial_contact, finger_transitions, options);
}

void GripperBrickTrajectoryOptimization::AssignVariableForContactForces(
    const std::unordered_map<Finger, BrickFace>& initial_contact,
    const std::vector<FingerTransition>& finger_transitions,
    const Options& options) {
  // I assume initially the system is in static equilibrium.
  std::vector<std::pair<Finger, BrickFace>> initial_finger_face_contacts;
  for (const auto& finger_face : initial_contact) {
    initial_finger_face_contacts.emplace_back(finger_face.first,
                                              finger_face.second);
    AddFingerTipInContactWithBrickFace(
        *gripper_brick_, finger_face.first, finger_face.second, prog_.get(),
        q_vars_.col(0), plant_mutable_contexts_[0], options.face_shrink_factor);
  }
  auto f_FB_B0 = AddBrickStaticEquilibriumConstraint(
      *gripper_brick_, initial_finger_face_contacts, q_vars_.col(0),
      plant_mutable_contexts_[0], prog_.get());
  int initial_contact_count = 0;
  for (const auto& finger_face : initial_contact) {
    f_FB_B_[0].emplace(finger_face.first, f_FB_B0.col(initial_contact_count++));
  }

  // fingers_in_contact[i] stores the fingers in contact with the brick at knot
  // i.
  // Initialize to be all fingers in contact, and then go over the transitions
  // to remove the non-contact fingers.
  std::vector<std::set<Finger>> fingers_in_contact(
      nT_, {Finger::kFinger1, Finger::kFinger2, Finger::kFinger3});
  std::set<Finger> initial_contact_fingers;
  for (const auto& finger_face : initial_contact) {
    initial_contact_fingers.emplace(finger_face.first);
  }

  // Now go through each of the transition,
  // to find out the active contacts at each knots.
  int initial_transition_knot = nT_;
  for (const auto& finger_transition : finger_transitions) {
    if (finger_transition.end_knot_index - finger_transition.start_knot_index <
        2) {
      throw std::invalid_argument(
          "GripperBrickTrajectoryOptimization: finger transition must take at "
          "least two time intervals.");
    }
    if (finger_transition.start_knot_index < initial_transition_knot) {
      initial_transition_knot = finger_transition.start_knot_index;
    }
    for (int i = finger_transition.start_knot_index + 1;
         i < finger_transition.end_knot_index; ++i) {
      auto it = fingers_in_contact[i].find(finger_transition.finger);
      if (it == fingers_in_contact[i].end()) {
        throw std::invalid_argument(
            "GripperBrickTrajectoryOptimization: both transitions move " +
            to_string(finger_transition.finger) + " at knot " +
            std::to_string(i));
      }
      fingers_in_contact[i].erase(it);
    }
  }
  if (initial_transition_knot == 0) {
    throw std::invalid_argument(
        "GripperBrickTrajectoryOptimization: the initial transition cannot "
        "start with knot 0.");
  }
  for (int i = 0; i <= initial_transition_knot; ++i) {
    fingers_in_contact[i] = initial_contact_fingers;
  }
  // Now assign the decision variables for the contact forces.
  for (int i = 1 /* The contact force variable for the initial state has been
                    assigned */
       ;
       i < nT_; ++i) {
    for (const auto& finger : fingers_in_contact[i]) {
      auto f = prog_->NewContinuousVariables<2>();
      f_FB_B_[i].emplace(finger, f);
    }
  }
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
