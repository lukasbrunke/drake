#include "drake/examples/planar_gripper/gripper_brick_trajectory_optimization.h"

#include "drake/examples/planar_gripper/brick_static_equilibrium_constraint.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GripperBrickTrajectoryOptimization::GripperBrickTrajectoryOptimization(
    const GripperBrickSystem<double>* const gripper_brick, int nT,
    const std::unordered_map<Finger, BrickFace>& initial_contact,
    const std::vector<FingerTransition>& finger_transitions)
    : gripper_brick_{gripper_brick},
      nT_{nT},
      prog_{std::make_unique<solvers::MathematicalProgram>()},
      q_vars_{prog_->NewContinuousVariables(
          gripper_brick_->plant().num_positions(), nT_)},
      brick_v_y_vars_{prog_->NewContinuousVariables(nT_)},
      brick_v_z_vars_{prog_->NewContinuousVariables(nT_)},
      brick_omega_x_vars_{prog_->NewContinuousVariables(nT_)},
      f_FB_B_(nT_),
      plant_mutable_contexts_(nT_) {
  // Create the contexts used at each knot.
  diagram_contexts_.reserve(nT_);
  for (int i = 0; i < nT_; ++i) {
    diagram_contexts_.push_back(
        gripper_brick->diagram().CreateDefaultContext());
    plant_mutable_contexts_[i] =
        &(gripper_brick_->diagram().GetMutableSubsystemContext(
            gripper_brick_->plant(), diagram_contexts_[i].get()));
  }
  // I assume initially the system is in static equilibrium.
  std::vector<std::pair<Finger, BrickFace>> initial_finger_face_contacts;
  for (const auto& finger_face : initial_contact) {
    initial_finger_face_contacts.emplace_back(finger_face.first,
                                              finger_face.second);
  }
  AddBrickStaticEquilibriumConstraint(
      *gripper_brick_, initial_finger_face_contacts, q_vars_.col(0),
      plant_mutable_contexts_[0], prog_.get());

  // Now go through each of the transition, to find out the active contacts at
  // each knots.
  for (const auto& finger_transition : finger_transitions) {
    if (finger_transition.end_knot_index - finger_transition.start_knot_index <
        2) {
      throw std::invalid_argument(
          "GripperBrickTrajectoryOptimization: finger transition must take at "
          "least two time intervals.");
    }
  }
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
