#include "drake/examples/planar_gripper/gripper_brick_planning_utils.h"

#include "drake/multibody/inverse_kinematics/position_constraint.h"

namespace drake {
namespace examples {
namespace planar_gripper {
template <typename T>
void AddFrictionConeConstraint(
    const GripperBrickSystem<T>& gripper_brick_system, const Finger finger,
    const BrickFace brick_face,
    const Eigen::Ref<const Vector2<symbolic::Variable>>& f_Cb_B,
    solvers::MathematicalProgram* prog) {
  const auto& plant = gripper_brick_system.plant();
  const multibody::CoulombFriction<double>& brick_friction =
      plant.default_coulomb_friction(plant.GetCollisionGeometriesForBody(
          gripper_brick_system.brick_frame().body())[0]);
  const multibody::CoulombFriction<double>& finger_tip_friction =
      plant.default_coulomb_friction(plant.GetCollisionGeometriesForBody(
          gripper_brick_system.finger_link2_frame(finger).body())[0]);
  const multibody::CoulombFriction<double> combined_friction =
      multibody::CalcContactFrictionFromSurfaceProperties(brick_friction,
                                                          finger_tip_friction);
  const double mu = combined_friction.static_friction();
  switch (brick_face) {
    case BrickFace::kNegY: {
      prog->AddLinearConstraint(f_Cb_B(0) >= 0);
      prog->AddLinearConstraint(f_Cb_B(1) <= mu * f_Cb_B(0));
      prog->AddLinearConstraint(f_Cb_B(1) >= -mu * f_Cb_B(0));
      break;
    }
    case BrickFace::kNegZ: {
      prog->AddLinearConstraint(f_Cb_B(1) >= 0);
      prog->AddLinearConstraint(f_Cb_B(0) <= mu * f_Cb_B(1));
      prog->AddLinearConstraint(f_Cb_B(0) >= -mu * f_Cb_B(1));
      break;
    }
    case BrickFace::kPosY: {
      prog->AddLinearConstraint(f_Cb_B(0) <= 0);
      prog->AddLinearConstraint(f_Cb_B(1) <= -mu * f_Cb_B(0));
      prog->AddLinearConstraint(f_Cb_B(1) >= mu * f_Cb_B(0));
      break;
    }
    case BrickFace::kPosZ: {
      prog->AddLinearConstraint(f_Cb_B(1) <= 0);
      prog->AddLinearConstraint(f_Cb_B(0) <= -mu * f_Cb_B(1));
      prog->AddLinearConstraint(f_Cb_B(0) >= mu * f_Cb_B(1));
      break;
    }
  }
}

void AddFingerTipInContactWithBrickFace(
    const GripperBrickSystem<double>& gripper_brick_system, Finger finger,
    BrickFace brick_face, solvers::MathematicalProgram* prog,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& q_vars,
    systems::Context<double>* plant_context, double face_shrink_factor) {
  const multibody::Frame<double>& finger_link2 =
      gripper_brick_system.finger_link2_frame(finger);
  // position of Tip in the finger link 2 farme (F2).
  const Eigen::Vector3d p_F2Tip = gripper_brick_system.p_F2Tip();
  const multibody::Frame<double>& brick = gripper_brick_system.brick_frame();
  const Eigen::Vector3d brick_size = gripper_brick_system.brick_size();
  Eigen::Vector3d p_BTip_lower = -brick_size * face_shrink_factor / 2;
  Eigen::Vector3d p_BTip_upper = brick_size * face_shrink_factor / 2;
  const double finger_tip_radius = gripper_brick_system.finger_tip_radius();
  const double depth = 1E-3;
  switch (brick_face) {
    case BrickFace::kPosZ: {
      p_BTip_lower(2) = brick_size(2) / 2 + finger_tip_radius - depth;
      p_BTip_upper(2) = brick_size(2) / 2 + finger_tip_radius - depth;
      break;
    }
    case BrickFace::kNegZ: {
      p_BTip_lower(2) = -brick_size(2) / 2 - finger_tip_radius + depth;
      p_BTip_upper(2) = -brick_size(2) / 2 - finger_tip_radius + depth;
      break;
    }
    case BrickFace::kPosY: {
      p_BTip_lower(1) = brick_size(1) / 2 + finger_tip_radius - depth;
      p_BTip_upper(1) = brick_size(1) / 2 + finger_tip_radius - depth;
      break;
    }
    case BrickFace::kNegY: {
      p_BTip_lower(1) = -brick_size(1) / 2 - finger_tip_radius + depth;
      p_BTip_upper(1) = -brick_size(1) / 2 - finger_tip_radius + depth;
      break;
    }
  }
  prog->AddConstraint(std::make_shared<multibody::PositionConstraint>(
                          &(gripper_brick_system.plant()), brick, p_BTip_lower,
                          p_BTip_upper, finger_link2, p_F2Tip, plant_context),
                      q_vars);
}

template void AddFrictionConeConstraint<double>(
    const GripperBrickSystem<double>&, Finger, BrickFace,
    const Eigen::Ref<const Vector2<symbolic::Variable>>&,
    solvers::MathematicalProgram*);
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
