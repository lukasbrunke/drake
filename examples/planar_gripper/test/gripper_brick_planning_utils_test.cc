#include "drake/examples/planar_gripper/gripper_brick_planning_utils.h"

#include <gtest/gtest.h>

#include "drake/solvers/solve.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GTEST_TEST(TestAddFingerNoSlidingConstraint, Test) {
  GripperBrickHelper<double> gripper_brick;
  const Finger finger{Finger::kFinger1};
  const BrickFace face{BrickFace::kNegZ};
  solvers::MathematicalProgram prog;
  auto q_from =
      prog.NewContinuousVariables(gripper_brick.plant().num_positions());
  auto q_to =
      prog.NewContinuousVariables(gripper_brick.plant().num_positions());
  auto diagram_context_from = gripper_brick.diagram().CreateDefaultContext();
  auto diagram_context_to = gripper_brick.diagram().CreateDefaultContext();
  systems::Context<double>* plant_context_from =
      &(gripper_brick.diagram().GetMutableSubsystemContext(
          gripper_brick.plant(), diagram_context_from.get()));
  systems::Context<double>* plant_context_to =
      &(gripper_brick.diagram().GetMutableSubsystemContext(
          gripper_brick.plant(), diagram_context_to.get()));
  const double face_shrink_factor{0.8};
  AddFingerTipInContactWithBrickFace(gripper_brick, finger, face, &prog, q_from,
                                     plant_context_from, face_shrink_factor);
  const double rolling_angle_bound{0.1 * M_PI};
  AddFingerNoSlidingConstraint(gripper_brick, finger, face, rolling_angle_bound,
                               &prog, plant_context_from, plant_context_to,
                               q_from, q_to, face_shrink_factor);

  // Now solve the problem.
  const auto result = solvers::Solve(prog);
  EXPECT_TRUE(result.is_success());

  // Make sure both "from" posture and "to" postures are in contact with the
  // brick face.
  auto check_finger_in_contact =
      [&gripper_brick, finger, &result, face_shrink_factor](
          const VectorX<symbolic::Variable>& q,
          systems::Context<double>* plant_context) -> math::RigidTransformd {
    const Eigen::VectorXd q_sol = result.GetSolution(q);
    gripper_brick.plant().SetPositions(plant_context, q_sol);
    const math::RigidTransform<double> X_BL2 =
        gripper_brick.plant().CalcRelativeTransform(
            *plant_context, gripper_brick.brick_frame(),
            gripper_brick.finger_link2_frame(finger));
    const Eigen::Vector3d p_BTip = X_BL2 * gripper_brick.p_L2Tip();
    const Eigen::Vector3d brick_size = gripper_brick.brick_size();
    const double depth = 1E-3;
    EXPECT_NEAR(p_BTip(2),
                -brick_size(2) / 2 - gripper_brick.finger_tip_radius() + depth,
                1E-4);
    EXPECT_GE(p_BTip(1), -brick_size(1) / 2 * face_shrink_factor - 1E-4);
    EXPECT_LE(p_BTip(1), brick_size(1) / 2 * face_shrink_factor + 1E-4);
    return X_BL2;
  };

  const math::RigidTransformd X_BL2_from =
      check_finger_in_contact(q_from, plant_context_from);
  const math::RigidTransformd X_BL2_to =
      check_finger_in_contact(q_to, plant_context_to);

  // Check the orientation difference between "from" posture and "to" posture.
  const Eigen::AngleAxisd angle_axis =
      (X_BL2_from.rotation().inverse() * X_BL2_to.rotation()).ToAngleAxis();
  double theta;
  if (angle_axis.axis().dot(Eigen::Vector3d::UnitX()) > 1 - 1E-4) {
    theta = angle_axis.angle();
  } else if (angle_axis.axis().dot(-Eigen::Vector3d::UnitX()) > 1 - 1E-4) {
    theta = -angle_axis.angle();
  }
  EXPECT_GE(theta, -rolling_angle_bound - 1E-5);
  EXPECT_LE(theta, rolling_angle_bound + 1E-5);

  const Eigen::Vector3d p_BTip_from = X_BL2_from * gripper_brick.p_L2Tip();
  const Eigen::Vector3d p_BTip_to = X_BL2_to * gripper_brick.p_L2Tip();
  EXPECT_NEAR(p_BTip_to(1) - p_BTip_from(1),
              -gripper_brick.finger_tip_radius() * theta, 1E-5);
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
