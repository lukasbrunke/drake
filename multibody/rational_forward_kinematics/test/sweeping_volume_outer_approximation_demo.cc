#include "drake/multibody/rational_forward_kinematics/sweeping_volume_outer_approximation.h"

#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

namespace drake {
namespace multibody {
int DoMain() {
  // weld the schunk gripper to iiwa link 7.
  Eigen::Isometry3d X_7S =
      Eigen::Translation3d(0, 0, 0.1) *
      Eigen::AngleAxisd(-21.0 / 180 * M_PI, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitX()) *
      Eigen::Isometry3d::Identity();
  auto plant = std::make_unique<MultibodyPlant<double>>();
  auto scene_graph = std::make_unique<geometry::SceneGraph<double>>();
  plant->RegisterAsSourceForSceneGraph(scene_graph.get());
  AddIiwaWithSchunk(X_7S, plant.get());
  plant->Finalize(scene_graph.get());
  std::array<BodyIndex, 8> iiwa_link;
  for (int i = 0; i < 8; ++i) {
    iiwa_link[i] =
        plant->GetBodyByName("iiwa_link_" + std::to_string(i)).index();
  }

  // Schunk gripper points.
  Eigen::Matrix<double, 3, 8> p_SV;
  p_SV.col(0) << -0.065, -0.035, 0.02;
  p_SV.col(1) << 0.065, -0.035, 0.02;
  p_SV.col(2) << -0.065, -0.035, -0.02;
  p_SV.col(3) << 0.065, -0.035, -0.02;
  p_SV.col(4) << -0.065, 0.105, 0.02;
  p_SV.col(5) << 0.065, 0.105, 0.02;
  p_SV.col(6) << -0.065, 0.105, -0.02;
  p_SV.col(7) << 0.065, 0.105, -0.02;
  Eigen::Matrix<double, 3, 8> p_7V = X_7S * p_SV;

  std::vector<std::shared_ptr<const ConvexPolytope>> link_polytopes;
  auto schunk_polytope =
      std::make_shared<const ConvexPolytope>(iiwa_link[7], p_7V);
  link_polytopes.push_back(schunk_polytope);

  Eigen::VectorXd q_star(7);
  q_star << 0.1, 0.2, 0.3, 0.1, -0.2, 0.3, 0.2;

  SweepingVolumeOuterApproximation dut(*plant, link_polytopes, q_star);
  Eigen::VectorXd q_upper = q_star + 0.1 * Eigen::VectorXd::Ones(7);
  Eigen::VectorXd q_lower = q_star - 0.1 * Eigen::VectorXd::Ones(7);

  SweepingVolumeOuterApproximation::VerificationOption verification_option{};
  verification_option.link_polynomial_type =
      solvers::MathematicalProgram::NonnegativePolynomial::kSos;
  verification_option.lagrangian_type =
      solvers::MathematicalProgram::NonnegativePolynomial::kSos;
  std::cout << dut.FindSweepingVolumeMaximalProjection(
      schunk_polytope->get_id(), Eigen::Vector3d::UnitZ(), q_lower, q_upper,
      verification_option);

  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
