#include "drake/manipulation/planner/quasi_dynamic_object_contact_planning.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/manipulation/planner/test/block_test_util.h"
#include "drake/math/rotation_matrix.h"
#include "drake/solvers/gurobi_solver.h"

namespace drake {
namespace manipulation {
namespace planner {
// Return the skew symmetric matrix V, that represents the cross product with
// a vector v. namely v x u = Vu.
Eigen::Matrix3d SkewSymmetric(const Eigen::Ref<const Eigen::Vector3d>& v) {
  Eigen::Matrix3d V;
  // clang-format off
  V << 0, -v(2), v(1),
       v(2), 0, -v(0),
       -v(1), v(0), 0;
  // clang-format on
  return V;
}

// Given initial pose and velocity, together with the final velocity, compute
// the final pose that would satisfy the mid-point interpolation.
void ComputeFinalPoseUsingMidPointInterpolation(
    double dt, const Eigen::Ref<const Eigen::Vector3d>& p_WB0,
    const Eigen::Ref<const Eigen::Matrix3d>& R_WB0,
    const Eigen::Ref<const Eigen::Vector3d>& v_B0,
    const Eigen::Ref<const Eigen::Vector3d>& omega_B0,
    const Eigen::Ref<const Eigen::Vector3d>& v_B1,
    const Eigen::Ref<const Eigen::Vector3d>& omega_B1, Eigen::Vector3d* p_WB1,
    Eigen::Matrix3d* R_WB1) {
  // The mid point interpolation for orientation is
  // R_WB1 - R_WB0 = (R_WB1 + R_WB0) * (SkewSymmetric((omega_B0 + omega_B1)/2))
  // * dt / 2
  const Eigen::Vector3d omega_average = (omega_B0 + omega_B1) / 2;
  *R_WB1 =
      R_WB0 *
      (Eigen::Matrix3d::Identity() + SkewSymmetric(omega_average * dt / 2)) *
      ((Eigen::Matrix3d::Identity() - SkewSymmetric(omega_average * dt / 2))
           .inverse());

  // The mid point interpolation for position is
  // p_WB1 - p_WB0 = (R_WB1 * v_B1 + R_WB0 * v_B0) * dt / 2
  *p_WB1 = p_WB0 + (*R_WB1 * v_B1 + R_WB0 * v_B0) * dt / 2;
}

GTEST_TEST(QuasiDynamicObjectContactPlanningTest, TestInterpolation) {
  Block block;
  const int nT = 2;
  const double dt = 0.1;
  const int num_pushers = 0;
  const double max_linear_velocity = 1;
  const double max_angular_velocity = M_PI;
  QuasiDynamicObjectContactPlanning problem(
      nT, dt, block.mass(), block.I_B(), block.center_of_mass(), block.p_BV(),
      num_pushers, block.Q(), max_linear_velocity, max_angular_velocity, false);

  auto p_WB0_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), problem.p_WB()[0]);
  auto p_WB1_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), problem.p_WB()[1]);
  auto R_WB0_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      0, 0, problem.R_WB()[0]);
  auto R_WB1_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      0, 0, problem.R_WB()[1]);
  auto v_B0_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), problem.v_B().col(0));
  auto v_B1_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), problem.v_B().col(1));
  auto omega_B0_constraint =
      problem.get_mutable_prog()->AddBoundingBoxConstraint(
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          problem.omega_B().col(0));
  auto omega_B1_constraint =
      problem.get_mutable_prog()->AddBoundingBoxConstraint(
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          problem.omega_B().col(1));

  auto CheckFeasibility = [&p_WB0_constraint, &p_WB1_constraint,
                           &R_WB0_constraint, &R_WB1_constraint,
                           &v_B0_constraint, &v_B1_constraint,
                           &omega_B0_constraint, &omega_B1_constraint](
      solvers::MathematicalProgram* prog,
      const Eigen::Ref<const Eigen::Vector3d>& p_WB0,
      const Eigen::Ref<const Eigen::Vector3d>& p_WB1,
      const Eigen::Ref<const Eigen::Matrix3d>& R_WB0,
      const Eigen::Ref<const Eigen::Matrix3d>& R_WB1,
      const Eigen::Ref<const Eigen::Vector3d>& v_B0,
      const Eigen::Ref<const Eigen::Vector3d>& v_B1,
      const Eigen::Ref<const Eigen::Vector3d>& omega_B0,
      const Eigen::Ref<const Eigen::Vector3d>& omega_B1, bool feasible) {
    auto UpdateEqualityBound = [](
        solvers::BoundingBoxConstraint* constraint,
        const Eigen::Ref<const Eigen::VectorXd>& bounds) {
      constraint->UpdateLowerBound(bounds);
      constraint->UpdateUpperBound(bounds);
    };
    UpdateEqualityBound(p_WB0_constraint.evaluator().get(), p_WB0);
    UpdateEqualityBound(p_WB1_constraint.evaluator().get(), p_WB1);
    Eigen::Matrix<double, 9, 1> R_WB0_flat, R_WB1_flat;
    R_WB0_flat << R_WB0.col(0), R_WB0.col(1), R_WB0.col(2);
    R_WB1_flat << R_WB1.col(0), R_WB1.col(1), R_WB1.col(2);
    UpdateEqualityBound(R_WB0_constraint.evaluator().get(), R_WB0_flat);
    UpdateEqualityBound(R_WB1_constraint.evaluator().get(), R_WB1_flat);
    UpdateEqualityBound(v_B0_constraint.evaluator().get(), v_B0);
    UpdateEqualityBound(v_B1_constraint.evaluator().get(), v_B1);
    UpdateEqualityBound(omega_B0_constraint.evaluator().get(), omega_B0);
    UpdateEqualityBound(omega_B1_constraint.evaluator().get(), omega_B1);

    prog->SetSolverOption(solvers::GurobiSolver::id(), "DualReductions", 0);
    const auto solution_result = prog->Solve();
    EXPECT_EQ(solution_result,
              feasible ? solvers::SolutionResult::kSolutionFound
                       : solvers::SolutionResult::kInfeasibleConstraints);
  };

  // Case 1, static
  CheckFeasibility(problem.get_mutable_prog(), Eigen::Vector3d::Ones(),
                   Eigen::Vector3d::Ones(), Eigen::Matrix3d::Identity(),
                   Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
                   Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                   Eigen::Vector3d::Zero(), true);

  Eigen::Matrix3d R_WB0 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p_WB0(0, 0, 0);
  Eigen::Vector3d v_B0(0.8, 0.9, 0.7);
  Eigen::Vector3d omega_B0(0, 0, 0);
  Eigen::Matrix3d R_WB1;
  Eigen::Vector3d p_WB1;
  Eigen::Vector3d v_B1(0.2, 0.3, 0.7);
  Eigen::Vector3d omega_B1(0, 0, 0);
  ComputeFinalPoseUsingMidPointInterpolation(dt, p_WB0, R_WB0, v_B0, omega_B0,
                                             v_B1, omega_B1, &p_WB1, &R_WB1);
  // Case 2, only translational motion, no rotational motion.
  CheckFeasibility(problem.get_mutable_prog(), p_WB0, p_WB1, R_WB0, R_WB1, v_B0,
                   v_B1, omega_B0, omega_B1, true);

  // Case 3, only translational motion, no rotational motion. The orientation
  // is not identity.
  R_WB0 = Eigen::AngleAxisd(0.2, Eigen::Vector3d(0.1, 0.2, 0.3).normalized())
              .toRotationMatrix();
  ComputeFinalPoseUsingMidPointInterpolation(dt, p_WB0, R_WB0, v_B0, omega_B0,
                                             v_B1, omega_B1, &p_WB1, &R_WB1);
  CheckFeasibility(problem.get_mutable_prog(), p_WB0, p_WB1, R_WB0, R_WB1, v_B0,
                   v_B1, omega_B0, omega_B1, true);

  // Case 4, only rotational motion, no translational motion, same angular
  // velocity
  R_WB0 = Eigen::Matrix3d::Identity();
  p_WB0 = Eigen::Vector3d::Zero();
  v_B0 = Eigen::Vector3d::Zero();
  omega_B0 = Eigen::Vector3d(0.2, 0.4, 1.2);
  v_B1 = Eigen::Vector3d::Zero();
  omega_B1 = Eigen::Vector3d(0.2, 0.4, 1.2);
  ComputeFinalPoseUsingMidPointInterpolation(dt, p_WB0, R_WB0, v_B0, omega_B0,
                                             v_B1, omega_B1, &p_WB1, &R_WB1);
  CheckFeasibility(problem.get_mutable_prog(), p_WB0, p_WB1, R_WB0, R_WB1, v_B0,
                   v_B1, omega_B0, omega_B1, true);

  // Case 5, only rotation motion, no translational motion, different angular
  // velocity.
  omega_B1 = Eigen::Vector3d(0.1, 0.5, 1.4);
  ComputeFinalPoseUsingMidPointInterpolation(dt, p_WB0, R_WB0, v_B0, omega_B0,
                                             v_B1, omega_B1, &p_WB1, &R_WB1);
  CheckFeasibility(problem.get_mutable_prog(), p_WB0, p_WB1, R_WB0, R_WB1, v_B0,
                   v_B1, omega_B0, omega_B1, true);

  // Case 6, with both rotation and translational motion.
  v_B0 = Eigen::Vector3d(0.4, -0.1, 0.5);
  v_B1 = Eigen::Vector3d(0.2, -0.4, 0.3);
  ComputeFinalPoseUsingMidPointInterpolation(dt, p_WB0, R_WB0, v_B0, omega_B0,
                                             v_B1, omega_B1, &p_WB1, &R_WB1);
  CheckFeasibility(problem.get_mutable_prog(), p_WB0, p_WB1, R_WB0, R_WB1, v_B0,
                   v_B1, omega_B0, omega_B1, true);
}

GTEST_TEST(QuasiDynamicObjectContactPlanningTest, TestQuasiDynamics) {
  Block block;
  const int nT = 2;
  const double dt = 0.1;
  const int num_pushers = 2;
  const double max_linear_velocity = 1;
  const double max_angular_velocity = M_PI;
  QuasiDynamicObjectContactPlanning problem(
      nT, dt, block.mass(), block.I_B(), block.center_of_mass(), block.p_BV(),
      num_pushers, block.Q(), max_linear_velocity, max_angular_velocity, false);

  // Set the active vertices
  problem.SetContactVertexIndices(0, block.bottom_vertex_indices(), 0.05);
  problem.SetContactVertexIndices(
      1, block.bottom_and_positive_x_vertex_indices(), 0.05);

  // Set the pusher contact point.
  problem.SetPusherContactPointIndices(0, {0, 1, 2}, block.mass() * kGravity);
  problem.SetPusherContactPointIndices(1, {0, 2, 3}, block.mass() * kGravity);

  problem.AddQuasiDynamicConstraint();

  problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), problem.p_WB()[0]);
  problem.get_mutable_prog()->AddLinearEqualityConstraint(
      problem.R_WB()[0].cast<symbolic::Expression>(),
      Eigen::Matrix3d::Identity());

  // Set the initial angular velocity to some arbitrary value.
  const Eigen::Vector3d omega0(0.2, 0.4, 0.5);
  problem.get_mutable_prog()->AddBoundingBoxConstraint(
      omega0, omega0, problem.omega_B().col(0));

  const auto solution_result = problem.get_mutable_prog()->Solve();

  EXPECT_EQ(solution_result, solvers::SolutionResult::kSolutionFound);

  const auto f_BV0_sol = problem.prog().GetSolution(problem.f_BV()[0]);
  const auto f_BV1_sol = problem.prog().GetSolution(problem.f_BV()[1]);

  const auto f_BQ0_sol = problem.prog().GetSolution(problem.f_BQ()[0]);
  const auto f_BQ1_sol = problem.prog().GetSolution(problem.f_BQ()[1]);

  const auto R_WB0_sol = problem.prog().GetSolution(problem.R_WB()[0]);
  const auto R_WB1_sol = problem.prog().GetSolution(problem.R_WB()[1]);

  const auto p_WB0_sol = problem.prog().GetSolution(problem.p_WB()[0]);
  const auto p_WB1_sol = problem.prog().GetSolution(problem.p_WB()[1]);
  std::cout << "p_WB0_sol: " << p_WB0_sol.transpose()
            << "\n p_WB1_sol: " << p_WB1_sol.transpose() << "\n";

  const auto v_B0_sol = problem.prog().GetSolution(problem.v_B().col(0));
  const auto v_B1_sol = problem.prog().GetSolution(problem.v_B().col(1));
  std::cout << "v_B0_sol: " << v_B0_sol.transpose()
            << "\n v_B1_sol: " << v_B1_sol.transpose() << "\n";

  const auto omega_B0_sol =
      problem.prog().GetSolution(problem.omega_B().col(0));
  const auto omega_B1_sol =
      problem.prog().GetSolution(problem.omega_B().col(1));
  std::cout << "omega_B0_sol: " << omega_B0_sol.transpose()
            << "\n omega_B1_sol: " << omega_B1_sol.transpose() << "\n";

  EXPECT_TRUE(CompareMatrices(
      p_WB1_sol - p_WB0_sol,
      (R_WB0_sol * v_B0_sol + R_WB1_sol * v_B1_sol) / 2 * dt, 0.01));

  const Eigen::Vector3d mg(0, 0, -block.mass() * kGravity);
  // const Eigen::Vector3d total_force0 = R_WB0_sol.transpose() * mg +
  //                                     f_BV0_sol.rowwise().sum() +
  //                                     f_BQ0_sol.rowwise().sum();
  const Eigen::Vector3d total_force1 = R_WB1_sol.transpose() * mg +
                                       f_BV1_sol.rowwise().sum() +
                                       f_BQ1_sol.rowwise().sum();
  EXPECT_TRUE(CompareMatrices(block.mass() * (v_B1_sol - v_B0_sol),
                              total_force1 * dt, 1E-12));

  Eigen::Vector3d total_torque1 =
      block.center_of_mass().cross(R_WB1_sol.transpose() * mg);
  for (int i = 0;
       i < static_cast<int>(problem.contact_vertex_indices()[1].size()); ++i) {
    total_torque1 += block.p_BV()
                         .col(problem.contact_vertex_indices()[1][i])
                         .cross(f_BV1_sol.col(i));
  }
  for (int i = 0; i < static_cast<int>(problem.contact_Q_indices()[1].size());
       ++i) {
    total_torque1 += block.Q()[problem.contact_Q_indices()[1][i]].p_BQ().cross(
        f_BQ1_sol.col(i));
  }
  EXPECT_TRUE(CompareMatrices(block.I_B() * (omega_B1_sol - omega_B0_sol),
                              total_torque1 * dt, 1E-12));
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
