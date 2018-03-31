#include "drake/manipulation/planner/object_contact_planning.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/manipulation/dev/remote_tree_viewer_wrapper.h"
#include "drake/manipulation/planner/friction_cone.h"
#include "drake/manipulation/planner/test/block_test_util.h"
#include "drake/math/rotation_matrix.h"
#include "drake/solvers/gurobi_solver.h"

using drake::symbolic::Expression;
using Eigen::Vector3d;
using Eigen::Matrix3d;

namespace drake {
namespace manipulation {
namespace planner {
GTEST_TEST(ObjectContactPlanningTest, TestStaticSinglePosture) {
  // Find a single posture, that the block is on the table, with bottom vertices
  // in contact with the table.
  Block block;
  const int num_pusher = 0;
  ObjectContactPlanning problem(1, block.mass(), block.center_of_mass(),
                                block.p_BV(), num_pusher, block.Q());

  // Constrain that all vertices of the block is above the table.
  AllVerticesAboveTable(block, &problem);

  const double mu_table = 1;
  const auto f_WV = SetTableContactVertices(
      block, block.positive_x_vertex_indices(), mu_table, 0, &problem);

  // At least one point in contact.
  problem.get_mutable_prog()->AddLinearConstraint(
      problem.vertex_contact_flag()[0].cast<Expression>().sum() >= 1);

  // Static equilibrium constraint.
  problem.AddStaticEquilibriumConstraint();

  // Now add a cost on all the vertex contact forces.
  const auto contact_force_cost =
      problem.get_mutable_prog()->NewContinuousVariables<1>(
          "contact_force_cost")(0);
  solvers::VectorDecisionVariable<1 + 3 * 4> contact_force_cost_lorentz_cone;
  contact_force_cost_lorentz_cone(0) = contact_force_cost;
  for (int i = 0; i < 4; ++i) {
    contact_force_cost_lorentz_cone.segment<3>(1 + 3 * i) =
        problem.f_BV()[0].col(i);
  }
  problem.get_mutable_prog()->AddLorentzConeConstraint(
      contact_force_cost_lorentz_cone);
  problem.get_mutable_prog()->AddLinearCost(+contact_force_cost);

  problem.get_mutable_prog()->AddBoundingBoxConstraint(0.5, 0.5, f_WV(0, 0));

  drake::solvers::GurobiSolver solver;
  problem.get_mutable_prog()->SetSolverOption(solvers::GurobiSolver::id(),
                                              "OutputFlag", 1);
  const auto solution_result = solver.Solve(*(problem.get_mutable_prog()));
  EXPECT_EQ(solution_result, drake::solvers::SolutionResult::kSolutionFound);

  // Now check the solution.
  const auto p_WB_sol = problem.prog().GetSolution(problem.p_WB()[0]);
  const auto R_WB_sol = problem.prog().GetSolution(problem.R_WB()[0]);

  // All vertices should be on or above the ground.
  const double tol = 1E-5;
  EXPECT_TRUE(
      ((R_WB_sol * block.p_BV()).row(2).array() >= -p_WB_sol(2) - tol).all());
  const Eigen::Matrix<double, 3, 8> p_WV_sol =
      p_WB_sol * Eigen::Matrix<double, 1, 8>::Ones() + R_WB_sol * block.p_BV();

  const Eigen::Vector4d color_gray(0.7, 0.7, 0.7, 0.9);
  const Eigen::Vector4d color_blue(0.3, 0.3, 1.0, 0.9);
  dev::RemoteTreeViewerWrapper viewer;
  viewer.PublishGeometry(DrakeShapes::MeshPoints(p_WV_sol),
                         Eigen::Affine3d::Identity(), color_gray, {"p_WV"});
  VisualizeBlock(&viewer, R_WB_sol, p_WB_sol, block);

  const auto f_BV_sol = problem.prog().GetSolution(problem.f_BV()[0]);
  const auto f_WV_sol = problem.prog().GetSolution(f_WV);
  for (int i = 0; i < 4; ++i) {
    VisualizeForce(&viewer, p_WV_sol.col(block.positive_x_vertex_indices()[i]),
                   f_WV_sol.col(i), block.mass() * kGravity * 5,
                   "f_WV" + std::to_string(i));
    VisualizeForce(&viewer, p_WV_sol.col(i), R_WB_sol * f_BV_sol.col(i),
                   block.mass() * kGravity * 5,
                   "R_WB * f_BV" + std::to_string(i));
  }

  // Make sure that static equilibrium is satisfied.
  const Eigen::Vector3d mg(0, 0, -block.mass() * kGravity);
  EXPECT_TRUE(CompareMatrices((R_WB_sol * f_BV_sol).rowwise().sum(), -mg, tol));
  const Eigen::Vector3d p_WC_sol = p_WB_sol + R_WB_sol * block.center_of_mass();
  Eigen::Vector3d total_torque = p_WC_sol.cross(mg);
  for (int i = 0; i < 4; ++i) {
    total_torque += p_WV_sol.col(i).cross(R_WB_sol * f_BV_sol.col(i));
  }
  EXPECT_TRUE(CompareMatrices(total_torque, Eigen::Vector3d::Zero(), tol));

  // Now make sure that R_WB approximatedly satisfies SO(3) constraint.
  double R_WB_quality_factor;
  const Matrix3d R_WB_project =
      math::RotationMatrix<double>::ProjectToRotationMatrix(
          R_WB_sol, &R_WB_quality_factor)
          .matrix();
  EXPECT_NEAR(R_WB_quality_factor, 1, 0.05);

  // Now make sure that f_WV ≈ R_WB * f_BV
  // Actually this constraint is violated a lot, in the -x and -y directions.
  // In the z direction, the difference is very small, to about 1E-10. But in
  // the x and y direction, the difference can be 0.4.
  EXPECT_TRUE(CompareMatrices(f_WV_sol, R_WB_sol * f_BV_sol, 0.35));
}

GTEST_TEST(ObjectContactPlanningTest, SinglePostureWithPushers) {
  Block block;
  const int num_pusher = 2;
  ObjectContactPlanning problem(1, block.mass(), block.center_of_mass(),
                                block.p_BV(), num_pusher, block.Q());

  // Constrain that all vertices of the block is above the table.
  AllVerticesAboveTable(block, &problem);

  const double mu_table = 1;
  const auto f_WV = SetTableContactVertices(
      block, block.bottom_vertex_indices(), mu_table, 0, &problem);

  // Choose all the body contact points except those on the top or bottom.
  std::vector<int> pusher_contact_point_indices = {2, 3, 4, 5, 14, 15, 16, 17};
  problem.SetPusherContactPointIndices(0, pusher_contact_point_indices,
                                       block.mass() * kGravity);

  // Static equilibrium constraint.
  problem.AddStaticEquilibriumConstraint();

  // Now add a cost on all the vertex contact forces.
  const auto contact_force_cost =
      problem.get_mutable_prog()->NewContinuousVariables<1>(
          "contact_force_cost")(0);
  solvers::VectorDecisionVariable<1 + 3 * 4> contact_force_cost_lorentz_cone;
  contact_force_cost_lorentz_cone(0) = contact_force_cost;
  for (int i = 0; i < 4; ++i) {
    contact_force_cost_lorentz_cone.segment<3>(1 + 3 * i) =
        problem.f_BV()[0].col(i);
  }
  problem.get_mutable_prog()->AddLorentzConeConstraint(
      contact_force_cost_lorentz_cone);
  problem.get_mutable_prog()->AddLinearCost(+contact_force_cost);

  problem.get_mutable_prog()->SetSolverOption(solvers::GurobiSolver::id(),
                                              "OutputFlag", 1);

  solvers::GurobiSolver solver;
  const auto solver_result = solver.Solve(*(problem.get_mutable_prog()));
  EXPECT_EQ(solver_result, solvers::SolutionResult::kSolutionFound);

  const auto p_WB_sol = problem.prog().GetSolution(problem.p_WB()[0]);
  const auto R_WB_sol = problem.prog().GetSolution(problem.R_WB()[0]);

  dev::RemoteTreeViewerWrapper viewer;
  VisualizeBlock(&viewer, R_WB_sol, p_WB_sol, block);

  const Eigen::Matrix<double, 3, 8> p_WV_sol =
      p_WB_sol * Eigen::Matrix<double, 1, 8>::Ones() + R_WB_sol * block.p_BV();
  const auto f_WV_sol = problem.prog().GetSolution(f_WV);
  const double viewer_force_normalizer = block.mass() * kGravity * 5;
  for (int i = 0; i < 4; ++i) {
    VisualizeForce(&viewer, p_WV_sol.col(block.bottom_vertex_indices()[i]),
                   f_WV_sol.col(i), viewer_force_normalizer,
                   "f_WV" + std::to_string(i));
  }

  const auto f_BQ_sol = problem.prog().GetSolution(problem.f_BQ()[0]);
  const auto f_WQ_sol = R_WB_sol * f_BQ_sol;
  Eigen::Matrix3Xd p_WQ_sol(3, pusher_contact_point_indices.size());
  for (int i = 0; i < p_WQ_sol.cols(); ++i) {
    p_WQ_sol.col(i) =
        p_WB_sol + R_WB_sol * block.Q()[pusher_contact_point_indices[i]].p_BQ();
    VisualizeForce(&viewer, p_WQ_sol.col(i), f_WQ_sol.col(i),
                   viewer_force_normalizer, "f_WQ" + std::to_string(i));
  }

  // Check the solution.
  // All vertices above the ground.
  const double tol = 1E-5;
  EXPECT_TRUE((p_WV_sol.row(2).array() >= -tol).all());

  // No table contact force at the vertices.
  const auto f_BV_sol = problem.prog().GetSolution(problem.f_BV()[0]);
  EXPECT_TRUE(
      CompareMatrices(f_WV_sol, Eigen::Matrix<double, 3, 4>::Zero(), tol));
  EXPECT_TRUE(
      CompareMatrices(f_BV_sol, Eigen::Matrix<double, 3, 4>::Zero(), tol));
  // At most two pusher contact points have non-zero contact forces.
  int nonzero_force_count = 0;
  for (int i = 0; i < f_BQ_sol.cols(); ++i) {
    if (f_BQ_sol.col(i).norm() > tol) {
      nonzero_force_count++;
    }
  }
  EXPECT_LE(nonzero_force_count, 2);

  // Check static equilibrium condition.
  const Eigen::Vector3d mg_W(0, 0, -block.mass() * kGravity);
  Eigen::Vector3d total_force = R_WB_sol.transpose() * mg_W;
  Eigen::Vector3d total_torque = block.center_of_mass().cross(total_force);
  for (int i = 0; i < f_BQ_sol.cols(); ++i) {
    total_force += f_BQ_sol.col(i);
    total_torque += block.Q()[pusher_contact_point_indices[i]].p_BQ().cross(
        f_BQ_sol.col(i));
  }
  EXPECT_TRUE(CompareMatrices(total_force, Eigen::Vector3d::Zero(), tol));
  EXPECT_TRUE(CompareMatrices(total_torque, Eigen::Vector3d::Zero(), tol));
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
