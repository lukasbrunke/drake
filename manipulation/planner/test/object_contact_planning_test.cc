#include "drake/manipulation/planner/object_contact_planning.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/manipulation/dev/remote_tree_viewer_wrapper.h"
#include "drake/math/rotation_matrix.h"
#include "drake/solvers/gurobi_solver.h"

using drake::symbolic::Expression;
using Eigen::Vector3d;
using Eigen::Matrix3d;

namespace drake {
namespace manipulation {
namespace planner {
class Block {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Block)

  Block() {
    // clang-format off
    p_BV_ << 1, 1, 1, 1, -1, -1, -1, -1,
             1, 1, -1, -1, 1, 1, -1, -1,
             1, -1, 1, -1, 1, -1, 1, -1;
    // clang-format on
    p_BV_.topRows<2>() *= width() / 2;
    p_BV_.row(2) *= height() / 2;
  }

  const Eigen::Matrix<double, 3, 8>& p_BV() const { return p_BV_; }

  double width() const { return 0.1; }

  double height() const { return 0.15; }

  double mass() const { return 1; }

  Vector3d center_of_mass() const { return Vector3d::Zero(); }

  std::vector<int> bottom_vertex_indices() const { return {1, 3, 5, 7}; }

  std::vector<int> top_vertex_indices() const { return {0, 2, 4, 6}; }

  std::vector<int> positive_x_vertex_indices() const { return {0, 1, 2, 3}; }

 private:
  Eigen::Matrix<double, 3, 8> p_BV_;
};

void VisualizeBlock(dev::RemoteTreeViewerWrapper* viewer,
                    const Eigen::Ref<const Matrix3d>& R_WB,
                    const Eigen::Ref<const Vector3d>& p_WB,
                    const Block& block) {
  Eigen::Affine3d tf;
  tf.linear() = R_WB;
  tf.translation() = p_WB;
  const Eigen::Vector4d color_blue(0.3, 0.3, 1, 0.9);
  viewer->PublishGeometry(
      DrakeShapes::Box({block.width(), block.width(), block.height()}), tf,
      color_blue, {"block"});
}

void VisualizeForce(dev::RemoteTreeViewerWrapper* viewer,
                    const Eigen::Ref<const Vector3d>& p_WP,
                    const Eigen::Ref<const Vector3d>& f_WP, double normalizer,
                    const std::string& path) {
  const Eigen::Vector3d f_WP_normalized = f_WP / normalizer;
  const std::vector<double> color_red = {1, 0.2, 0.2};
  if (f_WP_normalized.norm() > 1E-3) {
    viewer->PublishArrow(p_WP, p_WP + f_WP_normalized, {path}, 0.01, 0.02,
                         0.01);
  } else {
    viewer->PublishPointCloud(p_WP, {path}, {color_red});
  }
}

GTEST_TEST(ObjectContactPlanningTest, TestStaticSinglePosture) {
  // Find a single posture, that the block is on the table, with bottom vertices
  // in contact with the table.
  Block block;
  ObjectContactPlanning problem(1, block.mass(), block.center_of_mass(),
                                block.p_BV());

  // Constrain that all vertices of the block is above the table.
  for (int i = 0; i < 8; ++i) {
    problem.get_mutable_prog()->AddLinearConstraint(
        (problem.p_WB()[0] + problem.R_WB()[0] * block.p_BV().col(i))(2) >= 0);
  }

  problem.SetContactVertexIndices(0, block.positive_x_vertex_indices(),
                                  block.mass() * kGravity * 2);

  // At least one point in contact.
  problem.get_mutable_prog()->AddLinearConstraint(
      problem.vertex_contact_flag()[0].cast<Expression>().sum() >= 1);
  // Add the big-M constraint for the contact distance and the binary variable.
  // p_WV_z <= M * (1 - b) where b is the binary variable indicating contact.
  const double distance_big_M{0.05};
  for (int i = 0; i < 4; ++i) {
    const int vertex = block.positive_x_vertex_indices()[i];
    problem.get_mutable_prog()->AddLinearConstraint(
        (problem.p_WB()[0] + problem.R_WB()[0] * block.p_BV().col(vertex))(2) <=
        distance_big_M * (1 - problem.vertex_contact_flag()[0](i)));
  }

  // Compute the vertex contact force in the world frame f_WV.
  constexpr int num_bottom_vertices = 4;
  auto f_WV = problem.get_mutable_prog()
                  ->NewContinuousVariables<3, num_bottom_vertices>("f_WV");
  std::array<Eigen::VectorXd, 3> phi_f;
  phi_f[0] =
      Eigen::Matrix<double, 5, 1>::LinSpaced(-1, 1) * block.mass() * kGravity;
  phi_f[1] =
      Eigen::Matrix<double, 5, 1>::LinSpaced(-1, 1) * block.mass() * kGravity;
  phi_f[2] =
      Eigen::Matrix<double, 3, 1>::LinSpaced(0, 1) * block.mass() * kGravity;
  const double mu_table = 1;
  for (int i = 0; i < num_bottom_vertices; ++i) {
    problem.CalcContactForceInWorldFrame(problem.f_BV()[0].col(i), f_WV.col(i),
                                         0, false, phi_f);
    // Add friction cone constraint on f_WV
    AddFrictionConeConstraint(mu_table, Vector3d::UnitZ(),
                              f_WV.col(i).cast<Expression>(),
                              problem.get_mutable_prog());
  }
  // Compute the contact force at the vertices in the world frame.
  std::array<Eigen::VectorXd, 3> phi_f_B;

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

  drake::solvers::GurobiSolver solver;
  problem.get_mutable_prog()->SetSolverOption(solvers::GurobiSolver::id(),
                                              "OutputFlag", 1);
  const auto solution_result = solver.Solve(*(problem.get_mutable_prog()));
  EXPECT_EQ(solution_result, drake::solvers::SolutionResult::kSolutionFound);

  // Now check the solution.
  const auto p_WB_sol = problem.prog()->GetSolution(problem.p_WB()[0]);
  const auto R_WB_sol = problem.prog()->GetSolution(problem.R_WB()[0]);

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

  const auto f_BV_sol = problem.prog()->GetSolution(problem.f_BV()[0]);
  const auto f_WV_sol = problem.prog()->GetSolution(f_WV);
  for (int i = 0; i < 4; ++i) {
    VisualizeForce(&viewer, p_WV_sol.col(i), f_WV_sol.col(i),
                   block.mass() * kGravity * 5, "f_WV" + std::to_string(i));
  }

  // Make sure that static equilibrium is satisfied.
  const Eigen::Vector3d mg(0, 0, -block.mass() * kGravity);
  EXPECT_TRUE(CompareMatrices((R_WB_sol * f_BV_sol).rowwise().sum(), -mg, tol));
  const auto p_WC_sol = p_WB_sol + R_WB_sol * block.center_of_mass();
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
  // Actually this constraint is violated a lot, it is only satisfied to a high
  // accuracy, when the ground contact force is vertical, which is the optimal
  // solution. When the tangential contact force at each vertex is non-zero,
  // this constraint can be violated by about 0.5
  EXPECT_TRUE(CompareMatrices(f_WV_sol, R_WB_sol * f_BV_sol, 1e-3));
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
