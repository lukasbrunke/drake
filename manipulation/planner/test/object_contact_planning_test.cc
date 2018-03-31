#include "drake/manipulation/planner/object_contact_planning.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/manipulation/dev/remote_tree_viewer_wrapper.h"
#include "drake/manipulation/planner/friction_cone.h"
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

    // Add body contact point at the middle of each block edge, and the center
    // of each facet.
    Q_.reserve(18);

    auto AddFacetCenterContactPoint = [this](
        const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
        const Eigen::Ref<const Eigen::Vector3d>& n_B) {
      this->Q_.emplace_back(
          p_BQ, GenerateLinearizedFrictionConeEdges<6>(n_B, this->mu()));
    };
    // center of top facet
    AddFacetCenterContactPoint(Eigen::Vector3d(0, 0, height() / 2),
                               -Eigen::Vector3d::UnitZ());
    // center of bottom facet
    AddFacetCenterContactPoint(Eigen::Vector3d(0, 0, -height() / 2),
                               Eigen::Vector3d::UnitZ());
    // center of the positive x facet
    AddFacetCenterContactPoint(Eigen::Vector3d(width() / 2, 0, 0),
                               -Eigen::Vector3d::UnitX());
    // center of the negative x facet
    AddFacetCenterContactPoint(Eigen::Vector3d(-width() / 2, 0, 0),
                               Eigen::Vector3d::UnitX());
    // center of the positive y facet
    AddFacetCenterContactPoint(Eigen::Vector3d(0, width() / 2, 0),
                               -Eigen::Vector3d::UnitY());
    // center of the negative y facet
    AddFacetCenterContactPoint(Eigen::Vector3d(0, -width() / 2, 0),
                               Eigen::Vector3d::UnitY());

    auto AddEdgeMiddleContactPoint = [this](
        const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
        const Eigen::Ref<const Eigen::Vector3d>& n1_B,
        const Eigen::Ref<const Eigen::Vector3d>& n2_B) {
      const auto e1_B =
          GenerateLinearizedFrictionConeEdges<6>(n1_B, this->mu());
      const auto e2_B =
          GenerateLinearizedFrictionConeEdges<6>(n2_B, this->mu());
      // TODO(hongkai.dai): find the convex hull of (e1_B, e2_B)
      Eigen::Matrix<double, 3, 12> e_B;
      e_B.leftCols<6>() = e1_B;
      e_B.rightCols<6>() = e2_B;
      this->Q_.emplace_back(p_BQ, e_B);
    };

    // The order is
    // 0 + +
    // 0 + -
    // 0 - +
    // 0 - -
    // + 0 +
    // - 0 +
    // + 0 -
    // - 0 -
    // + + 0
    // + - 0
    // - + 0
    // - - 0
    for (int dim0 = 0; dim0 < 3; ++dim0) {
      const int dim1 = (dim0 + 1) % 3;
      const int dim2 = (dim1 + 1) % 3;
      for (double multiplier1 : {1, -1}) {
        for (double multiplier2 : {1, -1}) {
          Eigen::Vector3d p_BQ;
          p_BQ(dim0) = 0;
          p_BQ(dim1) = dimension()(dim1) / 2 * multiplier1;
          p_BQ(dim2) = dimension()(dim2) / 2 * multiplier2;
          Eigen::Vector3d n1_B = Eigen::Vector3d::Zero();
          n1_B(dim1) = -multiplier1;
          Eigen::Vector3d n2_B = Eigen::Vector3d::Zero();
          n2_B(dim2) = -multiplier2;
          AddEdgeMiddleContactPoint(p_BQ, n1_B, n2_B);
        }
      }
    }
  }

  const Eigen::Matrix<double, 3, 8>& p_BV() const { return p_BV_; }

  double width() const { return 0.1; }

  double height() const { return 0.15; }

  double mass() const { return 1; }

  Eigen::Vector3d dimension() const {
    return Eigen::Vector3d(width(), width(), height());
  }

  Vector3d center_of_mass() const { return Vector3d::Zero(); }

  std::vector<int> bottom_vertex_indices() const { return {1, 3, 5, 7}; }

  std::vector<int> top_vertex_indices() const { return {0, 2, 4, 6}; }

  std::vector<int> positive_x_vertex_indices() const { return {0, 1, 2, 3}; }

  std::vector<int> negative_x_vertex_indices() const { return {4, 5, 6, 7}; }

  std::vector<int> positive_y_vertex_indices() const { return {0, 1, 4, 5}; }

  std::vector<int> negative_y_vertex_indices() const { return {2, 3, 6, 7}; }

  const std::vector<BodyContactPoint>& Q() const { return Q_; }

  double mu() const { return 0.5; }

 private:
  Eigen::Matrix<double, 3, 8> p_BV_;

  std::vector<BodyContactPoint> Q_;
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

void AllVerticesAboveTable(const Block& block, ObjectContactPlanning* problem) {
  // Constrain that all vertices of the block is above the table.
  for (int i = 0; i < 8; ++i) {
    problem->get_mutable_prog()->AddLinearConstraint(
        (problem->p_WB()[0] + problem->R_WB()[0] * block.p_BV().col(i))(2) >=
        0);
  }
}

solvers::MatrixDecisionVariable<3, Eigen::Dynamic> SetTableContactVertices(
    const Block& block, const std::vector<int>& vertex_indices, double mu_table,
    int knot, ObjectContactPlanning* problem) {
  problem->SetContactVertexIndices(knot, vertex_indices,
                                   block.mass() * kGravity * 2);

  // Add the big-M constraint for the contact distance and the binary variable.
  // p_WV_z <= M * (1 - b) where b is the binary variable indicating contact.
  const double distance_big_M{0.1};
  for (int i = 0; i < 4; ++i) {
    const int vertex = vertex_indices[i];
    problem->get_mutable_prog()->AddLinearConstraint(
        (problem->p_WB()[knot] +
         problem->R_WB()[knot] * block.p_BV().col(vertex))(2) <=
        distance_big_M * (1 - problem->vertex_contact_flag()[knot](i)));
  }

  // Compute the vertex contact force in the world frame f_WV.
  const int num_vertices = static_cast<int>(vertex_indices.size());
  const auto f_WV =
      problem->get_mutable_prog()->NewContinuousVariables<3, Eigen::Dynamic>(
          3, num_vertices, "f_WV");
  std::array<Eigen::VectorXd, 3> phi_f;
  phi_f[0] =
      Eigen::Matrix<double, 5, 1>::LinSpaced(-1, 1) * block.mass() * kGravity;
  phi_f[1] =
      Eigen::Matrix<double, 5, 1>::LinSpaced(-1, 1) * block.mass() * kGravity;
  phi_f[2] =
      Eigen::Matrix<double, 3, 1>::LinSpaced(0, 1) * block.mass() * kGravity;
  const auto table_friction_edges = GenerateLinearizedFrictionConeEdges<8>(
      Eigen::Vector3d::UnitZ(), mu_table);
  for (int i = 0; i < num_vertices; ++i) {
    problem->CalcContactForceInWorldFrame(problem->f_BV()[knot].col(i),
                                          f_WV.col(i), knot, false, phi_f);
    // Add friction cone constraint on f_WV
    // AddFrictionConeConstraint(mu_table, Vector3d::UnitZ(),
    //                          f_WV.col(i).cast<Expression>(),
    //                          problem.get_mutable_prog());
    AddLinearizedFrictionConeConstraint(table_friction_edges, f_WV.col(i),
                                        problem->get_mutable_prog());
  }

  return f_WV;
}

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
