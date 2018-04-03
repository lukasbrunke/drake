#include "drake/manipulation/planner/object_contact_planning.h"

#include <chrono>
#include <thread>
#include <unordered_set>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/manipulation/planner/test/block_test_util.h"
#include "drake/math/rotation_matrix.h"
#include "drake/solvers/gurobi_solver.h"

using drake::solvers::MatrixDecisionVariable;
using drake::symbolic::Expression;

namespace drake {
namespace manipulation {
namespace planner {
namespace {
GTEST_TEST(ObjectContactPlanning, TestOrientationDifference) {
  // To verify the math
  // | R₁ - R₂ |² = (2√2 sin(α/2))²
  // where α is the angle between the rotation matrix R₁ and R₂.
  Eigen::AngleAxisd R(0.1, Eigen::Vector3d(0.1, 0.2, 0.3).normalized());
  EXPECT_NEAR(
      ((R.toRotationMatrix() - Eigen::Matrix3d::Identity()) *
       ((R.toRotationMatrix() - Eigen::Matrix3d::Identity()).transpose()))
          .trace(),
      std::pow(2 * std::sqrt(2) * std::sin(0.1 / 2), 2), 1E-10);
}

GTEST_TEST(ObjectContactPlanningTest, TestOnePusher) {
  Block block;
  const int nT = 4;
  const int num_pushers = 2;

  ObjectContactPlanning problem(nT, block.mass(), block.center_of_mass(),
                                block.p_BV(), num_pushers, block.Q());

  AllVerticesAboveTable(block, &problem);

  const double mu_table = 1;

  std::vector<MatrixDecisionVariable<3, Eigen::Dynamic>> f_WV(nT);
  const Eigen::Vector3d p_WB0(0, 0, block.height() / 2);
  problem.get_mutable_prog()->AddBoundingBoxConstraint(p_WB0, p_WB0,
                                                       problem.p_WB()[0]);
  // Initially, the bottom vertices are on the table.
  f_WV[0] = SetTableContactVertices(block, block.bottom_vertex_indices(),
                                    mu_table, 0, 0, &problem);
  problem.get_mutable_prog()->AddBoundingBoxConstraint(
      1, 1, problem.vertex_contact_flag()[0]);

  // Finally, the positive x vertices are on the table.
  f_WV[nT - 1] = SetTableContactVertices(
      block, block.positive_x_vertex_indices(), mu_table, nT - 1, 0, &problem);
  problem.get_mutable_prog()->AddBoundingBoxConstraint(
      1, 1, problem.vertex_contact_flag()[nT - 1]);

  // For all the points in between the first and last knots, the candidate
  // table contact vertices are bottom and positive x vertices.
  for (int knot = 1; knot < nT - 1; ++knot) {
    f_WV[knot] = SetTableContactVertices(
        block, block.bottom_and_positive_x_vertex_indices(), mu_table, knot,
        block.mass() * kGravity * 1.1, &problem);

    // At least one vertex on the table, at most 4 vertices on the table
    problem.get_mutable_prog()->AddLinearConstraint(
        Eigen::RowVectorXd::Ones(
            block.bottom_and_positive_x_vertex_indices().size()),
        1, 4, problem.vertex_contact_flag()[knot]);
  }

  // Choose all body contact points except those on the bottom or the positive
  // x facet.
  std::vector<int> pusher_contact_point_indices;
  for (int i = 0; i < static_cast<int>(block.Q().size()); ++i) {
    if (block.Q()[i].p_BQ()(0) <= 1E-10 && block.Q()[i].p_BQ()(2) >= -1E-10) {
      pusher_contact_point_indices.push_back(i);
    }
  }
  for (int knot = 0; knot < nT; ++knot) {
    problem.SetPusherContactPointIndices(knot, pusher_contact_point_indices,
                                         block.mass() * kGravity);
  }
  // Pusher remain in static contact, no sliding is allowed.
  for (int interval = 0; interval < nT - 1; ++interval) {
    problem.AddPusherStaticContactConstraint(interval);
  }

  // Add non-sliding contact constraint on the vertex.
  std::vector<solvers::VectorXDecisionVariable> b_non_sliding(nT);
  auto AddVertexNonSlidingConstraintAtKnot = [&problem, &b_non_sliding](
      int knot, const std::vector<int>& common_vertex_indices,
      double distance_big_M) {
    const int num_vertex_knot = common_vertex_indices.size();
    b_non_sliding[knot].resize(num_vertex_knot);
    for (int i = 0; i < num_vertex_knot; ++i) {
      b_non_sliding[knot](i) =
          (problem.AddVertexNonSlidingConstraint(
               knot, common_vertex_indices[i], Eigen::Vector3d::UnitX(),
               Eigen::Vector3d::UnitY(), distance_big_M))
              .value();
    }
  };

  AddVertexNonSlidingConstraintAtKnot(0, block.bottom_vertex_indices(), 0.1);
  AddVertexNonSlidingConstraintAtKnot(nT - 2, block.positive_x_vertex_indices(),
                                      0.1);
  for (int interval = 1; interval < nT - 2; ++interval) {
    AddVertexNonSlidingConstraintAtKnot(
        interval, block.bottom_and_positive_x_vertex_indices(), 0.1);
  }

  // Static equilibrium constraint.
  problem.AddStaticEquilibriumConstraint();

  // Bound the maximal angle difference in each interval.
  const double max_angle_difference = M_PI / 4;
  for (int interval = 0; interval < nT - 1; ++interval) {
    problem.AddOrientationDifferenceUpperBoundLinearApproximation(interval, max_angle_difference);
  }
  //// The block moves less than 10cms in each direction within an interval.
  // for (int interval = 0; interval < nT - 1; ++interval) {
  //  const Vector3<Expression> delta_p_WB =
  //      problem.p_WB()[interval + 1] - problem.p_WB()[interval];
  //  problem.get_mutable_prog()->AddLinearConstraint(delta_p_WB(0), -0.1, 0.1);
  //  problem.get_mutable_prog()->AddLinearConstraint(delta_p_WB(1), -0.1, 0.1);
  //  problem.get_mutable_prog()->AddLinearConstraint(delta_p_WB(2), -0.1, 0.1);
  //}

  problem.get_mutable_prog()->SetSolverOption(solvers::GurobiSolver::id(),
                                              "OutputFlag", 1);
  solvers::GurobiSolver solver;
  const auto solution_result = solver.Solve(*(problem.get_mutable_prog()));
  EXPECT_EQ(solution_result, solvers::SolutionResult::kSolutionFound);

  std::vector<Eigen::Vector3d> p_WB_sol(nT);
  std::vector<Eigen::Matrix3d> R_WB_sol(nT);
  std::vector<Eigen::Matrix3Xd> f_BV_sol(nT);
  std::vector<Eigen::Matrix3Xd> f_WV_sol(nT);
  std::vector<Eigen::Matrix3Xd> p_WV_sol(nT);
  std::vector<Eigen::VectorXd> b_V_contact_sol(nT);
  std::vector<Eigen::Matrix3Xd> f_BQ_sol(nT);
  std::vector<Eigen::Matrix3Xd> f_WQ_sol(nT);
  std::vector<Eigen::VectorXd> b_Q_contact_sol(nT);
  std::vector<Eigen::VectorXd> b_non_sliding_sol(nT);
  for (int knot = 0; knot < nT; ++knot) {
    p_WB_sol[knot] = problem.prog().GetSolution(problem.p_WB()[knot]);
    R_WB_sol[knot] = problem.prog().GetSolution(problem.R_WB()[knot]);
    f_BV_sol[knot] = problem.prog().GetSolution(problem.f_BV()[knot]);
    f_WV_sol[knot] = problem.prog().GetSolution(f_WV[knot]);
    const int num_vertices_knot = problem.contact_vertex_indices()[knot].size();
    p_WV_sol[knot].resize(3, num_vertices_knot);
    for (int i = 0; i < num_vertices_knot; ++i) {
      p_WV_sol[knot].col(i) =
          p_WB_sol[knot] +
          R_WB_sol[knot] *
              block.p_BV().col(problem.contact_vertex_indices()[knot][i]);
    }
    b_V_contact_sol[knot] =
        problem.prog().GetSolution(problem.vertex_contact_flag()[knot]);
    f_BQ_sol[knot] = problem.prog().GetSolution(problem.f_BQ()[knot]);
    f_WQ_sol[knot] = R_WB_sol[knot] * f_BQ_sol[knot];
    b_Q_contact_sol[knot] =
        problem.prog().GetSolution(problem.b_Q_contact()[knot]);
    b_non_sliding_sol[knot] = problem.prog().GetSolution(b_non_sliding[knot]);
    std::cout << "knot " << knot << std::endl;
    std::cout << "p_WB[" << knot << "]\n " << p_WB_sol[knot].transpose()
              << std::endl;
    std::cout << "R_WB[" << knot << "]\n " << R_WB_sol[knot] << std::endl;
    std::cout << "b_V_contact[" << knot << "]\n "
              << b_V_contact_sol[knot].transpose() << std::endl;
    std::cout << "f_BV_sol[" << knot << "]\n" << f_BV_sol[knot] << std::endl;
    std::cout << "p_WV_sol[" << knot << "]\n" << p_WV_sol[knot] << std::endl;

    std::cout << "b_Q_contact[" << knot << "]\n "
              << b_Q_contact_sol[knot].transpose() << std::endl;
    std::cout << "f_BQ_sol[" << knot << "]\n" << f_BQ_sol[knot] << std::endl;
    std::cout << "b_non_sliding_sol[" << knot << "]\n"
              << b_non_sliding_sol[knot].transpose() << std::endl;
  }

  // Now visualize the result.
  dev::RemoteTreeViewerWrapper viewer;
  const double viewer_force_normalizer = block.mass() * kGravity * 5;
  for (int knot = 0; knot < nT; ++knot) {
    VisualizeBlock(&viewer, R_WB_sol[knot], p_WB_sol[knot], block);
    // Visualize vertex contact force.
    for (int i = 0;
         i < static_cast<int>(problem.contact_vertex_indices()[knot].size());
         ++i) {
      VisualizeForce(&viewer, p_WV_sol[knot].col(i),
                     R_WB_sol[knot] * f_BV_sol[knot].col(i),
                     viewer_force_normalizer, "f_WV" + std::to_string(i));
    }
    // Visualize pusher contact force.
    for (int i = 0;
         i < static_cast<int>(problem.contact_Q_indices()[knot].size()); ++i) {
      const int Q_index = problem.contact_Q_indices()[knot][i];
      const Eigen::Vector3d p_WQ_sol =
          p_WB_sol[knot] + R_WB_sol[knot] * block.Q()[Q_index].p_BQ();
      VisualizeForce(&viewer, p_WQ_sol, R_WB_sol[knot] * f_BQ_sol[knot].col(i),
                     viewer_force_normalizer, "f_WQ" + std::to_string(i));
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }
}
}  // namespace
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
