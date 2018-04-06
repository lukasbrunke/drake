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

class BlockFlipTest : public ::testing::TestWithParam<std::tuple<int, int>> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BlockFlipTest)

  BlockFlipTest()
      : block_(),
        num_pushers_(std::get<0>(GetParam())),
        nT_(std::get<1>(GetParam())),
        problem_(nT_, block_.mass(), block_.center_of_mass(), block_.p_BV(),
                 num_pushers_, block_.Q()),
        f_WV_(nT_) {
    AllVerticesAboveTable(block_, &problem_);

    const double mu_table = 1;

    const Eigen::Vector3d p_WB0(0, 0, block_.height() / 2);
    problem_.get_mutable_prog()->AddBoundingBoxConstraint(p_WB0, p_WB0,
                                                          problem_.p_WB()[0]);
    // Initially, the bottom vertices are on the table.
    f_WV_[0] = SetTableContactVertices(block_, block_.bottom_vertex_indices(),
                                       mu_table, 0, 0, &problem_);
    problem_.get_mutable_prog()->AddBoundingBoxConstraint(
        1, 1, problem_.vertex_contact_flag()[0]);

    // Finally, the positive x vertices are on the table.
    f_WV_[nT_ - 1] =
        SetTableContactVertices(block_, block_.positive_x_vertex_indices(),
                                mu_table, nT_ - 1, 0, &problem_);
    problem_.get_mutable_prog()->AddBoundingBoxConstraint(
        1, 1, problem_.vertex_contact_flag()[nT_ - 1]);

    // For all the points in between the first and last knots, the candidate
    // table contact vertices are bottom and positive x vertices.
    for (int knot = 1; knot < nT_ - 1; ++knot) {
      f_WV_[knot] = SetTableContactVertices(
          block_, block_.bottom_and_positive_x_vertex_indices(), mu_table, knot,
          block_.mass() * kGravity * 1.1, &problem_);

      // At least one vertex on the table, at most 4 vertices on the table
      problem_.get_mutable_prog()->AddLinearConstraint(
          Eigen::RowVectorXd::Ones(
              block_.bottom_and_positive_x_vertex_indices().size()),
          1, 4, problem_.vertex_contact_flag()[knot]);
    }

    // Choose all body contact points except those on the bottom or the positive
    // x facet.
    std::vector<int> pusher_contact_point_indices;
    for (int i = 0; i < static_cast<int>(block_.Q().size()); ++i) {
      if (block_.Q()[i].p_BQ()(0) <= 1E-10 &&
          block_.Q()[i].p_BQ()(2) >= -1E-10) {
        pusher_contact_point_indices.push_back(i);
      }
    }
    for (int knot = 0; knot < nT_; ++knot) {
      problem_.SetPusherContactPointIndices(knot, pusher_contact_point_indices,
                                            block_.mass() * kGravity);
    }
    // Pusher remain in static contact, no sliding is allowed.
    for (int interval = 0; interval < nT_ - 1; ++interval) {
      problem_.AddPusherStaticContactConstraint(interval);
    }

    // Add non-sliding contact constraint on the vertex.
    std::vector<solvers::VectorXDecisionVariable> b_non_sliding(nT_);
    auto AddVertexNonSlidingConstraintAtKnot = [this, &b_non_sliding](
        int knot, const std::vector<int>& common_vertex_indices,
        double distance_big_M) {
      const int num_vertex_knot = common_vertex_indices.size();
      b_non_sliding[knot].resize(num_vertex_knot);
      for (int i = 0; i < num_vertex_knot; ++i) {
        b_non_sliding[knot](i) =
            (this->problem_.AddVertexNonSlidingConstraint(
                 knot, common_vertex_indices[i], Eigen::Vector3d::UnitX(),
                 Eigen::Vector3d::UnitY(), distance_big_M))
                .value();
      }
    };

    AddVertexNonSlidingConstraintAtKnot(0, block_.bottom_vertex_indices(), 0.1);
    AddVertexNonSlidingConstraintAtKnot(
        nT_ - 2, block_.positive_x_vertex_indices(), 0.1);
    for (int interval = 1; interval < nT_ - 2; ++interval) {
      AddVertexNonSlidingConstraintAtKnot(
          interval, block_.bottom_and_positive_x_vertex_indices(), 0.1);
    }

    // Static equilibrium constraint.
    problem_.AddStaticEquilibriumConstraint();

    // Bound the maximal angle difference in each interval.
    const double max_angle_difference = M_PI / 4;
    for (int interval = 0; interval < nT_ - 1; ++interval) {
      problem_.AddOrientationDifferenceUpperBoundLinearApproximation(
          interval, max_angle_difference);
      problem_.AddOrientationDifferenceUpperBoundBilinearApproximation(
          interval, max_angle_difference);
    }
    //// The block moves less than 10cms in each direction within an interval.
    // for (int interval = 0; interval < nT - 1; ++interval) {
    //  const Vector3<Expression> delta_p_WB =
    //      problem_.p_WB()[interval + 1] - problem_.p_WB()[interval];
    //  problem_.get_mutable_prog()->AddLinearConstraint(delta_p_WB(0), -0.1,
    //  0.1);
    //  problem_.get_mutable_prog()->AddLinearConstraint(delta_p_WB(1), -0.1,
    //  0.1);
    //  problem_.get_mutable_prog()->AddLinearConstraint(delta_p_WB(2), -0.1,
    //  0.1);
    //}
  }

  ~BlockFlipTest() = default;

  struct Solution {
    std::vector<Eigen::Matrix3d> R_WB_sol;
    std::vector<Eigen::Vector3d> p_WB_sol;
    std::vector<Eigen::Matrix3Xd> p_WV_sol;
    std::vector<Eigen::Matrix3Xd> p_WQ_sol;
    std::vector<Eigen::Matrix3Xd> f_BV_sol;
    std::vector<Eigen::Matrix3Xd> f_BQ_sol;
    std::vector<Eigen::Matrix3Xd> f_WV_sol;
    std::vector<Eigen::Matrix3Xd> f_WQ_sol;
    std::vector<Eigen::VectorXd> b_V_contact_sol;
    std::vector<Eigen::VectorXd> b_Q_contact_sol;
  };

  void GetSolution(Solution* sol) const {
    sol->R_WB_sol.reserve(nT_);
    sol->p_WB_sol.reserve(nT_);
    sol->p_WV_sol.reserve(nT_);
    sol->p_WQ_sol.reserve(nT_);
    sol->f_BV_sol.reserve(nT_);
    sol->f_BQ_sol.reserve(nT_);
    sol->f_WV_sol.reserve(nT_);
    sol->f_WQ_sol.reserve(nT_);
    sol->b_V_contact_sol.reserve(nT_);
    sol->b_Q_contact_sol.reserve(nT_);
    for (int knot = 0; knot < nT_; ++knot) {
      sol->p_WB_sol.push_back(
          problem_.prog().GetSolution(problem_.p_WB()[knot]));
      sol->R_WB_sol.push_back(
          problem_.prog().GetSolution(problem_.R_WB()[knot]));
      sol->f_BV_sol.push_back(
          problem_.prog().GetSolution(problem_.f_BV()[knot]));
      sol->f_WV_sol.push_back(problem_.prog().GetSolution(f_WV_[knot]));
      const int num_vertices_knot =
          problem_.contact_vertex_indices()[knot].size();
      sol->p_WV_sol.emplace_back(3, num_vertices_knot);
      for (int i = 0; i < num_vertices_knot; ++i) {
        sol->p_WV_sol[knot].col(i) =
            sol->p_WB_sol[knot] +
            sol->R_WB_sol[knot] *
                block_.p_BV().col(problem_.contact_vertex_indices()[knot][i]);
      }
      const int num_Q_points = problem_.contact_Q_indices()[knot].size();
      sol->p_WQ_sol.emplace_back(3, num_Q_points);
      for (int i = 0; i < num_Q_points; ++i) {
        const int Q_index = problem_.contact_Q_indices()[knot][i];
        sol->p_WQ_sol[knot].col(i) =
            sol->p_WB_sol[knot] +
            sol->R_WB_sol[knot] * block_.Q()[Q_index].p_BQ();
      }
      sol->b_V_contact_sol.push_back(
          problem_.prog().GetSolution(problem_.vertex_contact_flag()[knot]));
      sol->f_BQ_sol.push_back(
          problem_.prog().GetSolution(problem_.f_BQ()[knot]));
      sol->f_WQ_sol.push_back(sol->R_WB_sol[knot] * sol->f_BQ_sol[knot]);
      sol->b_Q_contact_sol.push_back(
          problem_.prog().GetSolution(problem_.b_Q_contact()[knot]));
      std::cout << "knot " << knot << std::endl;
      std::cout << "p_WB[" << knot << "]\n " << sol->p_WB_sol[knot].transpose()
                << std::endl;
      std::cout << "R_WB[" << knot << "]\n " << sol->R_WB_sol[knot]
                << std::endl;
      std::cout << "b_V_contact[" << knot << "]\n "
                << sol->b_V_contact_sol[knot].transpose() << std::endl;
      std::cout << "f_BV_sol[" << knot << "]\n"
                << sol->f_BV_sol[knot] << std::endl;
      std::cout << "p_WV_sol[" << knot << "]\n"
                << sol->p_WV_sol[knot] << std::endl;

      std::cout << "b_Q_contact[" << knot << "]\n "
                << sol->b_Q_contact_sol[knot].transpose() << std::endl;
      std::cout << "f_BQ_sol[" << knot << "]\n"
                << sol->f_BQ_sol[knot] << std::endl;
    }
  }

  void VisualizeResult(const Solution& sol) const {
    // Now visualize the result.
    dev::RemoteTreeViewerWrapper viewer;

    const Eigen::Vector4d color_red(1, 0, 0, 0.9);
    const Eigen::Vector4d color_green(0, 1, 0, 0.9);

    // VisualizeTable(&viewer);

    const double viewer_force_normalizer = block_.mass() * kGravity * 5;
    for (int knot = 0; knot < nT_; ++knot) {
      VisualizeBlock(&viewer, sol.R_WB_sol[knot], sol.p_WB_sol[knot], block_);
      // Visualize vertex contact force.
      for (int i = 0;
           i < static_cast<int>(problem_.contact_vertex_indices()[knot].size());
           ++i) {
        VisualizeForce(&viewer, sol.p_WV_sol[knot].col(i),
                       sol.R_WB_sol[knot] * sol.f_BV_sol[knot].col(i),
                       viewer_force_normalizer, "f_WV" + std::to_string(i),
                       color_red);
      }
      // Visualize pusher contact force.
      for (int i = 0;
           i < static_cast<int>(problem_.contact_Q_indices()[knot].size());
           ++i) {
        VisualizeForce(&viewer, sol.p_WQ_sol[knot].col(i),
                       sol.R_WB_sol[knot] * sol.f_BQ_sol[knot].col(i),
                       viewer_force_normalizer, "f_WQ" + std::to_string(i),
                       color_green);
      }
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  }

 protected:
  Block block_;
  int num_pushers_;
  int nT_;
  ObjectContactPlanning problem_;
  std::vector<MatrixDecisionVariable<3, Eigen::Dynamic>> f_WV_;
};

TEST_P(BlockFlipTest, TestOnePusher) {
  if (num_pushers_ == 1) {
    problem_.get_mutable_prog()->SetSolverOption(solvers::GurobiSolver::id(),
                                                 "OutputFlag", 1);

    // Always one pusher in contact
    for (int knot = 0; knot < nT_; ++knot) {
      problem_.get_mutable_prog()->AddLinearConstraint(
          problem_.b_Q_contact()[knot].cast<Expression>().sum() == 1);
    }

    solvers::GurobiSolver solver;
    const auto solution_result = solver.Solve(*(problem_.get_mutable_prog()));
    EXPECT_EQ(solution_result, solvers::SolutionResult::kSolutionFound);
    Solution sol;
    GetSolution(&sol);
    VisualizeResult(sol);
  }
}

TEST_P(BlockFlipTest, TestTwoPushers) {
  if (num_pushers_ == 2) {
    problem_.get_mutable_prog()->SetSolverOption(solvers::GurobiSolver::id(),
                                                 "OutputFlag", 1);

    // Some pusher points should not be active simultaneously. For example, if a 
    solvers::GurobiSolver solver;
    const auto solution_result = solver.Solve(*(problem_.get_mutable_prog()));
    EXPECT_EQ(solution_result, solvers::SolutionResult::kSolutionFound);
    Solution sol;
    GetSolution(&sol);
    VisualizeResult(sol);
  }
}

std::vector<std::tuple<int, int>> test_params() {
  std::vector<std::tuple<int, int>> params;
  params.push_back(std::make_tuple(1, 4));
  //params.push_back(std::make_tuple(1, 5));
  return params;
}

INSTANTIATE_TEST_CASE_P(ObjectContactPlanningTest, BlockFlipTest,
                        ::testing::ValuesIn(test_params()));
}  // namespace
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
