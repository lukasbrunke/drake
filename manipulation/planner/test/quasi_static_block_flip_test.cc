#include "drake/manipulation/planner/quasi_static_object_contact_planning.h"

#include <chrono>
#include <thread>
#include <unordered_set>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/manipulation/planner/test/block_test_util.h"
#include "drake/math/rotation_matrix.h"
#include "drake/solvers/gurobi_solver.h"

using drake::solvers::MatrixDecisionVariable;
using drake::solvers::VectorDecisionVariable;
using drake::symbolic::Expression;

namespace drake {
namespace manipulation {
namespace planner {
namespace {
class QuasiStaticBlockFlipTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuasiStaticBlockFlipTest)

  QuasiStaticBlockFlipTest()
      : block_(),
        num_pushers_(std::get<0>(GetParam())),
        nT_(std::get<1>(GetParam())),
        problem_(nT_, block_.mass(), block_.center_of_mass(), block_.p_BV(),
                 num_pushers_, block_.Q()),
        f_WV_(nT_) {
    SetUpBlockFlipTest(block_, num_pushers_, nT_, &problem_, &f_WV_);
    // Static equilibrium constraint.
    problem_.AddStaticEquilibriumConstraint();
  }

  ~QuasiStaticBlockFlipTest() = default;

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

  void TestOnePusher() {
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

  void TestTwoPushers() {
    problem_.get_mutable_prog()->SetSolverOption(solvers::GurobiSolver::id(),
                                                 "OutputFlag", 1);

    // Some pusher points should not be active simultaneously. For example, if
    // an edge point is active, then no other point should be.
    const auto edge_Q_indices = block_.edge_Q_indices();
    std::unordered_set<int> edge_Q_index_set(edge_Q_indices.begin(),
                                             edge_Q_indices.end());
    for (int knot = 0; knot < nT_; ++knot) {
      for (int i = 0;
           i < static_cast<int>(problem_.contact_Q_indices()[knot].size());
           ++i) {
        const bool is_Qi_on_edge =
            edge_Q_index_set.find(problem_.contact_Q_indices()[knot][i]) !=
            edge_Q_index_set.end();
        for (int j = i + 1;
             j < static_cast<int>(problem_.contact_Q_indices()[knot].size());
             ++j) {
          const bool is_Qj_on_edge =
              edge_Q_index_set.find(problem_.contact_Q_indices()[knot][j]) !=
              edge_Q_index_set.end();
          if (is_Qi_on_edge || is_Qj_on_edge) {
            problem_.get_mutable_prog()->AddLinearConstraint(
                Eigen::RowVector2d::Ones(), 0, 1,
                VectorDecisionVariable<2>(problem_.b_Q_contact()[knot](i),
                                          problem_.b_Q_contact()[knot](j)));
          }
        }
      }
    }
    solvers::GurobiSolver solver;
    const auto solution_result = solver.Solve(*(problem_.get_mutable_prog()));
    EXPECT_EQ(solution_result, solvers::SolutionResult::kSolutionFound);
    Solution sol;
    GetSolution(&sol);
    VisualizeResult(sol);
  }

 protected:
  Block block_;
  int num_pushers_;
  int nT_;
  QuasiStaticObjectContactPlanning problem_;
  std::vector<MatrixDecisionVariable<3, Eigen::Dynamic>> f_WV_;
};

TEST_P(QuasiStaticBlockFlipTest, Test) {
  if (num_pushers_ == 1) {
    TestOnePusher();
  } else if (num_pushers_ == 2) {
    TestTwoPushers();
  }
}

std::vector<std::tuple<int, int>> test_params() {
  std::vector<std::tuple<int, int>> params;
  params.push_back(std::make_tuple(1, 4));
  params.push_back(std::make_tuple(1, 5));
  params.push_back(std::make_tuple(2, 5));
  return params;
}

INSTANTIATE_TEST_CASE_P(ObjectContactPlanningTest, QuasiStaticBlockFlipTest,
                        ::testing::ValuesIn(test_params()));
}  // namespace
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
