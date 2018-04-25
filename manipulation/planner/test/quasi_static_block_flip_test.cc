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
    BlockFlipSolution sol;
    SetBlockFlipSolution(problem_, block_, f_WV_, true, &sol);
    VisualizeResult(problem_, block_, sol);
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
    BlockFlipSolution sol;
    SetBlockFlipSolution(problem_, block_, f_WV_, true, &sol);
    VisualizeResult(problem_, block_, sol);
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
