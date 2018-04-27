#include "drake/manipulation/planner/quasi_dynamic_object_contact_planning.h"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/manipulation/planner/test/block_test_util.h"
#include "drake/solvers/gurobi_solver.h"

using drake::solvers::MatrixDecisionVariable;
using drake::solvers::VectorDecisionVariable;
using drake::symbolic::Expression;

namespace drake {
namespace manipulation {
namespace planner {
namespace {
struct QuasiDynamicBlockFlipSolution : BlockFlipSolution {
  Eigen::Matrix3Xd omega_B_sol;
  Eigen::Matrix3Xd v_B_sol;
};

void SetQuasiDynamicBlockFlipSolution(
    const QuasiDynamicObjectContactPlanning& problem, const Block& block,
    const std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>>& f_WV,
    bool print_flag, QuasiDynamicBlockFlipSolution* sol) {
  SetBlockFlipSolution(problem, block, f_WV, print_flag, sol);
  sol->omega_B_sol = problem.prog().GetSolution(problem.omega_B());
  sol->v_B_sol = problem.prog().GetSolution(problem.v_B());
  if (print_flag) {
    std::cout << "omega_B:\n" << sol->omega_B_sol << "\n";
    std::cout << "v_B:\n" << sol->v_B_sol << "\n";
  }
}

class QuasiDynamicBlockFlipTest
    : public ::testing::TestWithParam<std::tuple<int, int, double>> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuasiDynamicBlockFlipTest)

  QuasiDynamicBlockFlipTest()
      : block_(),
        num_pushers_(std::get<0>(GetParam())),
        nT_(std::get<1>(GetParam())),
        dt_(std::get<2>(GetParam())),
        problem_(nT_, dt_, block_.mass(), block_.I_B(), block_.center_of_mass(),
                 block_.p_BV(), num_pushers_, block_.Q(), 0.5, M_PI, false),
        f_WV_(nT_) {
    SetUpBlockFlipTest(block_, num_pushers_, nT_, &problem_, &f_WV_);
    problem_.AddQuasiDynamicConstraint();
  }

  ~QuasiDynamicBlockFlipTest() = default;

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
    QuasiDynamicBlockFlipSolution sol;
    SetQuasiDynamicBlockFlipSolution(problem_, block_, f_WV_, true, &sol);
    VisualizeResult(problem_, block_, sol);
  }

  void TestTwoPushers() {
  }

 protected:
  Block block_;
  int num_pushers_;
  int nT_;
  double dt_;
  QuasiDynamicObjectContactPlanning problem_;
  std::vector<MatrixDecisionVariable<3, Eigen::Dynamic>> f_WV_;
};

TEST_P(QuasiDynamicBlockFlipTest, Test) {
  if (num_pushers_ == 1) {
    TestOnePusher();
  } else if (num_pushers_ == 2) {
    TestTwoPushers();
  }
}

std::vector<std::tuple<int, int, double>> test_params() {
  std::vector<std::tuple<int, int, double>> params;
  params.push_back(std::make_tuple(1, 4, 0.1));
  params.push_back(std::make_tuple(1, 5, 0.1));
  params.push_back(std::make_tuple(2, 5, 0.1));
  return params;
}

INSTANTIATE_TEST_CASE_P(ObjectContactPlanningTest, QuasiDynamicBlockFlipTest,
                        ::testing::ValuesIn(test_params()));
}  // namespace
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
