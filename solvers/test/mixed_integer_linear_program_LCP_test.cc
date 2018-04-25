#include "drake/solvers/mixed_integer_linear_program_LCP.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/gurobi_solver.h"

namespace drake {
namespace solvers {
namespace {
GTEST_TEST(TestMILPLCP, Cottle) {
  Eigen::Matrix3d M;
  // clang-format off
  M <<  0, -1, 2,
        2, 0, -2,
        -1, 1, 0;
  // clang-format on
  const Eigen::Vector3d q(-3, 6, -1);

  const Eigen::Vector3d z_max(10, 10, 10);
  const Eigen::Vector3d w_max(10, 10, 10);

  MixedIntegerLinearProgramLCP milp_lcp(q, M, z_max, w_max);

  GurobiSolver solver;
  const auto result = solver.Solve(*(milp_lcp.get_mutable_prog()));
  EXPECT_EQ(result, SolutionResult::kSolutionFound);
  EXPECT_TRUE(CompareMatrices(milp_lcp.prog().GetSolution(milp_lcp.z()),
                              Eigen::Vector3d(0, 1, 3), 1E-8));
  EXPECT_TRUE(CompareMatrices(milp_lcp.prog().GetSolution(milp_lcp.w()),
                              Eigen::Vector3d(2, 0, 0), 1E-8));
  EXPECT_TRUE(CompareMatrices(milp_lcp.prog().GetSolution(milp_lcp.b()),
                              Eigen::Vector3d(1, 0, 0), 1E-8));
}

GTEST_TEST(TestMILPLCP, Cycling) {
  Eigen::Matrix3d M;
  // clang-format off
  M <<  1, 2, 0,
        0, 1, 2,
        2, 0, 1;
  // clang-format on
  const Eigen::Vector3d q(-1, -1, -1);

  const Eigen::Vector3d z_max(10, 10, 10);
  const Eigen::Vector3d w_max(10, 10, 10);

  MixedIntegerLinearProgramLCP milp_lcp(q, M, z_max, w_max);

  GurobiSolver solver;
  const auto result = solver.Solve(*(milp_lcp.get_mutable_prog()));
  EXPECT_EQ(result, SolutionResult::kSolutionFound);
  EXPECT_TRUE(CompareMatrices(milp_lcp.prog().GetSolution(milp_lcp.z()),
                              Eigen::Vector3d(1.0 / 3, 1.0 / 3, 1.0 / 3),
                              4E-7));
  EXPECT_TRUE(CompareMatrices(milp_lcp.prog().GetSolution(milp_lcp.w()),
                              Eigen::Vector3d(0, 0, 0), 1E-8));
  EXPECT_TRUE(CompareMatrices(milp_lcp.prog().GetSolution(milp_lcp.b()),
                              Eigen::Vector3d(0, 0, 0), 1E-8));
}
}  // namespace
}  // namespace solvers
}  // namespace drake
