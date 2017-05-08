#include "drake/solvers/gurobi_solver.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_matrix_compare.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/test/linear_program_examples.h"
#include "drake/solvers/test/mathematical_program_test_util.h"
#include "drake/solvers/test/quadratic_program_examples.h"

namespace drake {
namespace solvers {
namespace test {
/*
TEST_P(LinearProgramTest, TestLP) {
  GurobiSolver solver;
  prob()->RunProblem(&solver);
}

INSTANTIATE_TEST_CASE_P(
    GurobiTest, LinearProgramTest,
    ::testing::Combine(::testing::ValuesIn(linear_cost_form()),
                       ::testing::ValuesIn(linear_constraint_form()),
                       ::testing::ValuesIn(linear_problems())));

TEST_F(InfeasibleLinearProgramTest0, TestGurobiInfeasible) {
  GurobiSolver solver;
  if (solver.available()) {
    // With dual reductions, gurobi may not be able to differentiate between
    // infeasible and unbounded.
    prog_->SetSolverOption(SolverType::kGurobi, "DualReductions", 1);
    SolutionResult result = solver.Solve(*prog_);
    EXPECT_EQ(result, SolutionResult::kInfeasible_Or_Unbounded);
    EXPECT_TRUE(std::isnan(prog_->GetOptimalCost()));
    prog_->SetSolverOption(SolverType::kGurobi, "DualReductions", 0);
    result = solver.Solve(*prog_);
    EXPECT_EQ(result, SolutionResult::kInfeasibleConstraints);
    EXPECT_TRUE(std::isnan(prog_->GetOptimalCost()));
  }
}

TEST_F(UnboundedLinearProgramTest0, TestGurobiUnbounded) {
  GurobiSolver solver;
  if (solver.available()) {
    // With dual reductions, gurobi may not be able to differentiate between
    // infeasible and unbounded.
    prog_->SetSolverOption(SolverType::kGurobi, "DualReductions", 1);
    SolutionResult result = solver.Solve(*prog_);
    EXPECT_EQ(result, SolutionResult::kInfeasible_Or_Unbounded);
    prog_->SetSolverOption(SolverType::kGurobi, "DualReductions", 0);
    result = solver.Solve(*prog_);
    EXPECT_EQ(result, SolutionResult::kUnbounded);
  }
}

TEST_P(QuadraticProgramTest, TestQP) {
  GurobiSolver solver;
  prob()->RunProblem(&solver);
}

INSTANTIATE_TEST_CASE_P(
    GurobiTest, QuadraticProgramTest,
    ::testing::Combine(::testing::ValuesIn(quadratic_cost_form()),
                       ::testing::ValuesIn(linear_constraint_form()),
                       ::testing::ValuesIn(quadratic_problems())));

GTEST_TEST(QPtest, TestUnitBallExample) {
  GurobiSolver solver;
  if (solver.available()) {
    TestQPonUnitBallExample(solver);
  }
}

GTEST_TEST(GurobiTest, TestInitialGuess) {
  GurobiSolver solver;
  if (solver.available()) {
    // Formulate a simple problem with multiple optimal
    // solutions, and solve it twice with two different
    // initial conditions. The resulting solutions should
    // match the initial conditions supplied. Doing two
    // solves from different initial positions ensures the
    // test doesn't pass by chance.
    MathematicalProgram prog;
    auto x = prog.NewBinaryVariables<1>("x");
    // Presolve and Heuristics would each independently solve
    // this problem inside of the Gurobi solver, but without
    // consulting the initial guess.
    prog.SetSolverOption(SolverType::kGurobi, "Presolve", 0);
    prog.SetSolverOption(SolverType::kGurobi, "Heuristics", 0.0);

    double x_expected0_to_test[] = {0.0, 1.0};
    for (int i = 0; i < 2; i++) {
      Eigen::VectorXd x_expected(1);
      x_expected[0] = x_expected0_to_test[i];
      prog.SetInitialGuess(x, x_expected);
      SolutionResult result = solver.Solve(prog);
      EXPECT_EQ(result, SolutionResult::kSolutionFound);
      const auto& x_value = prog.GetSolution(x);
      EXPECT_TRUE(CompareMatrices(x_value, x_expected, 1E-6,
                                  MatrixCompareType::absolute));
      ExpectSolutionCostAccurate(prog, 1E-6);
    }
  }
}*/

// Compute the closest distance between two 2D triangles.
double ComputeClosestDistanceBetweenTriangles(const Eigen::Matrix<double, 2, 3>& triangle1, const Eigen::Matrix<double, 2, 3>& triangle2) {
  // Formulate an optimization program
  // min (p1 - p2)ᵀ * (p1 - p2)
  // s.t p1 = V1 * lambda1
  //     p2 = V2 * lambda2
  //     sum lambda1 = 1
  //     sum lambda2 = 1
  //     lambda1 >= 0, lambda2 >= 0
  // where the columns of V1, V2 are the vertices in each triangle.
  MathematicalProgram prog;
  auto lambda1 = prog.NewContinuousVariables<3>();
  auto lambda2 = prog.NewContinuousVariables<3>();
  prog.AddLinearConstraint(lambda1.cast<symbolic::Expression>().sum() == 1);
  prog.AddLinearConstraint(lambda2.cast<symbolic::Expression>().sum() == 1);
  prog.AddBoundingBoxConstraint(0, 1, lambda1);
  prog.AddBoundingBoxConstraint(0, 1, lambda2);
  Vector2<symbolic::Expression> p1 = triangle1 * lambda1;
  Vector2<symbolic::Expression> p2 = triangle2 * lambda2;
  prog.AddCost((p1 - p2).dot(p1 - p2));
  GurobiSolver gurobi_solver;
  if (gurobi_solver.available()) {
    auto sol_result = gurobi_solver.Solve(prog);
    EXPECT_EQ(sol_result, SolutionResult::kSolutionFound);
    return prog.GetOptimalCost();
  }
  return 0;
}

// Test a mixed integer optimization problem. Retrieve the multiple
// solutions.
GTEST_TEST(GurobiTest, MIPtest1) {
  // With N polytopes in 2D, each polytope is represented by its vertices
  // `v`, we want to find the nearest distance between two points p1 and p2,
  // such that p1 and p2 are in two different polytopes.
  // This problem can be formulated as
  // min (p1 - p2)ᵀ * (p1 - p2)
  // p1 = sum_i (α_i1 * v_i1 + ... + α_in * v_in)  (1)
  // p2 = sum_j (β_i1 * v_i1 + ... + β_in * v_in)  (2)
  // α_i1 + ... + α_in = z1_i ∀ i                  (3)
  // β_i1 + ... + β_in = z2_i ∀ i                  (4)
  // α_ij >=0, β_ij >=0  ∀ i, j                    (5)
  // z1_i ∈ {0, 1}, z2_i ∈ {0, 1}                  (6)
  // sum_i z1_i = 1                                (7)
  // sum_i z2_i = 1                                (8)
  // z1_i + z2_i <= 1                              (9)
  // Equation (1) (3) (5) (6) (7) means that p1 is in one of the polytope.
  // Same for p2. Equation 9 means p1 and p2 are in different polytopes.

  // V[i] contains the vertices of the i'th polytope in 2D.
  std::vector<Eigen::Matrix<double, 2, 3>> V;
  Eigen::Matrix<double, 2, 3> Vi;
  Vi << 0, 1, 1,
        0, 0, 1;
  V.push_back(Vi);

  Vi << 2, 3, 5,
        0, 1, 4;
  V.push_back(Vi);

  Vi << 0, 1, -1,
        4, 3, 4;
  V.push_back(Vi);

  MathematicalProgram prog;
  std::vector<VectorDecisionVariable<3>> alpha;
  std::vector<VectorDecisionVariable<3>> beta;

  auto z1 = prog.NewBinaryVariables<3>();
  auto z2 = prog.NewBinaryVariables<3>();
  prog.AddLinearConstraint(z1.cast<symbolic::Expression>().sum() == 1);
  prog.AddLinearConstraint(z2.cast<symbolic::Expression>().sum() == 1);
  Vector2<symbolic::Expression> p1;
  Vector2<symbolic::Expression> p2;
  p1 << 0, 0;
  p2 << 0, 0;
  for (int i = 0; i < 3; ++i) {
    alpha.push_back(prog.NewContinuousVariables<3>());
    beta.push_back(prog.NewContinuousVariables<3>());
    prog.AddLinearConstraint(alpha[i].cast<symbolic::Expression>().sum() - z1(i) == 0);
    prog.AddLinearConstraint(beta[i].cast<symbolic::Expression>().sum() - z2(i) == 0);
    prog.AddBoundingBoxConstraint(0, 1, alpha[i]);
    prog.AddBoundingBoxConstraint(0, 1, beta[i]);
    p1 += V[i] * alpha[i];
    p2 += V[i] * beta[i];
    prog.AddLinearConstraint(z1(i) + z2(i) <= 1);
  }
  prog.AddCost((p1 - p2).dot(p1 - p2));
  prog.SetSolverOption(solvers::SolverType::kGurobi, "PoolSearchMode", 2);
  prog.SetSolverOption(solvers::SolverType::kGurobi, "PoolSolutions", 3);

  GurobiSolver gurobi_solver;
  if (gurobi_solver.available()) {
    auto sol_result = gurobi_solver.Solve(prog);
    EXPECT_EQ(sol_result, SolutionResult::kSolutionFound);
    double optimal_cost = prog.GetOptimalCost();
    std::array<double, 2> suboptimal_cost;
    for (int i = 0; i < 2; ++i) {
      suboptimal_cost[i] = prog.GetSuboptimalCost(i);
      Eigen::Vector2d p1_val = Eigen::Vector2d::Zero();
      Eigen::Vector2d p2_val = Eigen::Vector2d::Zero();
      for (int j = 0; j < 3; ++j) {
        auto alpha_j = prog.GetSuboptimalSolution(alpha[j], i);
        auto beta_j = prog.GetSuboptimalSolution(beta[j], i);
        p1_val += V[j] * alpha_j;
        p2_val += V[j] * beta_j;
        EXPECT_NEAR(alpha_j.sum(), prog.GetSuboptimalSolution(z1(j), i), 1E-6);
        EXPECT_NEAR(beta_j.sum(), prog.GetSuboptimalSolution(z2(j), i), 1E-6);
      }
      EXPECT_NEAR((p1_val - p2_val).squaredNorm(), suboptimal_cost[i], 1E-6);
    }
    EXPECT_GE(optimal_cost, suboptimal_cost[0]);
    EXPECT_GE(suboptimal_cost[0], suboptimal_cost[1]);
  }
}
}  // namespace test
}  // namespace solvers
}  // namespace drake
