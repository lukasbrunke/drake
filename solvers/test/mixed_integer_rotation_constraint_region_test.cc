#include <gtest/gtest.h>

#include "drake/solvers/create_constraint.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
namespace {
GTEST_TEST(TestRegion, Test1) {
  MathematicalProgram prog;
  auto u = prog.NewContinuousVariables<3>();
  prog.AddConstraint(internal::ParseQuadraticConstraint(
      u.cast<symbolic::Expression>().dot(u), 1, 1));
  auto v = prog.NewContinuousVariables<3>();
  prog.AddConstraint(internal::ParseQuadraticConstraint(
      v.cast<symbolic::Expression>().dot(v), 1, 1));
  prog.AddConstraint(internal::ParseQuadraticConstraint(
      v.cast<symbolic::Expression>().dot(u), 0, 0));
  prog.AddBoundingBoxConstraint(Eigen::Vector3d(0.5, 0.5, 0),
                                Eigen::Vector3d(1, 1, 0.5), u);
  auto v_x_bounds = prog.AddBoundingBoxConstraint(0, 0, v(0));
  auto v_y_bounds = prog.AddBoundingBoxConstraint(0, 0, v(1));
  auto v_z_bounds = prog.AddBoundingBoxConstraint(0, 0, v(2));
  for (int i = 0; i < 4; ++i) {
    v_x_bounds.evaluator()->UpdateLowerBound(Vector1d(-1 + i * 0.5));
    v_x_bounds.evaluator()->UpdateUpperBound(Vector1d(-0.5 + i * 0.5));
    for (int j = 0; j < 4; ++j) {
      v_y_bounds.evaluator()->UpdateLowerBound(Vector1d(-1 + j * 0.5));
      v_y_bounds.evaluator()->UpdateUpperBound(Vector1d(-0.5 + j * 0.5));
      for (int k = 0; k < 4; ++k) {
        v_z_bounds.evaluator()->UpdateLowerBound(Vector1d(-1 + k * 0.5));
        v_z_bounds.evaluator()->UpdateUpperBound(Vector1d(-0.5 + k * 0.5));
        prog.SetInitialGuess(u, Eigen::Vector3d(0.5, 0.5, 0.5));
        prog.SetInitialGuess(v, Eigen::Vector3d(0.5, 0.5, -0.5));

        const auto result = prog.Solve();
        std::cout << "i, j, k:" << i << "," << j << "," << k << " " << result << "\n";
        if (result == SolutionResult::kInfeasibleConstraints) {
          std::cout << "x interval: [" << -1 + i * 0.5 << " , "
                    << -0.5 + i * 0.5 << "]\n";
          std::cout << "y interval: [" << -1 + j * 0.5 << " , "
                    << -0.5 + j * 0.5 << "]\n";
          std::cout << "z interval: [" << -1 + k * 0.5 << " , "
                    << -0.5 + k * 0.5 << "]\n";
          std::cout << "\n";
        }
      }
    }
  }
}
}
}
}
