#include <thread>
#include <vector>
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace solvers {
bool Foo() {
  MathematicalProgram prog;
  constexpr int rows = 30;
  auto X = prog.NewSymmetricContinuousVariables<rows>();
  prog.AddPositiveSemidefiniteConstraint(X);
  Eigen::Matrix<symbolic::Variable, rows, 1> X_diag;
  for (int i = 0; i < rows; ++i) {
    X_diag(i) = X(i, i);
  }
  prog.AddLinearEqualityConstraint(Eigen::Matrix<double, 1, rows>::Ones(),
                                   Vector1d(1), X_diag);
  prog.AddLinearCost((Eigen::Matrix<double, rows, rows>::Ones() * X).trace());
  MosekSolver solver;
  //ScsSolver scs_solver;
  auto result = solver.Solve(prog);
  drake::log()->info(fmt::format("result is_success={}", result.is_success()));
  return result.is_success();
}

int DoMain() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; ++i) {
    threads.push_back(std::thread(Foo));
  }
  for (auto& th : threads) {
    th.join();
  }
  return 0;
}
}  // namespace solvers
}  // namespace drake

int main() {
  return drake::solvers::DoMain();
}
