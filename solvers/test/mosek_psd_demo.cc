#include <ctime>
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace solvers {
int DoMain() {
  const int n = 200;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Random(n, n);
  Q = Q + Q.transpose();

  MathematicalProgram prog;
  auto X = prog.NewSymmetricContinuousVariables(n, "X");
  prog.AddPositiveSemidefiniteConstraint(X);

  for (int i = 0; i < n; ++i) {
    prog.AddLinearConstraint(X(i, i) == 0);
  }
  std::clock_t start = std::clock();
  const symbolic::Expression cost = (Q * X.cast<symbolic::Expression>()).trace();
  std::cout << "computing cost = ((Q' * X).trace()) time (s): "
            << (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC)
            << "\n";
  start = std::clock();
  prog.AddLinearCost(cost);
  std::cout << "AddLinearCost(cost) time (s): "
            << (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC)
            << "\n";

  MosekSolver solver;
  MathematicalProgramResult result;
  start = std::clock();
  solver.Solve(prog, {}, {}, &result);
  std::cout << "solver.Solve() time (s): "
            << (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC)
            << "\n";
  std::cout << "Mosek actual solve time (s): "
            << result.get_solver_details<MosekSolver>().optimizer_time << "\n";
  return 0;
}
}  // namespace solvers
}  // namespace drake

int main() { return drake::solvers::DoMain(); }
