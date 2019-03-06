#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"

#include <ctime>
namespace drake {
namespace solvers {
int DoMain() {
  auto mosek_license = MosekSolver::AcquireLicense();
  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<2>();
  prog.AddLinearConstraint(x(0) >= 0);
  prog.AddLinearConstraint(x(1) >= 0);
  prog.AddLinearConstraint(x(0) <= 1);
  prog.AddLinearConstraint(x(1) <= 1);

  MosekSolver solver;
  MathematicalProgramResult result;
  std::clock_t start = std::clock();
  solver.Solve(prog, {}, {}, &result);
  double duration =
      (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "time: " << duration << "s \n";
  std::cout << "mosek solve time: "
            << result.get_solver_details<MosekSolver>().optimizer_time
            << "s \n";
  return 0;
}
}  // namespace solvers
}  // namespace drake

int main() { return drake::solvers::DoMain(); }
