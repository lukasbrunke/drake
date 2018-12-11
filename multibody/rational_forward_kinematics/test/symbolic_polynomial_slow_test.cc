#include <ctime>
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace multibody {
int DoMain() {
  solvers::MathematicalProgram prog;
  auto t = prog.NewIndeterminates<7>("t");
  const symbolic::Variables t_variables(t);
  const auto monomial_basis =
      GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(t_variables);
  std::clock_t start = std::clock();
  prog.NewSosPolynomial(monomial_basis);
  double duration =
      (std::clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "duration (s): " << duration << "\n";
  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
