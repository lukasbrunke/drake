#include "drake/solvers/mathematical_program.h"
#include "drake/common/symbolic_monomial_util.h"
#include <ctime>

namespace drake {
namespace solvers {
int DoMain() {
  MathematicalProgram prog;
  constexpr int kNumT = 12;
  auto t = prog.NewIndeterminates<kNumT>();
  const auto m = symbolic::MonomialBasis<kNumT, 3>(symbolic::Variables(t));
  std::cout << "add sdsos polynomial.\n";
  std::clock_t start = clock();
  prog.NewNonnegativePolynomial(m, MathematicalProgram::NonnegativePolynomial::kSdsos);
  double duration = (clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
  std::cout << "duration (s): " << duration << "\n";
  return 0;
}
}  // namespace solvers
}  // namespace drake

int main() {
  return drake::solvers::DoMain();
}
