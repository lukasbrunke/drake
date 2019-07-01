#include "drake/common/symbolic.h"

#include <ctime>

namespace drake {
namespace symbolic {
int DoMain() {
  Vector3<symbolic::Variable> x;
  for (int i = 0; i < 3; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }

  symbolic::Expression e;
  using std::exp;
  using std::pow;
  using std::sin;
  e = ((x(0) + x(2)) * (x(1) + pow(x(2) - x(0), 4)) + sin(x(1))) *
      (pow(x(2) + 2 * x(1) + sin(x(0) + x(2)), 30) -
       (2 * x(1) + exp(x(2) + 2 * x(1))));
  symbolic::Expression f1, f2;
  f1 = pow((x(2) + pow(x(1) + x(2), 3)), 10) -
       (x(0) + 3 * x(1)) * (2 + x(0) + 3 + pow(x(1) - x(0), 4));
  f2 = -pow(x(2) + 1 + 2 * x(0), 4) +
       (x(1) + x(2) + pow(x(2) - x(1) + 10, 5)) * (x(2) + 3 * x(0) + 5);

  std::clock_t start = std::clock();
  const int iter = 10;
  for (int i = 0; i < iter; ++i) {
    (e * f1).Expand();
    (e * f2).Expand();
  }
  std::cout << " compute (e * f1).Expand() and (e * f2).Expand(): "
            << (std::clock() - start) /
                   (static_cast<double>(CLOCKS_PER_SEC) * iter)
            << "s\n";

  start = std::clock();
  for (int i = 0; i < iter; ++i) {
    symbolic::Expression e_expand = e.Expand();
    (e_expand * f1).Expand();
    (e_expand * f2).Expand();
  }
  std::cout << " compute e_expand = e.Expand(); (e_expand * f1).Expand() and "
               "(e_expand * f2).Expand(): "
            << (std::clock() - start) /
                   (static_cast<double>(CLOCKS_PER_SEC) * iter)
            << "s\n";

  return 0;
}
}  // namespace symbolic
}  // namespace drake

int main() { return drake::symbolic::DoMain(); }
