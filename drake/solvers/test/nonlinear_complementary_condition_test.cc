#include "drake/solvers/nonlinear_complementary_constraint.h"

#include <gtest/gtest.h>

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace solvers {
namespace {
// Solve the problem
// min 0.5 * (x(0) + x(1) - y(0) - 15)² + 0.5 * (x(0) + x(1) + y(1) - 15)²
// s.t 0 <= y(0) ⊥ 8/3 * x(0) + 2x(1) + 2y(0) + 8/3 * y(1) - 36 >= 0
//     0 <= y(1) ⊥ 2x(0) + 5/4 * x(1) + 5/4 * y(0) + 2y(1) - 25 >= 0
//     0 <= x(0) <= 10
//     0 <= x(1) <= 10
// This problem is taken from Fukushima, M. Luo, Z. -Q. Pang, J. -S
// "A globally convergent Sequential Quadratic Programming Algorithm for
// Mathematical Programs with Linear Complementarity Constraints",
// Computational Optimization and Applications, 10(1), pp. 5-34, 1998
struct
}  // namespace
}  // namespace solvers
}  // namespace drake