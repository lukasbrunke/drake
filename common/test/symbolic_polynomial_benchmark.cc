#include <benchmark/benchmark.h>

#include "drake/common/symbolic.h"
#include "drake/tools/performance/fixture_common.h"

namespace drake {
namespace symbolic {
namespace {
class SymbolicFixture : public benchmark::Fixture {
 public:
  using benchmark::Fixture::SetUp;
  void SetUp(const ::benchmark::State&) override {
    tools::performance::AddMinMaxStatistics(this);
    x_ = Vector3<symbolic::Variable>();
    for (int i = 0; i < x_.rows(); ++i) {
      x_(i) = symbolic::Variable("x" + std::to_string(i));
    }
    a_ = symbolic::Variable("a");
    b_ = symbolic::Variable("b");
    c_ = symbolic::Variable("c");
    d_ = symbolic::Variable("d");
  }

 protected:
  VectorX<symbolic::Variable> x_;
  symbolic::Variable a_;
  symbolic::Variable b_;
  symbolic::Variable c_;
  symbolic::Variable d_;
};

BENCHMARK_F(SymbolicFixture, polynomial_addition)(benchmark::State& state) {
  using std::pow;
  for (auto _ : state) {
    symbolic::Monomial monomial1({{x_(0), 1}, {x_(1), 2}});
    symbolic::Monomial monomial2({{x_(0), 2}, {x_(1), 2}});
    symbolic::Monomial monomial3({{x_(2), 2}, {x_(1), 1}});
    symbolic::Monomial monomial4({{x_(2), 2}, {x_(0), 3}, {x_(1), 1}});
    symbolic::Monomial monomial5({{x_(2), 2}, {x_(0), 1}, {x_(1), 1}});
    symbolic::Monomial monomial6({{x_(2), 2}, {x_(1), 4}});
    // Generate arbitrary symbolic polynomials and sum them up together.
    symbolic::Polynomial sum{};
    for (int i = 0; i < 1000; ++i) {
      const symbolic::Polynomial p{{{monomial1, a_ * (i + 1) * b_},
                                    {monomial2, a_ * (i + 2) * c_ + b_},
                                    {monomial3, a_ * (i + 1) * c_ + b_ * c_},
                                    {monomial4, a_ + b_ * c_ * c_},
                                    {monomial5, a_ + 2},
                                    {monomial6, 2}}};
      sum += p;
    }
  }
}
}  // namespace
}  // namespace symbolic
}  // namespace drake

BENCHMARK_MAIN();
