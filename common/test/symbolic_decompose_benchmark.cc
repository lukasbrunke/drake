#include <benchmark/benchmark.h>

#include "drake/common/symbolic_decompose.h"
#include "drake/tools/performance/fixture_common.h"

namespace drake {
namespace symbolic {
namespace {
class ExtractVariableFixture : public benchmark::Fixture {
 public:
  using benchmark::Fixture::SetUp;
  void SetUp(const ::benchmark::State&) override {
    tools::performance::AddMinMaxStatistics(this);
    e_.resize(100);
    vars_.resize(100);
    for (int i = 0; i < e_.rows(); ++i) {
      vars_(i) = symbolic::Variable("x" + std::to_string(i));
      e_(i) = vars_(i);
    }
  }

 protected:
  VectorX<symbolic::Expression> e_;
  VectorX<symbolic::Variable> vars_;
};

BENCHMARK_F(ExtractVariableFixture, EigenVector)(benchmark::State& state) {
  VectorX<symbolic::Variable> vars;
  std::unordered_map<symbolic::Variable::Id, int> map_var_to_index;
  for (auto _ : state) {
    for (int i = 0; i < e_.rows(); ++i) {
      ExtractAndAppendVariablesFromExpression(e_(i), &vars, &map_var_to_index);
    }
  }
}

BENCHMARK_F(ExtractVariableFixture, StdVector)(benchmark::State& state) {
  std::vector<symbolic::Variable> vars;
  std::unordered_map<symbolic::Variable::Id, int> map_var_to_index;
  for (auto _ : state) {
    for (int i = 0; i < e_.rows(); ++i) {
      ExtractAndAppendVariablesFromExpression(e_(i), &vars, &map_var_to_index);
    }
  }
}

BENCHMARK_F(ExtractVariableFixture, std_vector_push_back)
(benchmark::State& state) {
  std::vector<symbolic::Variable> v;
  for (auto _ : state) {
    for (int i = 0; i < vars_.rows(); ++i) {
      v.push_back(vars_(i));
    }
  }
}

BENCHMARK_F(ExtractVariableFixture, eigen_vector_resize)
(benchmark::State& state) {
  VectorX<symbolic::Variable> v;
  for (auto _ : state) {
    for (int i = 0; i < vars_.rows(); ++i) {
      const int v_size = v.size();
      v.conservativeResize(v_size + 1, Eigen::NoChange);
      v(v_size) = vars_(i);
    }
  }
}
}  // namespace
}  // namespace symbolic
}  // namespace drake

BENCHMARK_MAIN();
