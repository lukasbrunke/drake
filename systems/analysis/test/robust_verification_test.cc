#include "drake/systems/analysis/robust_verification.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/systems/framework/vector_system.h"

namespace drake {
namespace systems {
namespace analysis {
namespace {
// A toy system that is not control affine.
// ẋ₁ = x₁ + (x₂-1)u₁
// ẋ₂ = x₂² + (x₁+1)u₂²
// y = x
class NonControlAffineToySystem
    : public systems::VectorSystem<symbolic::Expression> {
 public:
  NonControlAffineToySystem()
      : systems::VectorSystem<symbolic::Expression>(
            2, 2) {                   // Two inputs, two outputs
    this->DeclareContinuousState(2);  // Two state variables.
  }

  virtual ~NonControlAffineToySystem() {}

 private:
  virtual void DoCalcVectorTimeDerivatives(
      const systems::Context<symbolic::Expression>&,
      const Eigen::VectorBlock<const VectorX<symbolic::Expression>>& input,
      const Eigen::VectorBlock<const VectorX<symbolic::Expression>>& state,
      Eigen::VectorBlock<VectorX<symbolic::Expression>>* derivatives)
      const override {
    (*derivatives)(0) = state(0) + (state(1) - 1) * input(0);
    (*derivatives)(1) =
        state(1) * state(1) + (state(0) + 1) * input(1) * input(1);
  }

  virtual void DoCalcVectorOutput(
      const Context<symbolic::Expression>& context,
      const Eigen::VectorBlock<const VectorX<symbolic::Expression>>& input,
      const Eigen::VectorBlock<const VectorX<symbolic::Expression>>& state,
      Eigen::VectorBlock<VectorX<symbolic::Expression>>* output)
      const override {
    (*output)(0) = state(0);
    (*output)(1) = state(1);
  }
};

GTEST_TEST(TestRobustVerification, TestConstructor) {
  // Test the constructor with a system that is not control affine. Should throw
  // a logic error.
  NonControlAffineToySystem non_control_affine_system;
  Eigen::Matrix2d K = Eigen::Matrix2d::Identity();
  Eigen::Vector2d k0 = Eigen::Vector2d::Ones();
  Eigen::Matrix<double, 2, 3> x_err_vertices;

  EXPECT_THROW(RobustInvariantSetVerfication(non_control_affine_system, K, k0,
                                             x_err_vertices),
               std::logic_error);
}
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
