#include "drake/systems/analysis/robust_verification.h"

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/systems/framework/vector_system.h"

namespace drake {
namespace systems {
namespace analysis {
// This tester exposes some of the private interfaces of RobustVerification, so
// that we can test it.
class RobustVerificationTester {
 public:
  RobustVerificationTester() {}

  static void CalcVdot(const RobustInvariantSetVerfication& robust_verification,
                       const symbolic::Polynomial& V,
                       const Eigen::Ref<const Eigen::VectorXd>& x_val,
                       int taylor_order, VectorX<symbolic::Polynomial>* Vdot) {
    return robust_verification.CalcVdot(V, x_val, taylor_order, Vdot);
  }
};
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

// A simple control affine system
// ẋ=-x+x³+(1+x²)u
// y = x
class ControlAffineToySystem : public VectorSystem<symbolic::Expression> {
 public:
  ControlAffineToySystem()
      : systems::VectorSystem<symbolic::Expression>(
            1, 1) {                   // One input, one output
    this->DeclareContinuousState(1);  // One state variables.
  }

  virtual ~ControlAffineToySystem() {}

 private:
  virtual void DoCalcVectorTimeDerivatives(
      const systems::Context<symbolic::Expression>&,
      const Eigen::VectorBlock<const VectorX<symbolic::Expression>>& input,
      const Eigen::VectorBlock<const VectorX<symbolic::Expression>>& state,
      Eigen::VectorBlock<VectorX<symbolic::Expression>>* derivatives)
      const override {
    (*derivatives)(0) =
        -state(0) + pow(state(0), 3) + (1 + pow(state(0), 2)) * input(0);
  }

  virtual void DoCalcVectorOutput(
      const Context<symbolic::Expression>& context,
      const Eigen::VectorBlock<const VectorX<symbolic::Expression>>& input,
      const Eigen::VectorBlock<const VectorX<symbolic::Expression>>& state,
      Eigen::VectorBlock<VectorX<symbolic::Expression>>* output)
      const override {
    (*output)(0) = state(0);
  }
};

GTEST_TEST(TestRobustVerification, TestConstructor1) {
  // Test the constructor with a system that is not control affine. Should throw
  // a logic error.
  NonControlAffineToySystem non_control_affine_system;
  Eigen::Matrix2d K = Eigen::Matrix2d::Identity();
  Eigen::Vector2d k0 = Eigen::Vector2d::Ones();
  Eigen::Matrix<double, 2, 3> x_err_vertices;

  const int l_degree = 2;
  EXPECT_THROW(RobustInvariantSetVerfication(non_control_affine_system, K, k0,
                                             x_err_vertices, l_degree),
               std::logic_error);
}

GTEST_TEST(TestRobustVerification, TestConstructor2) {
  // Test the contructor for a control affine system.
  ControlAffineToySystem control_affine_system;
  Eigen::Matrix<double, 1, 1> K;
  K << -1;
  Vector1<double> k0;
  k0 << 0;
  constexpr int kNumXerrVertices = 2;
  Eigen::Matrix<double, 1, kNumXerrVertices> x_err_vertices;
  x_err_vertices << -0.01, 0.01;

  const int l_degree = 2;
  RobustInvariantSetVerfication verification(control_affine_system, K, k0,
                                             x_err_vertices, l_degree);

  EXPECT_EQ(verification.l_polynomials().size(), kNumXerrVertices);
  for (int i = 0; i < kNumXerrVertices; ++i) {
    EXPECT_EQ(verification.l_polynomials()[i].TotalDegree(), l_degree);
  }
}

GTEST_TEST(TestRobustVerification, TestCalcVdot) {
  ControlAffineToySystem control_affine_system;
  Eigen::Matrix<double, 1, 1> K;
  K << -1;
  Vector1<double> k0;
  k0 << 0;
  constexpr int kNumXerrVertices = 2;
  Eigen::Matrix<double, 1, kNumXerrVertices> x_err_vertices;
  x_err_vertices << -0.01, 0.01;

  const int l_degree = 2;
  RobustInvariantSetVerfication verification(control_affine_system, K, k0,
                                             x_err_vertices, l_degree);

  const solvers::VectorXIndeterminate& x = verification.x();
  const Vector1<double> x_val(0.1);
  const symbolic::Polynomial V(x.cast<symbolic::Expression>().dot(x));
  VectorX<symbolic::Polynomial> Vdot(2);
  const int taylor_order = 3;
  RobustVerificationTester::CalcVdot(verification, V, x_val, taylor_order,
                                     &Vdot);

  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(verification.x());
  VectorX<symbolic::Polynomial> Vdot_expected(2);
  for (int i = 0; i < 2; ++i) {
    Vdot_expected(i) =
        dVdx(0) *
        symbolic::Polynomial(
            -x(0) + pow(x(0), 3) +
            (1 + x(0) * x(0)) * (K(0) * (x(0) + x_err_vertices(0, i)) + k0(0)));
    EXPECT_PRED2(
        symbolic::test::PolyEqual,
        (Vdot(i) - Vdot_expected(i)).RemoveTermsWithSmallCoefficients(1E-8),
        symbolic::Polynomial());
  }
}

GTEST_TEST(TestRobustVerification, TestConstructLagrangianStep) {
  ControlAffineToySystem control_affine_system;
  Eigen::Matrix<double, 1, 1> K;
  K << -1;
  Vector1<double> k0;
  k0 << 0;
  constexpr int kNumXerrVertices = 2;
  Eigen::Matrix<double, 1, kNumXerrVertices> x_err_vertices;
  x_err_vertices << -0.01, 0.01;

  const int l_degree = 2;
  RobustInvariantSetVerfication verification(control_affine_system, K, k0,
                                             x_err_vertices, l_degree);

  const solvers::VectorXIndeterminate& x = verification.x();
  const Vector1<double> x_val(0.1);
  const symbolic::Polynomial V(x.cast<symbolic::Expression>().dot(x));
  auto prog = verification.ConstructLagrangianStep(V, x_val, 3, 0.1);

  solvers::MosekSolver mosek_solver;
  mosek_solver.set_stream_logging(true, "");
  auto result = mosek_solver.Solve(*prog);
  EXPECT_EQ(result, solvers::SolutionResult::kSolutionFound);
  const double eps_value = -prog->GetOptimalCost();
  EXPECT_GE(eps_value, 0);
}
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
