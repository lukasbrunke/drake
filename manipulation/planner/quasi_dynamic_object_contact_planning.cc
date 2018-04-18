#include "drake/manipulation/planner/quasi_dynamic_object_contact_planning.h"

#include "drake/solvers/bilinear_product_util.h"
#include "drake/solvers/mixed_integer_optimization_util.h"

using drake::symbolic::Expression;
namespace drake {
namespace manipulation {
namespace planner {
QuasiDynamicObjectContactPlanning::QuasiDynamicObjectContactPlanning(
    int nT, double dt, double mass,
    const Eigen::Ref<const Eigen::Matrix3d>& I_B,
    const Eigen::Ref<const Eigen::Vector3d>& p_BC,
    const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV, int num_pushers,
    const std::vector<BodyContactPoint>& Q, bool add_second_order_cone_for_R)
    : ObjectContactPlanning(nT, mass, p_BC, p_BV, num_pushers, Q,
                            add_second_order_cone_for_R),
      dt_{dt},
      I_B_((I_B + I_B.transpose()) / 2),
      v_B_{get_mutable_prog()->NewContinuousVariables<3, Eigen::Dynamic>(
          3, this->nT(), "v_B")},
      omega_B_{get_mutable_prog()->NewContinuousVariables<3, Eigen::Dynamic>(
          3, this->nT(), "omega_B")} {
  // Add the interpolation constraint
  DRAKE_DEMAND(dt_ > 0);
  AddInterpolationConstraint();
}

void QuasiDynamicObjectContactPlanning::AddInterpolationConstraint() {
  phi_v_B_ << -1, -0.5, 0, 0.5, 1;
  phi_omega_B_ << -M_PI, -M_PI / 2, 0, M_PI / 2, M_PI;
  for (int i = 0; i < 3; ++i) {
    b_v_B_[i].resize(nT());
    b_omega_B_[i].resize(nT());
    for (int j = 0; j < nT(); ++j) {
      b_v_B_[i][j] = get_mutable_prog()->NewBinaryVariables(
          solvers::CeilLog2(phi_v_B_.rows() - 1),
          "b_v_B_[" + std::to_string(i) + "][" + std::to_string(j) + "]");
      b_omega_B_[i][j] = get_mutable_prog()->NewBinaryVariables(
          solvers::CeilLog2(phi_omega_B_.rows() - 1),
          "b_omega_B_[" + std::to_string(i) + "][" + std::to_string(j) + "]");
    }
  }

  // R_times_v_B.col(knot) approximates R_WB_[knot] * v_B_.col(knot)
  Eigen::Matrix<Expression, 3, Eigen::Dynamic> R_times_v_B(3, nT());
  // R_times_omega_B_hat approximates R_WB_[knot] * hat(omega_B_.col(knot))
  std::vector<Matrix3<Expression>> R_times_omega_B_hat(nT());
  for (int knot = 0; knot < nT(); ++knot) {
    // R_times_v_B_element(i, j) represents R_WB[knot](i, j) * v_B[knot](j)
    const auto R_times_v_B_element =
        get_mutable_prog()->NewContinuousVariables<3, 3>(
            "R_times_v_B[" + std::to_string(knot) + "]");
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        solvers::AddBilinearProductMcCormickEnvelopeSos2(
            get_mutable_prog(), R_WB()[knot](i, j), v_B_(j, knot),
            R_times_v_B_element(i, j), phi_R_WB(), phi_v_B_,
            b_R_WB()[knot][i][j].cast<Expression>(),
            b_v_B_[j][knot].cast<Expression>(),
            solvers::IntervalBinning::kLogarithmic);
      }
    }
    R_times_v_B.col(knot) =
        R_times_v_B_element.cast<Expression>().rowwise().sum();

    // R_times_omega_B_hat_element(i, j) approximates R_WB_flat[knot](i) *
    // omega_B(j, knot)
    Eigen::Matrix<symbolic::Expression, 9, 3> R_times_omega_B_hat_element;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          if (k != j) {
            R_times_omega_B_hat_element(3 * j + i, k) =
                get_mutable_prog()->NewContinuousVariables<1, 1>(
                    "R_WB[" + std::to_string(knot) + "](" + std::to_string(i) +
                    "," + std::to_string(j) + ")*omega_B(" + std::to_string(k) +
                    "," + std::to_string(knot) + ")")(0);
            solvers::AddBilinearProductMcCormickEnvelopeSos2(
                get_mutable_prog(), R_WB()[knot](i, j), omega_B_(k, knot),
                R_times_omega_B_hat_element(3 * j + i, k), phi_R_WB(),
                phi_omega_B_, b_R_WB()[knot][i][j].cast<Expression>(),
                b_omega_B_[k][knot].cast<Expression>(),
                solvers::IntervalBinning::kLogarithmic);
          }
        }
      }
    }
    Matrix3<Expression> omega_hat;
    // clang-format off
    omega_hat << 0, -omega_B_(2, knot), omega_B_(1, knot),
              omega_B_(2, knot), 0, -omega_B_(0, knot),
              -omega_B_(1, knot), omega_B_(0, knot), 0;
    // clang-format on
    Matrix3<Expression> R_times_omega_B_hat_bilinear = R_WB()[knot] * omega_hat;
    solvers::VectorDecisionVariable<9> R_flat;
    R_flat << R_WB()[knot].col(0), R_WB()[knot].col(1), R_WB()[knot].col(2);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        R_times_omega_B_hat[knot](i, j) = solvers::ReplaceBilinearTerms(
            R_times_omega_B_hat_bilinear(i, j), R_flat, omega_B_.col(knot),
            R_times_omega_B_hat_element.cast<Expression>());
      }
    }
  }

  for (int interval = 0; interval < nT() - 1; ++interval) {
    const int knot0 = interval;
    const int knot1 = interval + 1;
    get_mutable_prog()->AddLinearEqualityConstraint(
        p_WB()[knot1] - p_WB()[knot0] -
            0.5 * dt_ * (R_times_v_B.col(knot0) + R_times_v_B.col(knot1)),
        Eigen::Vector3d::Zero());

    get_mutable_prog()->AddLinearEqualityConstraint(
        R_WB()[knot1] - R_WB()[knot0] -
            0.5 * dt_ *
                (R_times_omega_B_hat[knot0] + R_times_omega_B_hat[knot1]),
        Eigen::Matrix3d::Zero());
  }
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
