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
    const std::vector<BodyContactPoint>& Q, double max_linear_velocity,
    double max_angular_velocity, bool add_second_order_cone_for_R)
    : ObjectContactPlanning(nT, mass, p_BC, p_BV, num_pushers, Q,
                            add_second_order_cone_for_R),
      dt_{dt},
      I_B_((I_B + I_B.transpose()) / 2),
      v_B_{get_mutable_prog()->NewContinuousVariables<3, Eigen::Dynamic>(
          3, this->nT(), "v_B")},
      omega_B_{get_mutable_prog()->NewContinuousVariables<3, Eigen::Dynamic>(
          3, this->nT(), "omega_B")} {
  DRAKE_DEMAND(dt_ > 0);
  // Add the interpolation constraint
  AddTranslationInterpolationConstraint(max_linear_velocity);
  AddOrientationInterpolationConstraint(max_angular_velocity);
}

void QuasiDynamicObjectContactPlanning::AddTranslationInterpolationConstraint(
    double max_linear_velocity) {
  phi_v_B_ = Eigen::Matrix<double, 5, 1>::LinSpaced(-max_linear_velocity,
                                                    max_linear_velocity);
  phi_v_B_(2) = 0;
  for (int i = 0; i < 3; ++i) {
    b_v_B_[i].resize(nT());
    for (int j = 0; j < nT(); ++j) {
      b_v_B_[i][j] = get_mutable_prog()->NewBinaryVariables(
          solvers::CeilLog2(phi_v_B_.rows() - 1),
          "b_v_B_[" + std::to_string(i) + "][" + std::to_string(j) + "]");
    }
  }
  // R_times_v_B.col(knot) approximates R_WB_[knot] * v_B_.col(knot)
  Eigen::Matrix<Expression, 3, Eigen::Dynamic> R_times_v_B(3, nT());

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
  }

  for (int interval = 0; interval < nT() - 1; ++interval) {
    const int knot0 = interval;
    const int knot1 = interval + 1;
    get_mutable_prog()->AddLinearEqualityConstraint(
        p_WB()[knot1] - p_WB()[knot0] -
            0.5 * dt_ * (R_times_v_B.col(knot0) + R_times_v_B.col(knot1)),
        Eigen::Vector3d::Zero());
  }
}

void QuasiDynamicObjectContactPlanning::AddOrientationInterpolationConstraint(
    double max_angular_velocity) {
  phi_omega_B_ = Eigen::Matrix<double, 5, 1>::LinSpaced(-max_angular_velocity,
                                                        max_angular_velocity);
  phi_omega_B_(2) = 0;
  for (int i = 0; i < 3; ++i) {
    b_omega_average_[i].resize(nT() - 1);
    for (int j = 0; j < nT() - 1; ++j) {
      b_omega_average_[i][j] = get_mutable_prog()->NewBinaryVariables(
          solvers::CeilLog2(phi_omega_B_.rows() - 1),
          "b_omega_average_[" + std::to_string(i) + "][" + std::to_string(j) +
              "]");
    }
  }

  const auto omega_average =
      get_mutable_prog()->NewContinuousVariables<3, Eigen::Dynamic>(
          3, nT() - 1, "omega_average");

  // omega_average.col(i) = (omega_B_.col(i) + omega_B_.col(i + 1)) / 2
  get_mutable_prog()->AddLinearEqualityConstraint(
      omega_average - 0.5 * omega_B_.leftCols(nT() - 1) -
          0.5 * omega_B_.rightCols(nT() - 1),
      Eigen::Matrix3Xd::Zero(3, nT() - 1));
  // R1_times_omega_average_hat[knot] approximates R_WB[knot] *
  // SkewSymmetric(omega_average.col(knot))
  // R2_times_omega_average_hat[knot] approximates R_WB[knot + 1] *
  // SkewSymmetric(omega_average.col(knot))
  std::vector<Matrix3<symbolic::Expression>> R1_times_omega_average_hat(nT() -
                                                                        1);
  std::vector<Matrix3<symbolic::Expression>> R2_times_omega_average_hat(nT() -
                                                                        1);
  for (int knot = 0; knot < nT() - 1; ++knot) {
    // R1_flat_times_omega_average_element(i, j) approximates R_WB_flat[knot](i)
    // * omega_average(j, knot);
    // R2_flat_times_omega_average_element(i, j) approximates R_WB_flat[knot +
    // 1](i) * omega_average(j, knot);
    Eigen::Matrix<symbolic::Expression, 9, 3>
        R1_flat_times_omega_average_element,
        R2_flat_times_omega_average_element;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          if (k != j) {
            R1_flat_times_omega_average_element(3 * j + i, k) =
                get_mutable_prog()->NewContinuousVariables<1, 1>(
                    "R_WB[" + std::to_string(knot) + "](" + std::to_string(i) +
                    "," + std::to_string(j) + ")*omega_average(" +
                    std::to_string(k) + "," + std::to_string(knot) + ")")(0);
            const auto phi1 = solvers::AddBilinearProductMcCormickEnvelopeSos2(
                get_mutable_prog(), R_WB()[knot](i, j), omega_average(k, knot),
                R1_flat_times_omega_average_element(3 * j + i, k), phi_R_WB(),
                phi_omega_B_, b_R_WB()[knot][i][j].cast<Expression>(),
                b_omega_average_[k][knot].cast<Expression>(),
                solvers::IntervalBinning::kLogarithmic);
            R2_flat_times_omega_average_element(3 * j + i, k) =
                get_mutable_prog()->NewContinuousVariables<1, 1>(
                    "R_WB[" + std::to_string(knot + 1) + "](" +
                    std::to_string(i) + "," + std::to_string(j) +
                    ")*omega_average(" + std::to_string(k) + "," +
                    std::to_string(knot) + ")")(0);
            const auto phi2 = solvers::AddBilinearProductMcCormickEnvelopeSos2(
                get_mutable_prog(), R_WB()[knot + 1](i, j),
                omega_average(k, knot),
                R2_flat_times_omega_average_element(3 * j + i, k), phi_R_WB(),
                phi_omega_B_, b_R_WB()[knot + 1][i][j].cast<Expression>(),
                b_omega_average_[k][knot].cast<Expression>(),
                solvers::IntervalBinning::kLogarithmic);

            get_mutable_prog()->AddLinearEqualityConstraint(
                (phi1.cast<symbolic::Expression>().colwise().sum() -
                 phi2.cast<symbolic::Expression>().colwise().sum())
                    .transpose(),
                Eigen::VectorXd::Zero(phi1.cols()));
          }
        }
      }
    }
    Matrix3<Expression> omega_average_hat;
    // clang-format off
    omega_average_hat << 0, -omega_average(2, knot), omega_average(1, knot),
                         omega_average(2, knot), 0, -omega_average(0, knot),
                         -omega_average(1, knot), omega_average(0, knot), 0;
    // clang-format on
    const Matrix3<Expression> R1_times_omega_average_hat_bilinear =
        R_WB()[knot] * omega_average_hat;
    const Matrix3<Expression> R2_times_omega_average_hat_bilinear =
        R_WB()[knot + 1] * omega_average_hat;
    solvers::VectorDecisionVariable<9> R1_flat, R2_flat;
    R1_flat << R_WB()[knot].col(0), R_WB()[knot].col(1), R_WB()[knot].col(2);
    R2_flat << R_WB()[knot + 1].col(0), R_WB()[knot + 1].col(1),
        R_WB()[knot + 1].col(2);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        R1_times_omega_average_hat[knot](i, j) = solvers::ReplaceBilinearTerms(
            R1_times_omega_average_hat_bilinear(i, j), R1_flat,
            omega_average.col(knot),
            R1_flat_times_omega_average_element.cast<Expression>());
        R2_times_omega_average_hat[knot](i, j) = solvers::ReplaceBilinearTerms(
            R2_times_omega_average_hat_bilinear(i, j), R2_flat,
            omega_average.col(knot),
            R2_flat_times_omega_average_element.cast<Expression>());
      }
    }
  }

  for (int interval = 0; interval < nT() - 1; ++interval) {
    get_mutable_prog()->AddLinearEqualityConstraint(
        R_WB()[interval + 1] - R_WB()[interval] -
            0.5 * dt_ * (R1_times_omega_average_hat[interval] +
                         R2_times_omega_average_hat[interval]),
        Eigen::Matrix3d::Zero());
  }
}

void QuasiDynamicObjectContactPlanning::AddQuasiDynamicConstraint() {
  for (int knot = 1; knot < nT(); ++knot) {
    const Vector6<Expression> total_wrench = TotalWrench(knot);
    get_mutable_prog()->AddLinearEqualityConstraint(
        mass() * (v_B_.col(knot) - v_B_.col(knot - 1)) -
            total_wrench.tail<3>() * dt_,
        Eigen::Vector3d::Zero());
    get_mutable_prog()->AddLinearEqualityConstraint(
        I_B_ * (omega_B_.col(knot) - omega_B_.col(knot - 1)) -
            total_wrench.head<3>() * dt_,
        Eigen::Vector3d::Zero());
  }
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
