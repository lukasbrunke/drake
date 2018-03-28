#include "drake/manipulation/planner/object_contact_planning.h"

#include "drake/solvers/mixed_integer_optimization_util.h"
#include "drake/solvers/rotation_constraint.h"

using drake::solvers::MathematicalProgram;
using drake::solvers::MatrixDecisionVariable;
using drake::solvers::MatrixXDecisionVariable;
using drake::solvers::VectorDecisionVariable;
using drake::solvers::VectorXDecisionVariable;
using drake::symbolic::Expression;

namespace drake {
namespace manipulation {
namespace planner {
ObjectContactPlanning::ObjectContactPlanning(
    int nT, double mass, const Eigen::Ref<const Eigen::Vector3d>& p_BC,
    const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV)
    : prog_{std::make_unique<MathematicalProgram>()},
      nT_{nT},
      mass_{mass},
      p_BC_{p_BC},
      p_BV_{p_BV},
      p_WB_{static_cast<size_t>(nT_)},
      R_WB_{static_cast<size_t>(nT_)},
      b_R_WB_{static_cast<size_t>(nT_)},
      contact_vertex_indices_{static_cast<size_t>(nT_)},
      f_BV_{static_cast<size_t>(nT_)},
      vertex_contact_flag_{static_cast<size_t>(nT_)} {
  for (int i = 0; i < nT_; ++i) {
    p_WB_[i] =
        prog_->NewContinuousVariables<3>("p_WB[" + std::to_string(i) + "]");
    R_WB_[i] = solvers::NewRotationMatrixVars(
        prog_.get(), "R_WB[" + std::to_string(i) + "]");

    // Add the mixed-integer constraint as an approximation to the SO(3)
    // constraint.
    std::tie(b_R_WB_[i], phi_R_WB_) =
        solvers::AddRotationMatrixBilinearMcCormickMilpConstraints<2>(
            prog_.get(), R_WB_[i]);

    // Adds the SOCP constraint to approximate SO(3) constraint, such as
    // R_WB_[i].col(j).squaredNorm() <= 1
    solvers::AddRotationMatrixOrthonormalSocpConstraint(prog_.get(), R_WB_[i]);
  }
}

void ObjectContactPlanning::SetContactVertexIndices(
    int knot, const std::vector<int>& indices, double big_M) {
  DRAKE_DEMAND(big_M >= 0);
  contact_vertex_indices_[knot] = indices;
  const int num_vertices = static_cast<int>(indices.size());
  // Add the contact force variables at each vertex.
  f_BV_[knot] = prog_->NewContinuousVariables<3, Eigen::Dynamic>(
      3, num_vertices, "f_BV_[" + std::to_string(knot) + "]");
  vertex_contact_flag_[knot] = prog_->NewBinaryVariables(
      num_vertices, "vertex_contact_flag_[" + std::to_string(knot) + "]");

  // Add the big-M constraint, that -M * flag <= f_x <= M * flag, same for f_y
  // and f_z.
  for (int i = 0; i < 3; ++i) {
    prog_->AddLinearConstraint(f_BV_[knot].row(i) <=
                               big_M * vertex_contact_flag_[knot]);
    prog_->AddLinearConstraint(f_BV_[knot].row(i) >=
                               -big_M * vertex_contact_flag_[knot]);
  }
}

std::array<VectorXDecisionVariable, 3>
ObjectContactPlanning::CalcContactForceInWorldFrame(
    const Eigen::Ref<const VectorDecisionVariable<3>>& f_B,
    const Eigen::Ref<const VectorDecisionVariable<3>>& f_W, int knot,
    bool binning_f_B, const std::array<Eigen::VectorXd, 3>& phi_f) {
  std::array<VectorXDecisionVariable, 3> b_f;

  std::array<VectorXDecisionVariable, 3> lambda_f;
  const std::string lambda_name = binning_f_B ? "lambda_f_B" : "lambda_f_W";
  const std::string b_f_name = binning_f_B ? "b_f_B" : "b_f_W";
  for (int i = 0; i < 3; ++i) {
    lambda_f[i] = prog_->NewContinuousVariables(phi_f[i].rows(), lambda_name);
    b_f[i] = solvers::AddLogarithmicSos2Constraint(
        prog_.get(), lambda_f[i].cast<Expression>(), b_f_name);
  }

  // If binning_f_B = true, then R_times_f(i, j) is an approximation of
  // R_WB_[knot](i, j) * f_B(j).
  // otherwise, it is an approximatio R_WB_[knot](j, i) * f_W(j)
  const std::string R_times_f_name =
      binning_f_B ? "R_WB_times_f_B" : "R_WB_times_f_W";
  const auto R_times_f = prog_->NewContinuousVariables<3, 3>(R_times_f_name);
  std::array<std::array<MatrixXDecisionVariable, 3>, 3> lambda_R_times_f;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (binning_f_B) {
        lambda_R_times_f[i][j] =
            solvers::AddBilinearProductMcCormickEnvelopeSos2(
                prog_.get(), R_WB_[knot](i, j), f_B(j), R_times_f(i, j),
                phi_R_WB_, phi_f[j], b_R_WB_[knot][i][j].cast<Expression>(),
                b_f[j].cast<Expression>(),
                solvers::IntervalBinning::kLogarithmic);
      } else {
        lambda_R_times_f[i][j] =
            solvers::AddBilinearProductMcCormickEnvelopeSos2(
                prog_.get(), R_WB_[knot](j, i), f_W(j), R_times_f(i, j),
                phi_R_WB_, phi_f[j], b_R_WB_[knot][j][i].cast<Expression>(),
                b_f[j].cast<Expression>(),
                solvers::IntervalBinning::kLogarithmic);
      }
    }
  }

  if (binning_f_B) {
    // Now compute f_W(i) as ∑ⱼ R_WB_[knot](i, j) * f_B(j)
    prog_->AddLinearEqualityConstraint(
        f_W - R_times_f.cast<Expression>().rowwise().sum(),
        Eigen::VectorXd::Zero(f_W.rows()));
  } else {
    // Now compute f_B as R_WB_[knot]ᵀ * f_W
    prog_->AddLinearEqualityConstraint(
        f_B - R_times_f.cast<Expression>().rowwise().sum(),
        Eigen::VectorXd::Zero(f_B.rows()));
  }

  // for (int i = 0; i < 3; ++i) {
  //   for (int j = 0; j < 3; ++j) {
  //       // Now add the constraint that lambda_R_times_f[i, j].colwise.sum()
  //       ==
  //       // lambda_f[j]
  //       prog_->AddLinearEqualityConstraint(
  //           lambda_R_times_f[i][j].cast<Expression>().colwise().sum().transpose()
  //           -
  //               lambda_f[j],
  //           Eigen::VectorXd::Zero(lambda_f[j].rows()));
  //   }
  // }

  return b_f;
}

void ObjectContactPlanning::AddStaticEquilibriumConstraint() {
  // Write the static equilibrium constraint in the body frame.
  const Eigen::Vector3d mg(0, 0, -mass_ * kGravity);
  for (int knot = 0; knot < nT_; ++knot) {
    const Vector3<Expression> mg_B = R_WB_[knot].transpose() * mg;
    Vector3<Expression> total_force = mg_B;
    Vector3<Expression> total_torque = p_BC_.cross(mg_B);

    for (int i = 0; i < static_cast<int>(contact_vertex_indices_[knot].size());
         ++i) {
      total_force += f_BV_[knot].col(i);
      total_torque +=
          p_BV_.col(contact_vertex_indices_[knot][i]).cross(f_BV_[knot].col(i));
    }

    prog_->AddLinearConstraint(total_force == Eigen::Vector3d::Zero());
    prog_->AddLinearConstraint(total_torque == Eigen::Vector3d::Zero());
  }
}

void AddFrictionConeConstraint(double mu,
                               const Eigen::Ref<const Eigen::Vector3d>& n_F,
                               const Eigen::Ref<const Vector3<Expression>>& f_F,
                               MathematicalProgram* prog) {
  const Eigen::Vector3d n_F_normalized = n_F.normalized();
  // Find two vectors orthogonal to n_F as v1 and v2.
  Eigen::Vector3d v1 = n_F_normalized.cross(Eigen::Vector3d::UnitX());
  if (v1.norm() < 1E-2) {
    v1 = n_F_normalized.cross(Eigen::Vector3d::UnitY());
  }
  v1.normalized();
  const Eigen::Vector3d v2 = n_F_normalized.cross(v1);
  Vector3<Expression> lorentz_cone_expression;
  lorentz_cone_expression << mu * n_F_normalized.dot(f_F), v1.dot(f_F),
      v2.dot(f_F);
  prog->AddLorentzConeConstraint(lorentz_cone_expression);
}

}  // namespace planner
}  // namespace manipulation
}  // namespace drake
