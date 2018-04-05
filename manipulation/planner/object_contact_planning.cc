#include "drake/manipulation/planner/object_contact_planning.h"

#include "drake/manipulation/planner/friction_cone.h"
#include "drake/solvers/integer_optimization_util.h"
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
    const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV, int num_pushers,
    const std::vector<BodyContactPoint>& Q, bool add_second_order_cone_for_R)
    : prog_{std::make_unique<MathematicalProgram>()},
      nT_{nT},
      mass_{mass},
      p_BC_{p_BC},
      p_BV_{p_BV},
      num_pushers_{num_pushers},
      Q_{Q},
      p_WB_{static_cast<size_t>(nT_)},
      R_WB_{static_cast<size_t>(nT_)},
      b_R_WB_{static_cast<size_t>(nT_)},
      contact_vertex_indices_{static_cast<size_t>(nT_)},
      vertex_to_V_map_{static_cast<size_t>(nT_)},
      f_BV_{static_cast<size_t>(nT_)},
      vertex_contact_flag_{static_cast<size_t>(nT_)},
      contact_Q_indices_{static_cast<size_t>(nT_)},
      Q_to_index_map_{static_cast<size_t>(nT_)},
      b_Q_contact_{static_cast<size_t>(nT_)},
      f_BQ_{static_cast<size_t>(nT_)} {
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

    if (add_second_order_cone_for_R) {
      // Adds the SOCP constraint to approximate SO(3) constraint, such as
      // R_WB_[i].col(j).squaredNorm() <= 1
      solvers::AddRotationMatrixOrthonormalSocpConstraint(prog_.get(),
                                                          R_WB_[i]);
    }
  }
}

void ObjectContactPlanning::SetContactVertexIndices(
    int knot, const std::vector<int>& indices, double big_M) {
  DRAKE_DEMAND(big_M >= 0);
  contact_vertex_indices_[knot] = indices;
  const int num_vertices = static_cast<int>(indices.size());
  vertex_to_V_map_[knot].reserve(num_vertices);
  for (int i = 0; i < num_vertices; ++i) {
    vertex_to_V_map_[knot].emplace(indices[i], i);
  }
  // Add the contact force variables at each vertex.
  f_BV_[knot] = prog_->NewContinuousVariables<3, Eigen::Dynamic>(
      3, num_vertices, "f_BV_[" + std::to_string(knot) + "]");
  vertex_contact_flag_[knot] = prog_->NewBinaryVariables(
      num_vertices, "vertex_contact_flag_[" + std::to_string(knot) + "]");

  // Add the big-M constraint, that -M * flag <= f_x <= M * flag, same for f_y
  // and f_z.
  for (int i = 0; i < 3; ++i) {
    prog_->AddLinearConstraint(f_BV_[knot].row(i) <=
                               big_M * vertex_contact_flag_[knot].transpose());
    prog_->AddLinearConstraint(f_BV_[knot].row(i) >=
                               -big_M * vertex_contact_flag_[knot].transpose());
  }
}

void ObjectContactPlanning::SetPusherContactPointIndices(
    int knot, const std::vector<int>& indices, double big_M) {
  DRAKE_DEMAND(big_M >= 0);
  contact_Q_indices_[knot] = indices;
  Q_to_index_map_[knot].reserve(indices.size());
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    Q_to_index_map_[knot].emplace(indices[i], i);
  }
  const int num_Q = static_cast<int>(indices.size());
  // Add contact force at Q
  f_BQ_[knot] = prog_->NewContinuousVariables<3, Eigen::Dynamic>(
      3, num_Q, "f_BQ[" + std::to_string(knot) + "]");
  b_Q_contact_[knot] = prog_->NewBinaryVariables(
      num_Q, "b_Q_contact_[" + std::to_string(knot) + "]");

  for (int i = 0; i < num_Q; ++i) {
    const auto& Q = Q_[indices[i]];
    // Adds friction cone constraint
    const auto w_edges = AddLinearizedFrictionConeConstraint(
        Q.e_B(), f_BQ_[knot].col(i), prog_.get());

    // Now add the big M constraint ∑ᵢ wᵢ ≤ M * b
    prog_->AddLinearConstraint(w_edges.cast<Expression>().sum() <=
                               big_M * b_Q_contact_[knot](i));
  }

  // Adds the constraint that at most num_pusher_ points can be active.
  prog_->AddLinearConstraint(Eigen::RowVectorXd::Ones(num_Q), 0, num_pushers_,
                             b_Q_contact_[knot]);
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
  //  for (int j = 0; j < 3; ++j) {
  //      // Now add the constraint that lambda_R_times_f[i, j].colwise.sum() ==
  //      // lambda_f[j]
  //      prog_->AddLinearEqualityConstraint(
  //          lambda_R_times_f[i][j].cast<Expression>().colwise().sum().transpose()
  //          -
  //              lambda_f[j],
  //          Eigen::VectorXd::Zero(lambda_f[j].rows()));
  //  }
  //}

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

    for (int i = 0; i < static_cast<int>(contact_Q_indices_[knot].size());
         ++i) {
      total_force += f_BQ_[knot].col(i);
      total_torque +=
          Q_[contact_Q_indices_[knot][i]].p_BQ().cross(f_BQ_[knot].col(i));
    }

    prog_->AddLinearConstraint(total_force == Eigen::Vector3d::Zero());
    prog_->AddLinearConstraint(total_torque == Eigen::Vector3d::Zero());
  }
}

optional<symbolic::Variable>
ObjectContactPlanning::AddVertexNonSlidingConstraint(
    int interval, int vertex_index,
    const Eigen::Ref<const Eigen::Vector3d>& x_W,
    const Eigen::Ref<const Eigen::Vector3d>& y_W, double distance_big_M) {
  const int knot0 = interval;
  const int knot1 = interval + 1;
  const auto it0 = vertex_to_V_map_[knot0].find(vertex_index);
  const auto it1 = vertex_to_V_map_[knot1].find(vertex_index);
  optional<symbolic::Variable> ret_variable;
  if (it0 != vertex_to_V_map_[knot0].end() &&
      it1 != vertex_to_V_map_[knot1].end()) {
    // The requested vertex is a candidate contact vertex at both the beginning
    // and the ending knots of the interval.
    const Vector3<Expression> p_WV0 = p_WV(knot0, vertex_index);
    const Vector3<Expression> p_WV1 = p_WV(knot1, vertex_index);
    const auto z = prog_->NewContinuousVariables<1>(
        "V[" + std::to_string(vertex_index) + "]_in_contact_at_knot_" +
        std::to_string(knot0) + "_and_" + std::to_string(knot1))(0);
    prog_->AddConstraint(solvers::CreateLogicalAndConstraint(
        Expression(vertex_contact_flag_[knot0](it0->second)),
        Expression(vertex_contact_flag_[knot1](it1->second)), Expression(z)));
    const Vector3<Expression> delta_p_WV = p_WV0 - p_WV1;
    const Expression delta_p_WV_x = x_W.dot(delta_p_WV);
    const Expression delta_p_WV_y = y_W.dot(delta_p_WV);
    prog_->AddLinearConstraint(delta_p_WV_x <= distance_big_M * (1 - z));
    prog_->AddLinearConstraint(-delta_p_WV_x <= distance_big_M * (1 - z));
    prog_->AddLinearConstraint(delta_p_WV_y <= distance_big_M * (1 - z));
    prog_->AddLinearConstraint(-delta_p_WV_y <= distance_big_M * (1 - z));
    ret_variable.emplace(z);
  }
  return ret_variable;
}

void ObjectContactPlanning::AddPusherStaticContactConstraint(int interval) {
  const int knot0 = interval;
  const int knot1 = interval + 1;
  const auto b = prog_->NewContinuousVariables<2>(
      "b_make_break_contact[" + std::to_string(interval) + "]");
  for (int i = 0; i < static_cast<int>(Q_.size()); ++i) {
    const auto it0 = Q_to_index_map_[knot0].find(i);
    const auto it1 = Q_to_index_map_[knot1].find(i);
    const bool is_candidate_contact0 = it0 != Q_to_index_map_[knot0].end();
    const bool is_candidate_contact1 = it1 != Q_to_index_map_[knot1].end();
    if (is_candidate_contact0 && is_candidate_contact1) {
      prog_->AddLinearConstraint(b(0) >= b_Q_contact_[knot1](it1->second) -
                                             b_Q_contact_[knot0](it0->second));
      prog_->AddLinearConstraint(b(1) >= b_Q_contact_[knot0](it0->second) -
                                             b_Q_contact_[knot1](it1->second));
    } else if (is_candidate_contact0 && !is_candidate_contact1) {
      prog_->AddLinearConstraint(b(1) >= b_Q_contact_[knot0](it0->second));
    } else if (!is_candidate_contact0 && is_candidate_contact1) {
      prog_->AddLinearConstraint(b(0) >= b_Q_contact_[knot1](it1->second));
    }
  }
  prog_->AddLinearConstraint(Eigen::RowVector2d::Ones(), 0, 1, b);
}

void ObjectContactPlanning::AddAtMostOnePusherChangeOfContactConstraint(
    int interval) {
  DRAKE_ASSERT(interval < nT_ - 1);
  const int knot0 = interval;
  const int knot1 = interval + 1;
  // We will add the constraint
  // sum_{point_index} | b[knot0](point_index) -b[knot1](point_index) | ≤ 1
  // The left-hand side is represented by sum_contact_difference
  symbolic::Expression sum_contact_difference = 0;
  for (int i = 0; i < static_cast<int>(Q_.size()); ++i) {
    // For each point in Q_, find its index in contact_Q_indices_[knot] and
    // contact_Q_indices_[knot + 1]
    const auto it0 = Q_to_index_map_[knot0].find(i);
    const auto it1 = Q_to_index_map_[knot1].find(i);

    const bool is_candidate_contact0 = it0 != Q_to_index_map_[knot0].end();
    const bool is_candidate_contact1 = it1 != Q_to_index_map_[knot1].end();
    if (!is_candidate_contact0 && !is_candidate_contact1) {
      // Q_[i] is not a candidate pusher contact point at either knot0 or
      // knot1
      continue;
    } else if (!is_candidate_contact0 && is_candidate_contact1) {
      // Q_[i] is a candidate pusher contact point at knot1, but not knot0.
      sum_contact_difference += b_Q_contact_[knot1](it1->second);
    } else if (is_candidate_contact0 && !is_candidate_contact1) {
      // Q_[i] is a candidate pusher contact point at knot0, but not knot1.
      sum_contact_difference += b_Q_contact_[knot0](it0->second);
    } else {
      // Q_[i] is a candidate pusher contact point at both knot0 and knot1.
      // Introduce a slack variable b_Q_difference to represent the absolute
      // difference | b_Q_contact[knot0](pusher) - b_Q_contact[knot1](pusher)|
      auto b_Q_difference =
          prog_->NewContinuousVariables<1>("b_Q_difference")(0);
      prog_->AddLinearConstraint(b_Q_contact_[knot0](it0->second) -
                                     b_Q_contact_[knot1](it1->second) <=
                                 b_Q_difference);
      prog_->AddLinearConstraint(b_Q_contact_[knot1](it1->second) -
                                     b_Q_contact_[knot0](it0->second) <=
                                 b_Q_difference);
      sum_contact_difference += b_Q_difference;
    }
  }
  prog_->AddLinearConstraint(sum_contact_difference <= 1);
}

solvers::Binding<solvers::LorentzConeConstraint>
ObjectContactPlanning::AddOrientationDifferenceUpperBound(
    int interval, double max_angle_difference) {
  DRAKE_ASSERT(max_angle_difference >= 0 && max_angle_difference <= M_PI);
  Eigen::Matrix<Expression, 10, 1> lorentz_cone_expressions;
  const int knot0 = interval;
  const int knot1 = interval + 1;
  lorentz_cone_expressions << 2 * std::sqrt(2) * sin(max_angle_difference / 2),
      R_WB_[knot0].col(0) - R_WB_[knot1].col(0),
      R_WB_[knot0].col(1) - R_WB_[knot1].col(1),
      R_WB_[knot0].col(2) - R_WB_[knot1].col(2);
  return prog_->AddLorentzConeConstraint(lorentz_cone_expressions);
}

void ObjectContactPlanning::
    AddOrientationDifferenceUpperBoundLinearApproximation(
        int interval, double max_angle_difference) {
  DRAKE_ASSERT(max_angle_difference >= 0 && max_angle_difference <= M_PI);
  const int knot0 = interval;
  const int knot1 = interval + 1;
  const auto R1_minus_R2_abs = prog_->NewContinuousVariables<3, 3>(
      "R_WB[" + std::to_string(knot0) + "]_minus_R_WB[" +
      std::to_string(knot1) + "]_abs");
  prog_->AddLinearConstraint((R_WB_[knot0] - R_WB_[knot1]).array() <=
                             R1_minus_R2_abs.array());
  prog_->AddLinearConstraint((R_WB_[knot1] - R_WB_[knot0]).array() <=
                             R1_minus_R2_abs.array());
  const double two_sqrt2_sin_theta_over_2 =
      2 * std::sqrt(2) * sin(max_angle_difference / 2);
  VectorDecisionVariable<9> R1_minus_R2_abs_flat;
  R1_minus_R2_abs_flat << R1_minus_R2_abs.col(0), R1_minus_R2_abs.col(1),
      R1_minus_R2_abs.col(2);
  prog_->AddLinearConstraint(Eigen::Matrix<double, 1, 9>::Ones(), 0,
                             3 * two_sqrt2_sin_theta_over_2,
                             R1_minus_R2_abs_flat);
  prog_->AddBoundingBoxConstraint(0, two_sqrt2_sin_theta_over_2,
                                  R1_minus_R2_abs_flat);
}

void ObjectContactPlanning::
    AddOrientationDifferenceUpperBoundBilinearApproximation(
        int interval, double max_angle_difference) {
  const int knot0 = interval;
  const int knot1 = interval + 1;
  // Introduce a slack variable R1_times_R2, such that
  // R1_times_R2(i, j) approximates R1(i, j) * R2(i, j)
  const auto R1_times_R2 = prog_->NewContinuousVariables<3, 3>(
      "R[" + std::to_string(knot0) + "]_times_R[" + std::to_string(knot1) +
      "]");
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      AddBilinearProductMcCormickEnvelopeSos2(
          prog_.get(), R_WB()[knot0](i, j), R_WB()[knot1](i, j),
          R1_times_R2(i, j), phi_R_WB_, phi_R_WB_,
          b_R_WB_[knot0][i][j].cast<Expression>(),
          b_R_WB_[knot1][i][j].cast<Expression>(),
          solvers::IntervalBinning::kLogarithmic);
    }
  }
  prog_->AddLinearConstraint(
      Eigen::Matrix<double, 1, 9>::Ones(),
      1 + 2 * std::cos(max_angle_difference), 3,
      {R1_times_R2.col(0), R1_times_R2.col(1), R1_times_R2.col(2)});
}

Vector3<symbolic::Expression> ObjectContactPlanning::p_WV(
    int knot, int vertex_index) const {
  return p_WB_[knot] + R_WB_[knot] * p_BV_.col(vertex_index);
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
