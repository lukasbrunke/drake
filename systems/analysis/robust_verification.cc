#include "drake/systems/analysis/robust_verification.h"

#include "drake/common/eigen_types.h"

namespace drake {
namespace systems {
namespace analysis {
RobustInvariantSetVerfication::RobustInvariantSetVerfication(
    const System<symbolic::Expression>& system,
    const Eigen::Ref<const Eigen::MatrixXd>& K,
    const Eigen::Ref<const Eigen::VectorXd>& k0,
    const Eigen::Ref<const Eigen::MatrixXd>& x_err_vertices, int l_degree)
    : system_(&system),
      K_(K),
      k0_(k0),
      x_err_vertices_(x_err_vertices),
      l_polynomials_(x_err_vertices_.cols()),
      l_polynomials_coeffs_(x_err_vertices_.cols()) {
  // Check if the system has only continuous states.
  auto context = system_->AllocateContext();
  DRAKE_ASSERT(context->get_num_abstract_states() == 0);
  DRAKE_ASSERT(context->get_discrete_state().size() == 0);
  const int num_x = context->get_continuous_state().size();
  const int num_u = system_->get_num_total_inputs();
  DRAKE_ASSERT(K.rows() == num_u);
  DRAKE_ASSERT(K.cols() == num_x);
  DRAKE_ASSERT(k0.rows() == num_u);
  DRAKE_ASSERT(x_err_vertices.rows() == num_x);
  x_.resize(num_x);
  for (int i = 0; i < x_.size(); ++i) {
    x_(i) = symbolic::Variable("x" + std::to_string(i));
  }

  // Check if the system is control affine.
  {
    solvers::VectorXIndeterminate u(num_u);
    for (int i = 0; i < u.size(); ++i) {
      u(i) = symbolic::Variable("u" + std::to_string(i));
    }
    context->get_mutable_continuous_state().SetFromVector(
        x_.cast<symbolic::Expression>());

    int u_count = 0;
    for (int i = 0; i < system_->get_num_input_ports(); ++i) {
      context->FixInputPort(
          i, u.segment(u_count, system_->get_input_port(i).size())
                 .cast<symbolic::Expression>());
      u_count += system_->get_input_port(i).size();
    }

    auto derivatives = system_->AllocateTimeDerivatives();
    system_->CalcTimeDerivatives(*context, derivatives.get());

    symbolic::Variables u_set;
    for (int i = 0; i < num_u; ++i) {
      u_set.insert(u(i));
    }
    for (int i = 0; i < derivatives->size(); ++i) {
      symbolic::Polynomial xdot_poly_i((*derivatives)[i], u_set);
      if (xdot_poly_i.TotalDegree() > 1) {
        throw std::logic_error("The system is not control affine.");
      }
    }
  }

  // Constructs Lagrangian polynomials
  for (int i = 0; i < x_.size(); ++i) {
    x_set_.insert(x_(i));
  }
  VectorX<symbolic::Monomial> x_monomials{
      symbolic::MonomialBasis(x_set_, l_degree)};

  for (int i = 0; i < x_err_vertices_.cols(); ++i) {
    l_polynomials_coeffs_[i].resize(x_monomials.size());
    for (int j = 0; j < x_monomials.size(); ++j) {
      l_polynomials_coeffs_[i](j) = symbolic::Variable(
          "l_coeffs[" + std::to_string(i) + "](" + std::to_string(j) + ")");
      l_polynomials_[i].AddProduct(l_polynomials_coeffs_[i](j), x_monomials(j));
    }
  }
}

void RobustInvariantSetVerfication::ConstructSymbolicEnvironment(
    const Eigen::Ref<
        const Eigen::Matrix<symbolic::Variable, Eigen::Dynamic, 1>>& x_var,
    const Eigen::Ref<const Eigen::VectorXd>& x_val,
    symbolic::Environment* env) const {
  DRAKE_ASSERT(x_var.size() == x_val.size());
  DRAKE_ASSERT(env);
  DRAKE_ASSERT(env->empty());
  for (int i = 0; i < x_var.size(); ++i) {
    env->insert(x_var(i), x_val(i));
  }
}

void RobustInvariantSetVerfication::CalcVdot(
    const symbolic::Polynomial& V,
    const Eigen::Ref<const Eigen::VectorXd>& x_val, int taylor_order,
    VectorX<symbolic::Polynomial>* Vdot) const {
  auto context = system_->AllocateContext();
  context->get_mutable_continuous_state().SetFromVector(
      x_.cast<symbolic::Expression>());
  const Eigen::Matrix<symbolic::Polynomial, 1, Eigen::Dynamic> dVdx =
      V.Jacobian(x_);
  symbolic::Environment env;
  ConstructSymbolicEnvironment(x_, x_val, &env);
  for (int i = 0; i < x_err_vertices_.cols(); ++i) {
    const VectorX<symbolic::Expression> u_feedback =
        K_ * (x_ + x_err_vertices_.col(i)) + k0_;
    int u_count = 0;
    for (int j = 0; j < system_->get_num_input_ports(); ++j) {
      context->FixInputPort(
          j, u_feedback.segment(u_count, system_->get_input_port(j).size()));
      u_count += system_->get_input_port(j).size();
    }
    auto derivatives = system_->AllocateTimeDerivatives();
    system_->CalcTimeDerivatives(*context, derivatives.get());
    const VectorX<symbolic::Expression> xdot =
        derivatives->get_vector().CopyToVector();
    // Do a taylor expansion of xdot
    VectorX<symbolic::Expression> xdot_taylor(xdot.size());
    VectorX<symbolic::Polynomial> xdot_taylor_poly(xdot.size());
    for (int j = 0; j < xdot.size(); ++j) {
      xdot_taylor(j) = TaylorExpand(xdot(j), env, taylor_order);
      xdot_taylor_poly(j) = symbolic::Polynomial(xdot_taylor(j), x_set_);
    }
    (*Vdot)(i) = (dVdx * xdot_taylor_poly)(0);
  }
}

std::unique_ptr<solvers::MathematicalProgram>
RobustInvariantSetVerfication::ConstructLagrangianStep(
    const symbolic::Polynomial& V,
    const Eigen::Ref<const Eigen::VectorXd>& x_val, int taylor_order,
    double rho_value) const {
  solvers::MathematicalProgram* prog = new solvers::MathematicalProgram();
  prog->AddIndeterminates(x_);
  for (const auto& l_poly_coeffs : l_polynomials_coeffs_) {
    prog->AddDecisionVariables(l_poly_coeffs);
  }
  VectorX<symbolic::Polynomial> Vdot(x_err_vertices_.cols());
  CalcVdot(V, x_val, taylor_order, &Vdot);
  auto eps_var = prog->NewContinuousVariables<1>("eps")(0);
  const symbolic::Polynomial eps_times_x_norm_squared(
      eps_var * x_.cast<symbolic::Expression>().dot(x_), x_set_);
  for (int i = 0; i < Vdot.size(); ++i) {
    prog->AddSosConstraint(-Vdot(i) - l_polynomials_[i] * (rho_value - V) -
                           eps_times_x_norm_squared);
  }
  prog->AddCost(-eps_var);
  return std::unique_ptr<solvers::MathematicalProgram>(prog);
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
