#include "drake/systems/analysis/robust_verification.h"

namespace drake {
namespace systems {
namespace analysis {
RobustInvariantSetVerfication::RobustInvariantSetVerfication(
    const System<symbolic::Expression>& system,
    const Eigen::Ref<const Eigen::MatrixXd>& K,
    const Eigen::Ref<const Eigen::VectorXd>& k0,
    const Eigen::Ref<const Eigen::MatrixXd>& x_err_vertices)
    : system_(&system), K_(K), k0_(k0), x_err_vertices_(x_err_vertices) {
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
  solvers::VectorXIndeterminate u(num_u);
  for (int i = 0; i < u.size(); ++i) {
    u(i) = symbolic::Variable("u" + std::to_string(i));
  }
  context->get_mutable_continuous_state().SetFromVector(
      x_.cast<symbolic::Expression>());

  int u_count = 0;
  for (int i = 0; i < system_->get_num_input_ports(); ++i) {
    context->FixInputPort(i,
                          u.segment(u_count, system_->get_input_port(i).size())
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
}  // namespace analysis
}  // namespace systems
}  // namespace drake
