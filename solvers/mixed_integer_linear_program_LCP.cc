#include "drake/solvers/mixed_integer_linear_program_LCP.h"

namespace drake {
namespace solvers {
MixedIntegerLinearProgramLCP::MixedIntegerLinearProgramLCP(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::MatrixXd>& M,
    const Eigen::Ref<const Eigen::ArrayXd>& z_max,
    const Eigen::Ref<const Eigen::ArrayXd>& w_max)
    : n_(q.rows()),
      q_{q},
      M_{M},
      z_max_{z_max},
      w_max_{w_max},
      prog_{std::make_unique<MathematicalProgram>()},
      w_{prog_->NewContinuousVariables(n_, "w")},
      z_{prog_->NewContinuousVariables(n_, "z")},
      b_{prog_->NewBinaryVariables(n_, "b")} {
  DRAKE_DEMAND(M_.rows() == n_ && M_.cols() == n_);
  DRAKE_DEMAND(z_max_.rows() == n_);
  DRAKE_DEMAND(w_max_.rows() == n_);
  DRAKE_DEMAND((z_max_ > 0).all());
  DRAKE_DEMAND((w_max_ > 0).all());
  prog_->AddBoundingBoxConstraint(Eigen::VectorXd::Zero(n_), z_max_, z_);
  prog_->AddBoundingBoxConstraint(Eigen::VectorXd::Zero(n_), w_max_, w_);
  prog_->AddLinearEqualityConstraint(w_ - M_ * z_, q_);
  prog_->AddLinearConstraint(w_.array() <= w_max_ * b_.array());
  prog_->AddLinearConstraint(z_.array() <=
                             z_max_ * (Eigen::ArrayXd::Ones(n_) - b_.array()));
}
}  // namespace solvers
}  // namespace drake
