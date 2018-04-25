#include "drake/solvers/mixed_integer_linear_program_LCP.h"

#include "drake/solvers/gurobi_solver.h"

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

void MixedIntegerLinearProgramLCP::PolishSolution(const Eigen::Ref<const Eigen::VectorXd>& b_val, Eigen::VectorXd* w_sol, Eigen::VectorXd* z_sol) const {
  //Eigen::MatrixXd A(2 * n_, 2 * n_);
  //Eigen::VectorXd b(2 * n_);
  //A.setZero();
  //b.setZero();
  //A.topLeftCorner(n_, n_) = Eigen::MatrixXd::Identity(n_, n_);
  //A.topRightCorner(n_, n_) = -M_;
  //b.head(n_) = q_;
  //for (int i = 0; i < n_; ++i) {
  //  if (b_val(i) < 0.5) {
  //    A(i + n_, i) = 1;
  //  } else {
  //    A(i + n_, i + n_) = 1;
  //  }
  //}

  //const Eigen::VectorXd wz = A.colPivHouseholderQr().solve(b);
  //*w_sol = wz.head(n_);
  //*z_sol = wz.tail(n_);
  MathematicalProgram prog;
  auto w = prog.NewContinuousVariables(n_);
  auto z = prog.NewContinuousVariables(n_);
  prog.AddLinearEqualityConstraint(w - M_ * z, q_);
  for (int i = 0; i < n_; ++i) {
    if (b_val(i) < 0.5) {
      prog.AddBoundingBoxConstraint(0, 0, w(i));
      prog.AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), z(i));
    } else {
      prog.AddBoundingBoxConstraint(0, 0, z(i));
      prog.AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), w(i));
    }
  }
  prog.SetSolverOption(GurobiSolver::id(), "FeasibilityTol", 1E-7);
  const auto result = prog.Solve();
  if (result != SolutionResult::kSolutionFound) {
    std::cerr << "Polishing failed.\n";
  }
  *w_sol = prog.GetSolution(w);
  *z_sol = prog.GetSolution(z);
}
}  // namespace solvers
}  // namespace drake
