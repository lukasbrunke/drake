#include "drake/manipulation/planner/friction_cone.h"

using drake::symbolic::Expression;
using drake::solvers::MathematicalProgram;
using drake::solvers::VectorDecisionVariable;
using drake::solvers::VectorXDecisionVariable;

namespace drake {
namespace manipulation {
namespace planner {
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

VectorXDecisionVariable AddLinearizedFrictionConeConstraint(
    const Eigen::Ref<const Eigen::Matrix3Xd>& e_F,
    const Eigen::Ref<const VectorDecisionVariable<3>>& f_F,
    MathematicalProgram* prog) {
  const auto w =
      prog->NewContinuousVariables(e_F.cols(), "friction_cone_weights");
  prog->AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), w);
  prog->AddLinearEqualityConstraint(e_F * w - f_F, Eigen::Vector3d::Zero());
  return w;
}

}  // namespace planner
}  // namespace manipulation
}  // namespace drake
