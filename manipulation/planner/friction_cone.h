#pragma once

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace manipulation {
namespace planner {
/**
 * A helper function to generate the edges of a linearized friction cone.
 * The edges would be evenly spaced in the cone, and the magnitude of each edge
 * is 1.
 */
template <int NumEdges>
Eigen::Matrix<double, 3, NumEdges> GenerateLinearizedFrictionConeEdges(
    const Eigen::Ref<const Eigen::Vector3d>& n, double mu) {
  // First generate a linearized friction cone, whose normal vector is the 
  // unit-z vector.
  // theta is the angle of the tangential vectors.
  const Eigen::Matrix<double, 1, NumEdges> theta =
      (Eigen::Matrix<double, 1, NumEdges + 1>::LinSpaced(0, 2 * M_PI))
          .template head<NumEdges>();
  Eigen::Matrix<double, 3, NumEdges> edges;
  edges.row(0) = (theta.array().cos() * mu).matrix();
  edges.row(1) = (theta.array().sin() * mu).matrix();
  edges.row(2) = Eigen::Matrix<double, 1, NumEdges>::Ones();

  edges /= std::sqrt(1 + mu * mu);

  // Now we need to rotate edges, such that the unit-z vector will be aligned
  // with the normal vector `n`.
  const Eigen::Vector3d n_normalized = n.normalized();
  // Now find a unit length vector v that is perpendicular to n_normalized.
  Eigen::Vector3d v = n_normalized.cross(Eigen::Vector3d::UnitX());
  double v_norm = v.norm();
  if (v_norm < 1E-2) {
    // n_normalized is almost parrallel to unit-x vector. So it must have a
    // large angle with unit-y vector.
    v = n_normalized.cross(Eigen::Vector3d::UnitY());
    v_norm = v.norm();
  }
  v /= v_norm;
  const Eigen::Vector3d u = n_normalized.cross(v);
  // The matrix R rotates unit-z vector to `n_normalized`.
  Eigen::Matrix3d R;
  R.col(2) = n_normalized;
  R.col(0) = v;
  R.col(1) = u;
  return R * edges;
}

/**
 * Adds friction cone constraint to contact force f_F, expressed in a frame F.
 * The friction cone has its unit length normal direction as n_F, expressed in
 * the same frame F, with a coefficient of friction being mu.
 */
void AddFrictionConeConstraint(
    double mu, const Eigen::Ref<const Eigen::Vector3d>& n_F,
    const Eigen::Ref<const Vector3<symbolic::Expression>>& f_F,
    solvers::MathematicalProgram* prog);

/**
 * Adds a linearized friction cone constraint. The linearized friction cone is
 * described by its edges `e_F`, expressed in a frame F. We will add the slack
 * variable `w` to the program, with the constraint
 * f_F = w(i) * e_F.col(i)
 * w(i) ≥ 0
 * @param e_F The edged of the friction cone.
 * @param f_F The force in the frame F.
 * @param prog The program to which the constraint and decision variables are
 * added.
 * @retval w The non-negative weights for each friction cone edge.
 */
solvers::VectorXDecisionVariable AddLinearizedFrictionConeConstraint(
    const Eigen::Ref<const Eigen::Matrix3Xd>& e_F,
    const Eigen::Ref<const solvers::VectorDecisionVariable<3>>& f_F,
    solvers::MathematicalProgram* prog);
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
