#pragma once

#include <vector>
#include "drake/common/symbolic_expression.h"
#include "drake/common/eigen_types.h"

// This file only exists to expose some internal methods for unit testing.  It
// should NOT be included in user code.
// The API documentation for these functions lives in rotation_constraint.cc,
// where they are implemented.
namespace drake {
namespace solvers {
namespace internal {

std::vector<Eigen::Vector3d> ComputeBoxEdgesAndSphereIntersection(
    const Eigen::Vector3d& bmin, const Eigen::Vector3d& bmax);

void ComputeHalfSpaceRelaxationForBoxSphereIntersection(
    const std::vector<Eigen::Vector3d>& pts, Eigen::Vector3d* n, double* d);

bool AreAllVerticesCoPlanar(const std::vector<Eigen::Vector3d>& pts,
                            Eigen::Vector3d* n, double* d);

void ComputeInnerFacetsForBoxSphereIntersection(
    const std::vector<Eigen::Vector3d>& pts,
    Eigen::Matrix<double, Eigen::Dynamic, 3>* A, Eigen::VectorXd* b);

template<typename Scalar1, typename Scalar2>
Vector3<Scalar1> CalcBoxBinaryExpressionInOrthant(
    int xi, int yi, int zi, int orthant,
    const Eigen::Ref<const Eigen::MatrixXi>& gray_codes,
    const std::array<VectorX<Scalar2>, 3>& B_vec,
    int num_intervals_per_half_axis);

// Given (an integer enumeration of) the orthant, takes a vector in the
// positive orthant into that orthant by flipping the signs of the individual
// elements.
template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1>
FlipVector(const Derived& vpos, int orthant) {
  DRAKE_ASSERT(vpos.rows() == 3 && vpos.cols() == 1);
  DRAKE_ASSERT(vpos(0) >= 0 && vpos(1) >= 0 && vpos(2) >= 0);
  DRAKE_DEMAND(orthant >= 0 && orthant <= 7);
  Eigen::Matrix<typename Derived::Scalar, 3, 1> v = vpos;
  if (orthant & (1 << 2)) v(0) = -v(0);
  if (orthant & (1 << 1)) v(1) = -v(1);
  if (orthant & 1) v(2) = -v(2);
  return v;
}
}  // namespace internal
}  // namespace solvers
}  // namespace drake
