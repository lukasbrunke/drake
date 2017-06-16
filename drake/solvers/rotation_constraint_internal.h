#pragma once

#include <vector>

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

Vector3<symbolic::Expression> CalcBoxBinaryExpressionInOrthant(
    int xi, int yi, int zi, int orthant,
    const Eigen::Ref<const Eigen::MatrixXi>& gray_codes,
    const std::array<VectorXDecisionVariable, 3>& B_vec,
    int num_intervals_per_half_axis);
}  // namespace internal
}  // namespace solvers
}  // namespace drake
