#pragma once

#include "drake/multibody/rational_forward_kinematics/convex_geometry.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"

namespace drake {
namespace multibody {
// Given a box region in the configuration space q_lower <= q <= q_upper, we
// want to find the outer approximation of the sweeping volume in the workspace,
// namely nᵀp_WQ≤ d for all points Q on a link, and q_lower <= q <= q_upper
class SweepingVolumeOuterApproximation {
 public:
  /** We need to verify that these polynomials are non-negative
   * Lagrangians l_lower(t) >= 0, l_upper(t) >= 0
   * Link polynomial p(t) - l_lower(t) * (t - t_lower) - l_upper(t)(t_upper_t)
   * >= 0
   */
  struct VerificationOption {
    VerificationOption()
        : link_polynomial_type{solvers::MathematicalProgram::
                                   NonnegativePolynomial::kSos},
          lagrangian_type{
              solvers::MathematicalProgram::NonnegativePolynomial::kSos} {}
    solvers::MathematicalProgram::NonnegativePolynomial link_polynomial_type;
    solvers::MathematicalProgram::NonnegativePolynomial lagrangian_type;
  };
  /**
   * @param link_polytopes A polytopic representation of the link collision
   * geometry.
   * @param q_star The nominal configuration inside the box. We will write p_WQ
   * as a rational function of t, where t = tan((q - q_star) / 2).
   */
  SweepingVolumeOuterApproximation(
      const MultibodyPlant<double>& plant,
      const std::vector<std::shared_ptr<const ConvexPolytope>>& link_polytopes,
      const Eigen::Ref<const Eigen::VectorXd>& q_star);

  /**
   * Find an upper bound on the projection of the sweeping volume along the
   * direction n_W. Namely an upper bound on max nᵀp_WQ s.t q_lower <= q <=
   * q_upper.
   * The user is supposed to have clampped q_lower and q_upper within the joint
   * limits.
   */
  double FindSweepingVolumeMaximalProjection(
      ConvexGeometry::Id link_polytope_id,
      const Eigen::Ref<const Eigen::Vector3d>& n_W,
      const Eigen::Ref<const Eigen::VectorXd>& q_lower,
      const Eigen::Ref<const Eigen::VectorXd>& q_upper,
      const VerificationOption& verification_option = {}) const;

 private:
  RationalForwardKinematics rational_forward_kin_;
  const std::vector<std::shared_ptr<const ConvexPolytope>>& link_polytopes_;
  Eigen::VectorXd q_star_;
  // p_WV_poly_[i][j] is the position of the j'th vertex on the i'th
  // link_polytopes expressed in the world frame. The position is written as
  // a multilinear polynomial of sinθ and cosθ.
  std::vector<std::vector<Vector3<symbolic::Polynomial>>> p_WV_poly_;

  std::unordered_map<ConvexGeometry::Id, int> link_polytope_id_to_index_;
};

}  // namespace multibody
}  // namespace drake
