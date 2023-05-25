#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "drake/common/drake_deprecated.h"
#include "drake/geometry/optimization/c_iris_collision_geometry.h"
#include "drake/geometry/optimization/c_iris_separating_plane.h"
#include "drake/geometry/optimization/cspace_free_structs.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/multibody/rational/rational_forward_kinematics.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace geometry {
namespace optimization {
/**
 This class tries to find large axis-algined bounding boxes in the robot
 configuration space, suthat that all configurations in the boxes are collision
 free.
 */
class CspaceFreeBox {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CspaceFreeBox)

  using IgnoredCollisionPairs =
      std::unordered_set<SortedPair<geometry::GeometryId>>;

  ~CspaceFreeBox() {}

  /** Optional argument for constructing CspaceFreeBox*/
  struct Options {
    Options() {}

    /**
     For non-polytopic collision geometries, we will impose a matrix-sos
     constraint X(s) being psd, with a slack indeterminates y, such that the
     polynomial
     <pre>
     p(s, y) = ⌈ 1 ⌉ᵀ * X(s) * ⌈ 1 ⌉
               ⌊ y ⌋           ⌊ y ⌋
     </pre>
     is positive. This p(s, y) polynomial doesn't contain the cross term of y
     (namely it doesn't have y(i)*y(j), i≠j). When we select the monomial
     basis for this polynomial, we can also exclude the cross term of y in the
     monomial basis.

     To illustrate the idea, let's consider the following toy example: if we
     want to certify that
     a(0) + a(1)*y₀ + a(2)*y₁ + a(3)*y₀² + a(4)*y₁² is positive
     (this polynomial doesn't have the cross term y₀*y₁), we can write it as
     <pre>
     ⌈ 1⌉ᵀ * A₀ * ⌈ 1⌉ + ⌈ 1⌉ᵀ * A₁ * ⌈ 1⌉
     ⌊y₀⌋         ⌊y₀⌋   ⌊y₁⌋         ⌊y₁⌋
     </pre>
     with two small psd matrices A₀, A₁
     Instead of
     <pre>
     ⌈ 1⌉ᵀ * A * ⌈ 1⌉
     |y₀|        |y₀|
     ⌊y₁⌋        ⌊y₁⌋
     </pre>
     with one large psd matrix A. The first parameterization won't have the
     cross term y₀*y₁ by construction, while the second parameterization
     requires imposing extra constraints on certain off-diagonal terms in A
     so that the cross term vanishes.

     If we set with_cross_y = false, then we will use the monomial basis that
     doesn't generate cross terms of y, leading to smaller size sos problems.
     If we set with_cross_y = true, then we will use the monomial basis that
     will generate cross terms of y, causing larger size sos problems, but
     possibly able to certify a larger C-space box.
     */
    bool with_cross_y{false};
  };

  /**
   @param plant The plant for which we compute the C-space free boxes. It
   must outlive this CspaceFreeBox object.
   @param scene_graph The scene graph that has been connected with `plant`. It
   must outlive this CspaceFreeBox object.
   @param plane_order The order of the polynomials in the plane to separate a
   pair of collision geometries.

   @note CspaceFreeBox knows nothing about contexts. The plant and
   scene_graph must be fully configured before instantiating this class.
   */
  CspaceFreeBox(const multibody::MultibodyPlant<double>* plant,
                const geometry::SceneGraph<double>* scene_graph,
                SeparatingPlaneOrder plane_order,
                const Options& options = Options{});

  [[nodiscard]] const multibody::RationalForwardKinematics&
  rational_forward_kin() const {
    return rational_forward_kin_;
  }

  /**
   separating_planes()[map_geometries_to_separating_planes.at(geometry1_id,
   geometry2_id)] is the separating plane that separates geometry1 and
   geometry 2.
   */
  [[nodiscard]] const std::unordered_map<SortedPair<geometry::GeometryId>, int>&
  map_geometries_to_separating_planes() const {
    return map_geometries_to_separating_planes_;
  }

  [[nodiscard]] const std::vector<CIrisSeparatingPlane<symbolic::Variable>>&
  separating_planes() const {
    return separating_planes_;
  }

  [[nodiscard]] const Vector3<symbolic::Variable>& y_slack() const {
    return y_slack_;
  }

  /**
   When searching for the separating plane, we want to certify that the
   numerator of a rational is non-negative in the C-space region
   q_box_lower <= q <= q_box_upper. Hence for each of the rational we will
   introduce Lagrangian multipliers for the constraint s - s_box_lower >= 0,
   s_box_upper - s >= 0.
   */
  class SeparatingPlaneLagrangians {
   public:
    SeparatingPlaneLagrangians(int s_size)
        : s_box_lower_(s_size), s_box_upper_(s_size) {}

    /** Substitutes the decision variables in each Lagrangians with its value in
     * result, returns the substitution result.
     */
    [[nodiscard]] SeparatingPlaneLagrangians GetSolution(
        const solvers::MathematicalProgramResult& result) const;

    /// The Lagrangians for s - s_box_lower >= 0.
    const VectorX<symbolic::Polynomial>& s_box_lower() const {
      return s_box_lower_;
    }

    /// The Lagrangians for s - s_box_lower >= 0.
    VectorX<symbolic::Polynomial>& mutable_s_box_lower() {
      return s_box_lower_;
    }

    /// The Lagrangians for s_box_upper - s >= 0.
    const VectorX<symbolic::Polynomial>& s_box_upper() const {
      return s_box_upper_;
    }

    /// The Lagrangians for box_upper - s >= 0.
    VectorX<symbolic::Polynomial>& mutable_s_box_upper() {
      return s_box_upper_;
    }

   private:
    // The Lagrangians for s - s_box_lower >= 0.
    VectorX<symbolic::Polynomial> s_box_lower_;
    // The Lagrangians for s_box_upper - s >= 0.
    VectorX<symbolic::Polynomial> s_box_upper_;
  };

  /**
   We certify that a pair of geometries is collision free in the C-space region
   {s | box_lower<=s<=box_upper} by finding the separating plane and the
   Lagrangian multipliers. This struct contains the certificate, that the
   separating plane {x | aᵀx+b=0 } separates the two geometries in
   separating_planes()[plane_index] in the C-space polytope.
   */
  struct SeparationCertificateResult : SeparationCertificateResultBase {
    std::vector<SeparatingPlaneLagrangians> positive_side_rational_lagrangians;
    std::vector<SeparatingPlaneLagrangians> negative_side_rational_lagrangians;
  };
};
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
