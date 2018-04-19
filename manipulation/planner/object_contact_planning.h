#pragma once

#include <memory>
#include <unordered_map>

#include "drake/common/drake_optional.h"
#include "drake/manipulation/planner/body_contact_point.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace manipulation {
namespace planner {
const double kGravity = 9.81;

class ObjectContactPlanning {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ObjectContactPlanning)

  /**
   * @param nT The number of knots.
   * @param mass The mass of the object.
   * @param p_BC The position of the center of mass C, in the object body frame
   * B.
   * @param p_BV The position of the vertices of the object, in the object body
   * frame B.
   * @param num_pushers The total number of pushers.
   * @param Q The candidate pusher contact location Q
   * @param add_second_order_cone_for_R Set to true if we will add second order
   * cone constraint to the rotation matrix R_WB, representing the orientation
   * of the object.
   */
  ObjectContactPlanning(int nT, double mass,
                        const Eigen::Ref<const Eigen::Vector3d>& p_BC,
                        const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV,
                        int num_pushers, const std::vector<BodyContactPoint>& Q,
                        bool add_second_order_cone_for_R = false);

  ~ObjectContactPlanning() = default;

  /** Sets the indices of all possible contact vertices at a knot.
   * For each vertex in the body indexed by `indices`, we will add binary
   * variable b to indicate whether the vertex is in contact with the
   * environment, and the contact force vector f at the vertex, expressed in the
   * object frame. We will add the constraint
   * -M * b <= f_x <= M * b
   * -M * b <= f_y <= M * b
   * -M * b <= f_z <= M * b
   * where M is a big number, so that the binary variable b will determine
   * whether the contact force is zero 0 not.
   * @note The user should call this function for just once, at a knot point.
   */
  void SetContactVertexIndices(int knot, const std::vector<int>& indices,
                               double big_M);

  /** Sets the indices of all possible pusher contact points at a knot.
   * For each point Q[indices[i]], we will add a binary variable b to indicate
   * whether the point is in contact with a pusher, and also add a contact
   * force vector f at the point. We will further add the linearized friction
   * cone constraint
   * f = ∑ᵢ  wᵢ * eᵢ
   * wᵢ ≥ 0
   * where eᵢ is the i'th edge of the friction cone, with unit length.
   * together with a big-M constraint to activate/deactivate the contact force
   * using a binary variable b
   * ∑ᵢ wᵢ ≤ M * b
   * @note The user should call this function for just once, at a knot point.
   */
  void SetPusherContactPointIndices(int knot, const std::vector<int>& indices,
                                    double big_M);
  /**
   * Add the variable f_W, for the contact force f expressed in the world frame
   * W.
   * The force in world frame and the body frame satisfy the constraint
   * f_W = R_WB_[knot] * f_B       (1)
   * or equivalently
   * f_B = R_WB_[knot]ᵀ * f_W      (1)
   * The right-hand side of (1) (2) are both nonconvex and bilinear, we will use
   * McCormick Envelope to approximate the bilinear constraint.
   * @param f_B The variables representing the contact force in the body frame
   * B.
   * @param f_W The variables representing the contact force in the world frame
   * W.
   * @param knot The knot point.
   * @param binning_f_B True if we will add binning for f_B, and impose
   * constraint (1); false if we will add binning for f_W, and impose constraint
   * (2).
   * @param phi_f. The binning for the sos2 constraint on f. Refer to
   * solvers::AddBilinearProductMcCormickEnvelopeSos2 for more details.
   * @return b_f b_f[i] are the binary variables to determine which interval
   * f_B(i) or f_W(i) is in.
   */
  std::array<solvers::VectorXDecisionVariable, 3> CalcContactForceInWorldFrame(
      const Eigen::Ref<const solvers::VectorDecisionVariable<3>>& f_B,
      const Eigen::Ref<const solvers::VectorDecisionVariable<3>>& f_W, int knot,
      bool binning_f_B, const std::array<Eigen::VectorXd, 3>& phi_f);

  /**
   * Adds static equilibrium constraint at a knot.
   */
  void AddStaticEquilibriumConstraintAtKnot(int knot);

  /**
   * A vertex in contact with the world cannot slide within an interval.
   * Namely if the vertex contact is active at both the beginning and the end
   * of the interval, then its position in the world frame does not change.
   * Mathematically, the constraint is
   *  -M * (1 - z) <=
   *  p_WV_x[interval + 1](vertex_index) - p_WV_x[interval](vertex_index)
   *   <= M * (1 - z)
   * -M * (1 - z) <=
   *  p_WV_y[interval + 1](vertex_index) - p_WV_y[interval](vertex_index)
   *   <= M * (1 - z)
   * where `z` is a slack variable, that represents if the vertex is in contact
   * at both knot and knot + 1. The slack variable `z` satisfies the constraint
   * z == vertex_contact_flag[interval](point_index) &&
   * vertex_contact_flag[interval + 1](point_index)
   * @param interval The interval in which the non-sliding constraint is
   * enforced.
   * @param vertex_index p_BV()[vertex_index] will be constrained not to slide.
   * @param x_W The unit length tangential x vector, expressed in the world
   * frame W.
   * @param y_W The unit length tangential y vector, expressed in the world
   * frame W.
   * @param distance_big_M The big M constant used to enforce the non-sliding
   * constraint.
   */
  optional<symbolic::Variable> AddVertexNonSlidingConstraint(
      int interval, int vertex_index,
      const Eigen::Ref<const Eigen::Vector3d>& x_W,
      const Eigen::Ref<const Eigen::Vector3d>& y_W, double distance_big_M);
  /**
   * Between two adjacent knots at the beginning and the end of an interval,
   * the pushers can make or break contact, but there cannot exist a point
   * breaking contact, and another point making contact.
   * Mathematically we introduce two slack continuous variables b_making_contact
   * and b_breaking_contact, which represents if any point makes (breaks)
   * contact in the interval, and the constraints
   * b_making_contact ≥ b_Q_contact[interval + 1](point_index) -
   * b_Q[contact[interval](point_index) ∀point_index
   * b_breaking_contact ≥ b_Q_contact[interval](point_index) -
   * b_Q[contact[interval + 1](point_index) ∀point_index
   * b_making_contact + b_breaking_contact ≤ 1
   */
  void AddPusherStaticContactConstraint(int interval);

  /**
   * Between two adjacent knots at the beginning and the end of an interval, at
   * most one pusher can break or make contact. Namely at most only one pusher
   * contact point can change from inactive to active, or vice versa.
   * Mathematically, the constraint we impose is that
   * ∑_{point_index} | b[interval](point_index) -b[interval + 1](point_index) |
   * ≤ 1
   * where b are binary variables, b[interval](point_index) = 1 means that the
   * sampled point with point_index is in contact with a pusher at a knot.
   */
  void AddAtMostOnePusherChangeOfContactConstraint(int interval);

  /**
   * Bounds the maximal orientation difference of the block between the
   * beginning and the end of an interval.
   * For two rotation matrix R₁, R₂, the angle difference between them is no
   * larger than θ, if and only if
   *  |R₁ - R₂|² ≤ 4(1 - cosθ)    (1)
   * where the left-hand side is the sum of element wise square, namely
   * |R₁ - R₂|² ≐ ∑ᵢⱼ(R₁(i, j) - R₂(i, j))²
   * The derivation is that
   * | R₁ - R₂ |² = trace[(R₁ - R₂)ᵀ(R₁ - R₂)]
   *              = trace[2I - 2R₁ᵀR₂]
   *              = trace[2I - 2(I + sinα K + (1 - cosα) K²)]
   *              = -2 * (1 - cosα) trace[K²]
   *              = 4(1 - cosα)
   *              ≤ 4(1 - cosθ) = (2√2 sin(θ/2))²
   * where α is the angle between R₁ and R₂, and K is the 3 x 3 skew-symmetric
   * matrix, representing cross product with the axis of the rotation matrix
   * R₁ᵀR₂
   * Notice that we add a convex quadratic constraint (or a second order cone
   * constraint) to the program.
   */
  solvers::Binding<solvers::LorentzConeConstraint>
  AddOrientationDifferenceUpperBound(int interval, double max_angle_difference);

  /** Approximate the quadratic constraint | R₁ - R₂ |₂ ≤ 2√2 sin(θ/2) with
   * linear constraint | R₁ - R₂ |₁ ≤ 3| R₁ - R₂ |₂ ≤ 6√2 sin(θ/2)
   * and | R₁ - R₂ |∞ ≤ | R₁ - R₂ |₂ ≤ 2√2 sin(θ/2)
   */
  void AddOrientationDifferenceUpperBoundLinearApproximation(
      int interval, double max_angle_difference);

  /**
   * If we denote the angle between the object orientation between two
   * consecutive knots as α = ∠(R₁, R₂), and constrain that α ≤ θ for some given
   * maximal angle θ, mathematically this means
   * trace(R₁ᵀR₂) = 1 + 2cosα
   *              ≥ 1 + 2cosθ
   * We will use McCormick envelope to approximate the bilinear product on the
   * left-hand side of the inequality.
   */
  void AddOrientationDifferenceUpperBoundBilinearApproximation(
      int interval, double max_angle_difference);

  /** Return the position of vertex p_BV_.col(vertex_index) at a knot.*/
  Vector3<symbolic::Expression> p_WV(int knot, int vertex_index) const;

  /** Getter for the optimization program. */
  const solvers::MathematicalProgram& prog() const { return *prog_; }

  /** Getter for the mutable optimization program. */
  solvers::MathematicalProgram* get_mutable_prog() { return prog_.get(); }

  int nT() const { return nT_; }

  /** Getter for the object position variables. */
  const std::vector<solvers::VectorDecisionVariable<3>>& p_WB() const {
    return p_WB_;
  }

  /** Getter for the object orientation variables. */
  const std::vector<solvers::MatrixDecisionVariable<3, 3>>& R_WB() const {
    return R_WB_;
  }

  const std::vector<solvers::VectorXDecisionVariable>& vertex_contact_flag()
      const {
    return vertex_contact_flag_;
  }

  const std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>>& f_BV()
      const {
    return f_BV_;
  }

  const std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>>& f_BQ()
      const {
    return f_BQ_;
  }

  const std::vector<solvers::VectorXDecisionVariable>& b_Q_contact() const {
    return b_Q_contact_;
  }

  const std::vector<std::vector<int>>& contact_vertex_indices() const {
    return contact_vertex_indices_;
  }

  const std::vector<std::vector<int>>& contact_Q_indices() const {
    return contact_Q_indices_;
  }

 protected:
  const Eigen::Matrix<double, 5, 1>& phi_R_WB() const { return phi_R_WB_; }

  const std::vector<
      std::array<std::array<solvers::VectorDecisionVariable<2>, 3>, 3>>&
  b_R_WB() const {
    return b_R_WB_;
  }

  /**
   * Return the total wrench ([torque force]) at the object body frame origin,
   * expressed in the object body frame. This wrench includes the gravity
   * wrench, the pusher wrench and the vertex contact wrench.
   */
  Vector6<symbolic::Expression> TotalWrench(int knot) const;

 private:
  std::unique_ptr<solvers::MathematicalProgram> prog_;
  const int nT_;
  const double mass_;
  const Eigen::Vector3d p_BC_;  // CoM position in the body frame B.
  // p_BV_ contains the position of all vertices of the object, expressed in the
  // body frame.
  const Eigen::Matrix3Xd p_BV_;
  int num_pushers_;
  // Q_ contains all candidate pusher contact points on the
  // surface of the object, expressed in the body frame.
  const std::vector<BodyContactPoint> Q_;

  // p_WB_[i] is the position of the object body frame B, expressed in the world
  // frame W.
  std::vector<solvers::VectorDecisionVariable<3>> p_WB_;
  // R_WB_[i] is the orientation of the object body frame B, expressed in the
  // world frame W.
  std::vector<solvers::MatrixDecisionVariable<3, 3>> R_WB_;
  // We will use a mixed-integer convex relaxation of SO(3) constraint. For each
  // entry R_WB_[i](m, n), we will have binary variables b_R_WB_[i](m, n), that
  // determines the interval that R_wb_[i](m, n) is in. We use logarithmic
  // binning for sos2 constraint.
  std::vector<std::array<std::array<solvers::VectorDecisionVariable<2>, 3>, 3>>
      b_R_WB_;
  // The intervals in the sos2 constraint for the rotation matrix R_WB_
  Eigen::Matrix<double, 5, 1> phi_R_WB_;

  // contact_vertex_indices_[i] contains all the indices of the vertices in
  // p_BV_, that may or may not be active at the i'th knot.
  std::vector<std::vector<int>> contact_vertex_indices_;
  // vertex_to_V_map_[knot] is the inverse mapping of
  // contact_vertex_indices_[knot],
  // vertex_to_V_map_[knot][contact_vertex_indices_[knot][i]] = i;
  std::vector<std::unordered_map<int, int>> vertex_to_V_map_;
  // f_BV_[i] is of size 3 x contact_vertex_indices_[i].size(), it contains the
  // contact forces at the possible contact vertices V at knot i, expressed in
  // the body frame.
  std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>> f_BV_;
  // vertex_contact_flag_[i](j) is 1, if the vertex
  // p_BV_.col(contact_vertex_indices[i](j)) is in contact with the environment
  // at knot i; 0 otherwise.
  std::vector<solvers::VectorXDecisionVariable> vertex_contact_flag_;

  // contact_Q_indices_[knot] contains the indices of all possible active pusher
  // contact points in Q, at a given knot.
  std::vector<std::vector<int>> contact_Q_indices_;
  // Q_to_index_map_[knot] is the inverse mapping of contact_Q_indices_[knot],
  // Q_to_index_map_[knot][contact_Q_indices_[knot][i]] = i.
  std::vector<std::unordered_map<int, int>> Q_to_index_map_;
  // b_Q_contact_[knot](i) is true, if the point
  // Q_[contact_Q_indices_[knot](i)] is in contact at a knot; 0 otherwise.
  std::vector<solvers::VectorXDecisionVariable> b_Q_contact_;
  // f_BQ_[knot].col(i) is the contact force at the point
  // Q_[contact_Q_indices_[knot](i)] at a knot, expressed in the body frame B.
  std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>> f_BQ_;
};
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
