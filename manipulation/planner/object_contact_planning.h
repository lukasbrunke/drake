#pragma once

#include <memory>
#include <unordered_map>

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
   * @param Q The candidate pusher contact location Q
   */
  ObjectContactPlanning(int nT, double mass,
                        const Eigen::Ref<const Eigen::Vector3d>& p_BC,
                        const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV,
                        int num_pushers,
                        const std::vector<BodyContactPoint>& Q);

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
   * Adds static equilibrium constraint at each knot.
   */
  void AddStaticEquilibriumConstraint();

  /**
   * Between two adjacent knots, the pusher cannot move from sampled contact
   * point to the another sampled point. So in between two adjacent knots, at
   * most one pusher can break or  make contact. All other pushers must remain
   * static in the body frame.
   * Mathematically, the constraint we impose is that
   * sum_{point_index} | b[knot-1](point_index) -b[knot](point_index) | ≤ 1
   * where b are binary variables, b[knot](point_index) = 1 means that the
   * sampled point with point_index is in contact with a pusher at a knot.
   */
  void AddPusherStaticContactConstraint();

  /** Getter for the optimization program. */
  const solvers::MathematicalProgram& prog() const { return *prog_; }

  /** Getter for the mutable optimization program. */
  solvers::MathematicalProgram* get_mutable_prog() { return prog_.get(); }

  /** Getter for the object position variables. */
  const std::vector<solvers::VectorDecisionVariable<3>>& p_WB() const {
    return p_WB_;
  }

  /** Getter for the object orientation variables. */
  const std::vector<solvers::MatrixDecisionVariable<3, 3>>& R_WB() const {
    return R_WB_;
  }

  const std::vector<solvers::MatrixDecisionVariable<1, Eigen::Dynamic>>&
  vertex_contact_flag() const {
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
  // f_BV_[i] is of size 3 x contact_vertex_indices_[i].size(), it contains the
  // contact forces at the possible contact vertices V at knot i, expressed in
  // the body frame.
  std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>> f_BV_;
  // vertex_contact_flag_[i](j) is 1, if the vertex
  // p_BV_.col(contact_vertex_indices[i](j)) is in contact with the environment
  // at knot i; 0 otherwise.
  std::vector<solvers::MatrixDecisionVariable<1, Eigen::Dynamic>>
      vertex_contact_flag_;

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
