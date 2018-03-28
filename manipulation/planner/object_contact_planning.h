#pragma once

#include <memory>

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
   */
  ObjectContactPlanning(int nT, double mass,
                        const Eigen::Ref<const Eigen::Vector3d>& p_BC,
                        const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV);

  ~ObjectContactPlanning() = default;

  /** Set the indices of all possible contact vertices at a knot.
   * For each vertex in the body indexed by `indices`, we will add binary
   * variable b to indicate whether the vertex is in contact with the
   * environment, and the contact force vector f at the vertex, expressed in the
   * object frame. We will add the constraint
   * -M * b <= f_x <= M * b
   * -M * b <= f_y <= M * b
   * -M * b <= f_z <= M * b
   *  where M is a big number, so that the binary variable b will determine
   *  whether the contact force is zero 0 not.
   */
  void SetContactVertexIndices(int knot, const std::vector<int>& indices,
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

  /** Getter for the optimization program. */
  const solvers::MathematicalProgram* prog() const { return prog_.get(); }

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

 private:
  std::unique_ptr<solvers::MathematicalProgram> prog_;
  const int nT_;
  const double mass_;
  const Eigen::Vector3d p_BC_;  // CoM position in the body frame B.
  // p_BV_ contains the position of all vertices of the object, expressed in the
  // body frame.
  const Eigen::Matrix3Xd p_BV_;
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
};

/**
 * Adds friction cone constraint to contact force f_F, expressed in a frame F.
 * The friction cone has its unit length normal direction as n_F, expressed in
 * the same frame F, with a coefficient of friction being mu.
 */
void AddFrictionConeConstraint(
    double mu, const Eigen::Ref<const Eigen::Vector3d>& n_F,
    const Eigen::Ref<const Vector3<symbolic::Expression>>& f_F,
    solvers::MathematicalProgram* prog);
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
