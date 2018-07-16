#pragma once

#include <vector>

#include "drake/examples/multi_contact_planning/friction_cone.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace examples {
namespace multi_contact_planning {
/**
 * Both the contact position and the friction cone are expressed in the same
 * frame.
 */
struct ContactPoint {
  ContactPoint(const Eigen::Ref<const Eigen::Vector3d>& pos_in, int num_edges,
               const Eigen::Ref<const Eigen::Vector3d>& unit_normal, double mu)
      : pos(pos_in), cone(num_edges, unit_normal, mu) {}

  const Eigen::Vector3d pos;
  const LinearizedFrictionCone cone;
};

/**
 * Given a desired trajectory of a single rigid object (the trajectory includes
 * both the translation and the orientation position/velocity/acceleration), and
 * some candidate contact points on the surface of the object, we will select
 * the contact locations, and also compute the contact forces at each location,
 * so as to realize the motion.
 */
class ContactForcePlanning : public solvers::MathematicalProgram {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactForcePlanning)

  ContactForcePlanning(
      int nT, double mass, const Eigen::Ref<const Eigen::Matrix3d>& I_B,
      const std::vector<ContactPoint>& candidate_contact_points,
      int num_arm_points, double max_normal_contact_force);

  /**
   * Set the total wrench of the object at a given knot.
   */
  void SetObjectWrench(int knot, const Eigen::Ref<const Eigen::Matrix3d>& R_WB,
                       const Eigen::Ref<const Eigen::Vector3d>& a_WB,
                       const Eigen::Ref<const Eigen::Vector3d>& omega_WB,
                       const Eigen::Ref<const Eigen::Vector3d>& DomegaDt_WB);

  int nT() const { return nT_; }

  double mass() const { return mass_; }

  const Eigen::Matrix3d& I_B() const { return I_B_; }

  double gravity() const { return gravity_; }

  const std::vector<ContactPoint>& candidate_contact_points() const {
    return candidate_contact_points_;
  }

  int num_arm_points() const { return num_arm_points_; }

  const std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>>& f_B()
      const {
    return f_B_;
  }

  const std::vector<solvers::MatrixXDecisionVariable>& b() const { return b_; }

 private:
  int num_candidate_contact_points() const {
    return static_cast<int>(candidate_contact_points_.size());
  }

  const int nT_;               // number of time knots
  const double mass_;          // The mass of the object.
  const Eigen::Matrix3d I_B_;  // The moment of ineria of the object, expressed
                               //  in the body frame B.
  const double gravity_{9.81};
  // candidate_contact_points contains all the candidate contact points on the
  // surface of the object. The contact location and the friction cones are
  // all expressed in the body frame.
  const std::vector<ContactPoint> candidate_contact_points_;
  // num_arm_points are the maximal number of contact points on all arms, which
  // is also the maximal number of candidate_contat_points that can be active
  // simultaneously.
  const int num_arm_points_;

  // f_B_ is of size num_candidate_contact_points(). f_B_[i] is a 3 x nT_
  // matrix,
  // f_B_[i].col(j) is the contact force (expressed in the body frame) of the
  // i'th contact_point, at j'th knot.
  std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>> f_B_;

  // b_ is of size num_arm_points_. b_[i] is a
  // num_candidate_contact_points() x nT_ matrix, b_[i](j, k) means that arm i
  // touches contact point j at knot k.
  std::vector<solvers::MatrixXDecisionVariable> b_;
};
}  // namespace multi_contact_planning
}  // namespace examples
}  // namespace drake
