#include "drake/examples/multi_contact_planning/contact_force_planning.h"

#include <numeric>

namespace drake {
namespace examples {
namespace multi_contact_planning {
ContactForcePlanning::ContactForcePlanning(
    int nT, double mass, const Eigen::Ref<const Eigen::Matrix3d>& I_B,
    const std::vector<ContactPoint>& candidate_contact_points,
    int num_arm_points, double max_normal_contact_force)
    : solvers::MathematicalProgram(),
      nT_(nT),
      mass_(mass),
      I_B_(I_B),
      candidate_contact_points_(candidate_contact_points),
      num_arm_points_(num_arm_points),
      f_B_(candidate_contact_points.size()),
      b_(num_arm_points_) {
  for (int i = 0; i < num_candidate_contact_points(); ++i) {
    const std::string f_B_name = "f_B[" + std::to_string(i) + "]";
    f_B_[i] = NewContinuousVariables<3, Eigen::Dynamic>(3, nT_, f_B_name);
    const std::string edge_weights_name =
        "edge_weights[" + std::to_string(i) + "]";
    auto edge_weights = NewContinuousVariables(
        candidate_contact_points_[i].cone.num_edges(), nT_, edge_weights_name);
    AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(),
                             edge_weights);
  }
  for (int i = 0; i < num_arm_points_; ++i) {
    const std::string b_name = "b[" + std::to_string(i) + "]";
    b_[i] = NewBinaryVariables(num_candidate_contact_points(), nT_, b_name);
    // Add the constraint that one arm touches at most one contact point at a
    // time.
    AddLinearConstraint(b_[i].cast<symbolic::Expression>().colwise().sum() <=
                        Eigen::RowVectorXd::Ones(1, nT_));
  }
  // Use the big-M trick to determine when the contact force is active.
  // f_B_[j].col(k).dot(cone[j].normal()) <= max_normal_contact_force * (sum_i
  // b_[i](j, k)
  MatrixX<symbolic::Expression> point_in_contact(num_candidate_contact_points(),
                                                 nT_);
  point_in_contact.setZero();
  for (int i = 0; i < num_arm_points_; ++i) {
    point_in_contact += b_[i].cast<symbolic::Expression>();
  }
  for (int j = 0; j < num_candidate_contact_points(); ++j) {
    AddLinearConstraint(
        candidate_contact_points_[j].cone.unit_normal().transpose() * f_B_[j] <=
        point_in_contact.row(j) * max_normal_contact_force);
  }

  // Add the constraint that one point cannot be touched by more than 1 arms at
  // a time.
  Eigen::Array<symbolic::Expression, Eigen::Dynamic, Eigen::Dynamic>
      num_touched_arms(num_candidate_contact_points(), nT_);
  num_touched_arms.setZero();
  for (int i = 0; i < num_arm_points; ++i) {
    num_touched_arms += b_[i].cast<symbolic::Expression>().array();
  }
  AddLinearConstraint(
      num_touched_arms <=
      Eigen::MatrixXd::Ones(num_candidate_contact_points(), nT_).array());

  // Add the constraint that the arm cannot slide between two contact points.
  // Namely sum_j |b_[i](j, k) - b_[i](j, k+1)| <= 1
  for (int i = 0; i < num_arm_points; ++i) {
    const std::string contact_change_name =
        "contact_change[" + std::to_string(i) + "]";
    // contact_change(j, k) >= b_[i](j, k) - b_[i](j, k+1)
    // contact_change(j, k) >= b_[i](j, k+1) - b_[i](j, k)
    auto contact_change = NewContinuousVariables(num_candidate_contact_points(),
                                                 nT_ - 1, contact_change_name);
    AddLinearConstraint(
        contact_change.array() >=
        b_[i].leftCols(nT_ - 1).cast<symbolic::Expression>().array() -
            b_[i].rightCols(nT_ - 1).cast<symbolic::Expression>().array());
    AddLinearConstraint(
        contact_change.array() >=
        b_[i].rightCols(nT_ - 1).cast<symbolic::Expression>().array() -
            b_[i].leftCols(nT_ - 1).cast<symbolic::Expression>().array());
    AddLinearConstraint(
        contact_change.cast<symbolic::Expression>().colwise().sum() <=
        Eigen::RowVectorXd::Ones(1, nT_ - 1));
  }
}

}  // namespace multi_contact_planning
}  // namespace examples
}  // namespace drake
