#include "drake/examples/grasping/forceClosure/dev/force_closure_grasp_optimization.h"

using Eigen::VectorXd;

using drake::symbolic::Expression;

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
ForceClosureGraspOptimization::ForceClosureGraspOptimization(
    int num_contacts,
    const std::vector<ContactFacet>& contact_facets,
    const std::vector<std::vector<int>> facet_idx_per_contact,
    ForceClosureMetric metric)
    : num_contacts_(num_contacts),
      prog_(std::make_unique<solvers::MathematicalProgram>()),
      contact_facets_(contact_facets),
      facet_idx_per_contact_(facet_idx_per_contact) {
  // Now add the constraint that the contact point is on one of the facet.
  contact_on_facet_.resize(num_contacts_);
  facet_vertices_weights_.resize(num_contacts_);
  for (int i = 0; i < num_contacts_; ++i) {
    int num_facet_for_contact_i = facet_idx_per_contact_[i].size();
    contact_on_facet_[i] = prog_->NewBinaryVariables(num_facet_for_contact_i, "b");
    // Each contact is assigned to one facet.
    Expression contact_on_facet_sum(0);
    prog_->AddLinearConstraint(contact_on_facet_[i].cast<Expression>().sum() == 1);

    facet_vertices_weights_[i].resize(num_facet_for_contact_i);
    for (int j = 0; j < num_facet_for_contact_i; ++j) {
      int facet_idx = facet_idx_per_contact_[i][j];
      facet_vertices_weights_[i][j] = prog_->NewContinuousVariables(contact_facets_[facet_idx].num_vertices(),"lambda");
      prog_->AddBoundingBoxConstraint(0, 1, facet_vertices_weights_[i][j]);
      prog_->AddLinearConstraint(facet_vertices_weights_[i][j].cast<Expression>().sum() == contact_on_facet_[i](j));
    }
  }
}

std::vector<VectorXd> ForceClosureGraspOptimization::contact_on_facet_value() const {
  std::vector<VectorXd> contact_on_facet(num_contacts_);
  for (int i = 0; i < num_contacts_; ++i) {
    contact_on_facet[i] = prog_->GetSolution(contact_on_facet_[i]);
  }
  return contact_on_facet;
}

std::vector<std::vector<VectorXd>> ForceClosureGraspOptimization::facet_vertices_weights_value() const {
  std::vector<std::vector<VectorXd>> facet_vertices_weights(num_contacts_);
  for (int i = 0; i < num_contacts_; ++i) {
    facet_vertices_weights[i].resize(facet_idx_per_contact_[i].size());
    for (int j = 0; j < static_cast<int>(facet_idx_per_contact_[i].size()); ++j) {
      facet_vertices_weights[i][j] = prog_->GetSolution(facet_vertices_weights_[i][j]);
    }
  }
  return facet_vertices_weights;
}
Eigen::Matrix3Xd ForceClosureGraspOptimization::contact_pos(
    const std::vector<VectorXd>& contact_on_facet,
    const std::vector<std::vector<VectorXd>>& facet_vertices_weights) const {
  Eigen::Matrix3Xd pos(3, num_contacts_);
  pos.setZero();
  for (int i = 0; i < num_contacts_; ++i) {
    for (int j = 0; j < static_cast<int>(facet_idx_per_contact_[i].size()); ++j) {
      pos.col(i) += contact_on_facet[i](j) * contact_facets_[facet_idx_per_contact_[i][j]].vertices() * facet_vertices_weights[i][j];
    }
  }
  return pos;
}
}  // namespace forceClosure
}  // namespace grasping
}  // namespace examples
}  // namespace drake
