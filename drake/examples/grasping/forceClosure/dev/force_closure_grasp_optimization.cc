#include "drake/examples/grasping/forceClosure/dev/force_closure_grasp_optimization.h"

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
    for (int j = 0; j < num_facet_for_contact_i; ++j) {
      contact_on_facet_sum += contact_on_facet_[i](j);
    }
    prog_->AddLinearConstraint(contact_on_facet_sum == 1);

    facet_vertices_weights_[i].resize(num_facet_for_contact_i);
    for (int j = 0; j < num_facet_for_contact_i; ++j) {
      int facet_idx = facet_idx_per_contact_[i][j];
      facet_vertices_weights_[i][j] = prog_->NewContinuousVariables(contact_facets_[facet_idx].num_vertices(),"lambda");
      prog_->AddBoundingBoxConstraint(0, 1, facet_vertices_weights_[i][j]);
      Expression facet_vertices_weights_sum(0);
      for (int k = 0; k < facet_vertices_weights_[i][j].rows(); ++k) {
        facet_vertices_weights_sum += facet_vertices_weights_[i][j](k);
      }
      prog_->AddLinearConstraint(facet_vertices_weights_sum == contact_on_facet_[i](j));
    }

  }
}
}
}
}
}