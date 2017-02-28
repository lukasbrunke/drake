#pragma once

#include <array>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "drake/examples/grasping/forceClosure/dev/contact_facet.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
enum class ForceClosureMetric {
  kQ1Norm = 0,
  kQinfNorm = 1,
};

class ForceClosureGraspOptimization {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ForceClosureGraspOptimization)

  ForceClosureGraspOptimization(
      int num_contacts,
      const std::vector<ContactFacet>& contact_facets,
      const std::vector<std::vector<int>> facet_idx_per_contact,
      ForceClosureMetric metric);

  solvers::MathematicalProgram* prog() const {return prog_.get();};

  const std::vector<solvers::VectorXDecisionVariable>& contact_on_facet() const {return contact_on_facet_;};

  const std::vector<std::vector<solvers::VectorXDecisionVariable>> facet_vertices_weights() const {return facet_vertices_weights_;};

  std::vector<Eigen::VectorXd> contact_on_facet_value() const;

  std::vector<std::vector<Eigen::VectorXd>> facet_vertices_weights_value() const;

  Eigen::Matrix3Xd contact_pos(const std::vector<Eigen::VectorXd>& contact_on_facet,
    const std::vector<std::vector<Eigen::VectorXd>>& facet_vertices_weights) const;

 private:
  int num_contacts_{};
  std::unique_ptr<drake::solvers::MathematicalProgram> prog_{};
  const std::vector<ContactFacet> contact_facets_{};

  // facet_idx_per_contact_[i] contains the indices of all possible contact
  // facets, that the i'th contact can land.
  std::vector<std::vector<int>> facet_idx_per_contact_;

  // contact_on_facet_[i][j] is true if i'th contact is on the facet
  // contact_facet[facet_idx_per_contact[i][j]]. This is the binary variable
  // to indicate which contact is on which facet.
  std::vector<solvers::VectorXDecisionVariable> contact_on_facet_;

  // facet_vertices_weights[i][j] is has length contact_facets_[facet_idx_per_contact_[i][j]].num_vertices()
  // Namely if contact_on_facet[i][j] == 1, then the contact location for the i'th
  // contact is the convex combination of the vertices of facet facet_idx_per_contact_[i][j]
  std::vector<std::vector<solvers::VectorXDecisionVariable>> facet_vertices_weights_{};

};
}  // namespace forceClosure
}  // namespace grasping
}  // namespace examples
}  // namespace drake