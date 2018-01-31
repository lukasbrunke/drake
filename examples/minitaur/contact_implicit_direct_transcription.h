#pragma once

#include <memory>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/rigid_body_tree.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/trajectory_optimization/multiple_shooting.h"

namespace drake {
namespace examples {
namespace minitaur {
// Implement the contact implicit trajectory optimization from the paper
// A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact
// by Michael Posa, Cecilia Cantu and Russ Tedrake
class ContactImplicitDirectTranscription
    : public systems::trajectory_optimization::MultipleShooting {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ContactImplicitDirectTranscription)

  ContactImplicitDirectTranscription(const RigidBodyTree<double>& tree,
                                     int num_time_samples,
                                     double minimum_timestep,
                                     double maximum_timestep);

  ~ContactImplicitDirectTranscription() override {}

 private:
  void DoAddRunningCost(const symbolic::Expression& e) override;

  // Store system-relevant data for e.g. computing the derivatives during
  // trajectory reconstruction.
  const RigidBodyTree<double>* tree_{nullptr};
};
}  // namespace minitaur
}  // namespace examples
}  // namespace drake
