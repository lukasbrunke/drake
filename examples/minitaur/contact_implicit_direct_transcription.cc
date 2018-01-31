#include "drake/examples/minitaur/contact_implicit_direct_transcription.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace examples {
namespace minitaur {
namespace {

class KinematicsCacheHelper {
 public:
  explicit KinematicsCacheHelper(const RigidBodyTree<double>& tree)
      : tree_{&tree},
        kinsol_(tree.CreateKinematicsCacheWithType<AutoDiffXd>()) {}

  KinematicsCache<AutoDiffXd>& UpdateKinematics(
      const Eigen::Ref<const Eigen::VectorXd>& q,
      const Eigen::Ref<const Eigen::VectorXd>& v) {
    if (q.size() != last_q_.size() || q != last_q_ ||
        v.size() != last_v_.size() || v != last_v_) {
      last_q_ = q;
      last_v_ = v;
      const AutoDiffVecXd q_t = math::initializeAutoDiff(q);
      const AutoDiffVecXd v_t = math::initializeAutoDiff(v);
      kinsol_.initialize(q_t, v_t);
      tree_->doKinematics(kinsol_);
    }
    return kinsol_;
  }

 private:
  const RigidBodyTree<double>* tree_;
  Eigen::VectorXd last_q_;
  Eigen::VectorXd last_v_;
  KinematicsCache<AutoDiffXd> kinsol_;
};

class DirectTranscriptionConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DirectTranscriptionConstraint)

  DirectTranscriptionConstraint(const RigidBodyTree<double>& tree,
                                int num_lambda)
      : Constraint(tree.get_num_positions() + tree.get_num_velocities(),
                   1 + 2 * tree.get_num_positions() +
                       2 * tree.get_num_velocities() + tree.get_num_actuators() +
                       num_lambda,
                   Eigen::VectorXd::Zero(tree.get_num_positions() +
                                         tree.get_num_velocities()),
                   Eigen::VectorXd::Zero(tree.get_num_positions() +
                                         tree.get_num_velocities())),
        tree_(&tree),
        num_positions_{tree.get_num_positions()},
        num_velocities_{tree.get_num_velocities()},
        num_actuators_{tree.get_num_actuators()},
        num_lambda_{num_lambda},
        kinematics_helper1_{std::make_unique<KinematicsCacheHelper>(tree)} {
    DRAKE_THROW_UNLESS(num_positions_ == num_velocities_);
  }

  ~DirectTranscriptionConstraint() override = default;

 protected:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd& y) const override {
    AutoDiffVecXd y_t;
    Eval(math::initializeAutoDiff(x), y_t);
    y = math::autoDiffToValueMatrix(y_t);
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd& y) const override {
    DRAKE_ASSERT(x.size() == num_vars());

    int x_count = 0;
    // A lambda expression to take num_element entreis from x, in a certain order.
    auto x_segment = [x, &x_count](int num_element) {
      x_count += num_element;
      return x.segment(x_count - num_element, num_element);
    };

    const AutoDiffXd h = x(0);
    x_count++;
    const AutoDiffVecXd q0 = x_segment(num_positions_);
    const AutoDiffVecXd v0 = x_segment(num_velocities_);
    const AutoDiffVecXd q1 = x_segment(num_positions_);
    const AutoDiffVecXd v1 = x_segment(num_velocities_);
    const AutoDiffVecXd u1 = x_segment(num_actuators_);
    const AutoDiffVecXd lambda1 = x_segment(num_lambda_);

    kinematics_helper1_->UpdateKinematics(math::autoDiffToValueMatrix(q1),
                                          math::autoDiffToValueMatrix(v1));

  }

 private:
  const RigidBodyTree<double>* tree_;
  const int num_positions_;
  const int num_velocities_;
  const int num_actuators_;
  const int num_lambda_;
  // Stores the kinematics cache at the right knot point.
  mutable std::unique_ptr<KinematicsCacheHelper> kinematics_helper1_;
};
}  // namespace
}  // namespace minitaur
}  // namespace examples
}  // namespace drake
