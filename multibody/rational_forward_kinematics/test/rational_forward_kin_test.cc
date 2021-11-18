#include <gtest/gtest.h>

#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

namespace drake {
namespace multibody {
TEST_F(IiwaTest, DummyVarTest) {
  RationalForwardKinematics forward_kin(*iiwa_);
  for (int i = 0; i < forward_kin.t().rows(); ++i) {
    std::cout << forward_kin.t()(i).get_name() << " "
              << forward_kin.t()(i).get_id() << "\n";
  }
  const symbolic::Variable dummy_var("dummy_var");
  std::cout << "dummy_var.get_id(): " << dummy_var.get_id() << "\n";
  for (int i = 0; i < forward_kin.t().rows(); ++i) {
    EXPECT_GT(dummy_var.get_id(), forward_kin.t()(i).get_id());
  }
}
}  // namespace multibody
}  // namespace drake
