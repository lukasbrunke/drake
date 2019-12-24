#include "drake/examples/planar_gripper/planar_gripper_udp_utils.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GTEST_TEST(DrakeToSpeedgoatUdpMessageTest, Test) {
  DrakeToSpeedgoatUdpMessage dut(3);
  EXPECT_EQ(dut.message_size(), sizeof(uint32_t) + sizeof(double) * 6);
  dut.utime = 1000;
  dut.f_BC << 0.1, 0.2, 0.3, -0.1, -0.2, -0.3;
  std::vector<uint8_t> msg;
  dut.Serialize(&msg);
  EXPECT_EQ(msg.size(), dut.message_size());
  DrakeToSpeedgoatUdpMessage recovered_msg(3);
  recovered_msg.Deserialize(msg.data(), msg.size());
  EXPECT_EQ(recovered_msg.utime, dut.utime);
  EXPECT_TRUE(CompareMatrices(recovered_msg.f_BC, dut.f_BC));
}

GTEST_TEST(SpeedgoatToDrakeUdpMessageTest, Test) {
  SpeedgoatToDrakeUdpMessage dut(9, 9, 3);
  EXPECT_EQ(dut.q.rows(), 9);
  EXPECT_EQ(dut.v.rows(), 9);
  EXPECT_EQ(dut.p_BC.rows(), 2);
  EXPECT_EQ(dut.p_BC.cols(), 3);
  EXPECT_EQ(dut.in_contact.size(), 3);
  dut.q = Eigen::Matrix<double, 9, 1>::LinSpaced(0.1, 0.9);
  dut.v = Eigen::Matrix<double, 9, 1>::LinSpaced(-0.9, -0.1);
  dut.p_BC << 1.1, 1.2, 1.3, 1.4, 1.5, 1.6;
  dut.in_contact[0] = false;
  dut.in_contact[1] = true;
  dut.in_contact[2] = true;

  std::vector<uint8_t> msg;
  dut.Serialize(&msg);
  EXPECT_EQ(msg.size(), dut.message_size());
  SpeedgoatToDrakeUdpMessage new_udp_msg(9, 9, 3);
  new_udp_msg.Deserialize(msg.data(), msg.size());
  EXPECT_EQ(dut.utime, new_udp_msg.utime);
  EXPECT_TRUE(CompareMatrices(dut.q, new_udp_msg.q));
  EXPECT_TRUE(CompareMatrices(dut.v, new_udp_msg.v));
  EXPECT_TRUE(CompareMatrices(dut.p_BC, new_udp_msg.p_BC));
  EXPECT_EQ(dut.in_contact.size(), new_udp_msg.in_contact.size());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(dut.in_contact[i], new_udp_msg.in_contact[i]);
  }
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
