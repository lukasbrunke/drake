#pragma once

#include <netinet/in.h>
#include <string>

#include "drake/systems/framework/leaf_system.h"
namespace drake {
namespace examples {
namespace planar_gripper {
// This is rather arbitrary, for now.
constexpr double kGripperUdpStatusPeriod = 0.010;

/**
 * This system takes the vector-valued finger tip contact force (fy, fz),
 * serialize the vector to binary data, and then publish the binary data to
 * UDP.
 */
class DesiredContactForceUdpPublisherSystem
    : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DesiredContactForceUdpPublisherSystem)

  /**
   * @param num_fingers The number of fingers in the system.
   * @param client_port The port number of the client (sender).
   * @param server_port The port number on the server (receiver).
   * @param server_address The IP address of the server (converted to unsigned
   * long).
   */
  DesiredContactForceUdpPublisherSystem(double publish_period, int num_fingers,
                                        int client_port, int server_port,
                                        unsigned long server_address);

  ~DesiredContactForceUdpPublisherSystem();

 private:
  systems::EventStatus PublishInputAsUdpMessage(
      const systems::Context<double>& context) const;

  std::string MakeOutput() const;

  void Output(const systems::Context<double>& context,
              std::string* output) const;

  std::string Serialize(const systems::Context<double>& context) const;

  int num_fingers_{};
  int file_descriptor_{};
  int server_port_{};
  // server address.
  unsigned long server_address_{};
};

/**
 * This is the abstract state of the SpeedgoatUdpReceiverSystem.
 */
struct SpeedgoatToDrakeUdpMessage {
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(SpeedgoatToDrakeUdpMessage)

  SpeedgoatToDrakeUdpMessage(int m_num_positions, int m_num_velocities,
                             int m_num_fingers)
      : q(m_num_positions),
        v(m_num_velocities),
        p_BC(2, m_num_fingers),
        in_contact(m_num_fingers, false) {}

  int message_size() const;

  void Deserialize(uint8_t* msg, int msg_size);

  void Serialize(std::vector<uint8_t>* msg) const;

  // Time in microseconds.
  uint32_t utime;
  // Generalized position (including the finger and the manipuland).
  Eigen::VectorXd q;
  // Generalized velocity.
  Eigen::VectorXd v;
  // The y/z position of witness point C on the manipuland frame B.
  // p_BC.col(i) is the position of the manipuland witness point (on the
  // manipuland face) for the i'th finger.
  Eigen::Matrix2Xd p_BC;
  // in_contact_[i] is true if i'th finger is in contact with the manipuland,
  // false otherwise.
  std::vector<bool> in_contact;
};

/**
 * This system runs on the Drake machine. It listens to the UDP message sent
 * from the SpeedGoat machine. The UDP message contains the system state (both
 * the finger and the manipuland state) and the finger contact points.
 * The UDP datagram is binary bits, that should be encoded as
 * utime (int), state (double), witness_points (double), in_contact (bool)
 * as a string.
 *
 * The system has several outputs, including one output port for state, one
 * output for the witness point for each finger, and one output for whether the
 * finger is in contact or not.
 */
class SpeedgoatUdpReceiverSystem : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SpeedgoatUdpReceiverSystem)

  SpeedgoatUdpReceiverSystem(int num_positions, int num_velocities,
                             int num_fingers, int local_port);

  ~SpeedgoatUdpReceiverSystem();

  const systems::OutputPort<double>& get_state_output_port() const {
    DRAKE_DEMAND(state_output_port_ != nullptr);
    return *state_output_port_;
  }

 private:
  systems::EventStatus ProcessMessageAndStoreToAbstractState(
      const systems::Context<double>& context,
      systems::State<double>* state) const;
  void OutputStateStatus(const systems::Context<double>& context,
                         systems::BasicVector<double>* output) const;
  void OutputWitnessPointsStatus(const systems::Context<double>& context,
                                 systems::BasicVector<double>* output) const;
  void OutputInContactStatus(const systems::Context<double>& context,
                             std::vector<bool>* output) const;

  std::vector<bool> MakeInContactOutputStatus() const;

  int num_positions_{};
  int num_velocities_{};
  int num_fingers_{};
  int file_descriptor_{};

  // The mutex that guards buffer_;
  mutable std::mutex received_message_mutex_;

  std::vector<uint8_t> buffer_;

  const systems::OutputPort<double>* state_output_port_{};
  const systems::OutputPort<double>* witness_points_output_port_{};
  const systems::OutputPort<double>* in_contact_output_port_{};
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
