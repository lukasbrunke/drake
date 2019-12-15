#pragma once

#include <netinet/in.h>
#include <string>

#include "drake/systems/framework/leaf_system.h"
namespace drake {
namespace examples {
namespace planar_gripper {
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
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
