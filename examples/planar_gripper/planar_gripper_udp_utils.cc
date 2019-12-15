#include "drake/examples/planar_gripper/planar_gripper_udp_utils.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace drake {
namespace examples {
namespace planar_gripper {

DesiredContactForceUdpPublisherSystem::DesiredContactForceUdpPublisherSystem(
    double publish_period, int num_fingers, int client_port, int server_port,
    unsigned long server_address)
    : num_fingers_(num_fingers),
      file_descriptor_{socket(AF_INET, SOCK_DGRAM, 0)},
      server_port_{server_port},
      server_address_{server_address} {
  struct sockaddr_in myaddr;
  // memset((char*) &myaddr, 0, sizeof(myaddr));
  myaddr.sin_family = AF_INET;
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr.sin_port = htons(client_port);
  int status =
      bind(file_descriptor_, reinterpret_cast<struct sockaddr*>(&myaddr),
           sizeof(myaddr));
  if (status < 0) {
    throw std::runtime_error("Cannot bind the UDP file descriptor.");
  }

  this->DeclareForcedPublishEvent(
      &DesiredContactForceUdpPublisherSystem::PublishInputAsUdpMessage);

  const double offset = 0.0;
  this->DeclarePeriodicPublishEvent(
      publish_period, offset,
      &DesiredContactForceUdpPublisherSystem::PublishInputAsUdpMessage);

  this->DeclareVectorInputPort("contact force",
                               systems::BasicVector<double>(2 * num_fingers_));
  this->DeclareAbstractOutputPort(
      &DesiredContactForceUdpPublisherSystem::MakeOutput,
      &DesiredContactForceUdpPublisherSystem::Output);
}

std::string DesiredContactForceUdpPublisherSystem::Serialize(
    const systems::Context<double>& context) const {
  std::stringstream ss;
  int utime = context.get_time() * 1e6;
  ss << std::to_string(utime) << " ";
  const systems::BasicVector<double>* input = this->EvalVectorInput(context, 0);
  for (int i = 0; i < num_fingers_; ++i) {
    ss << input->GetAtIndex(2 * i) << " ";
    ss << input->GetAtIndex(2 * i + 1) << " ";
  }
  return ss.str();
}

systems::EventStatus
DesiredContactForceUdpPublisherSystem::PublishInputAsUdpMessage(
    const systems::Context<double>& context) const {
  const std::string output_msg = this->Serialize(context);

  struct sockaddr_in servaddr;
  servaddr.sin_family = AF_INET;
  servaddr.sin_port = htons(server_port_);
  servaddr.sin_addr.s_addr = htonl(server_address_);
  int status =
      sendto(file_descriptor_, output_msg.c_str(), output_msg.length(), 0,
             reinterpret_cast<struct sockaddr*>(&servaddr), sizeof(servaddr));
  if (status < 0) {
    throw std::runtime_error("Cannot send the UDP message.");
  }
  std::cout << "send UDP message\n";
  return systems::EventStatus::Succeeded();
}

std::string DesiredContactForceUdpPublisherSystem::MakeOutput() const {
  std::string output;
  return output;
}

void DesiredContactForceUdpPublisherSystem::Output(
    const systems::Context<double>& context, std::string* output) const {
  *output = this->Serialize(context);
}

DesiredContactForceUdpPublisherSystem::
    ~DesiredContactForceUdpPublisherSystem() {
  close(file_descriptor_);
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
