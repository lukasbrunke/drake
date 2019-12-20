#include "drake/examples/planar_gripper/planar_gripper_udp_utils.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "drake/systems/framework/abstract_values.h"

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

int SpeedgoatToDrakeUdpMessage::message_size() const {
  return sizeof(uint32_t) +
         sizeof(double) * (q.rows() + v.rows() + p_BC.cols() * 2) +
         sizeof(bool) * in_contact.size();
}
void SpeedgoatToDrakeUdpMessage::Deserialize(uint8_t* msg, int msg_size) {
  DRAKE_DEMAND(this->message_size() == msg_size);
  memcpy(&(this->utime), msg, sizeof(uint32_t));
  int start = sizeof(uint32_t);
  memcpy(this->q.data(), msg + start, sizeof(double) * q.rows());
  start += sizeof(double) * q.rows();
  memcpy(this->v.data(), msg + start, sizeof(double) * v.rows());
  start += sizeof(double) * v.rows();
  memcpy(this->p_BC.data(), msg + start, sizeof(double) * 2 * p_BC.cols());
  start += sizeof(double) * 2 * p_BC.cols();
  for (int i = 0; i < static_cast<int>(in_contact.size()); ++i) {
    bool flag;
    memcpy(&flag, msg + start, sizeof(bool));
    in_contact[i] = flag;
    start += sizeof(bool);
  }
}

void SpeedgoatToDrakeUdpMessage::Serialize(std::vector<uint8_t>* msg) const {
  msg->resize(this->message_size());
  memcpy(msg->data(), &this->utime, sizeof(this->utime));
  int start = sizeof(uint32_t);
  memcpy(msg->data() + start, q.data(), sizeof(double) * q.rows());
  start += sizeof(double) * q.rows();
  memcpy(msg->data() + start, v.data(), sizeof(double) * v.rows());
  start += sizeof(double) * v.rows();
  memcpy(msg->data() + start, p_BC.data(), sizeof(double) * 2 * p_BC.cols());
  start += sizeof(double) * p_BC.cols() * 2;
  for (int i = 0; i < static_cast<int>(in_contact.size()); ++i) {
    bool flag = in_contact[i];
    memcpy(msg->data() + start, &flag, sizeof(bool));
    start += sizeof(bool);
  }
}

SpeedgoatUdpReceiverSystem::SpeedgoatUdpReceiverSystem(int num_positions,
                                                       int num_velocities,
                                                       int num_fingers,
                                                       int local_port)
    : num_positions_{num_positions},
      num_velocities_{num_velocities},
      num_fingers_{num_fingers},
      file_descriptor_{socket(AF_INET, SOCK_DGRAM, 0)} {
  // The implementation of this class follows
  // https://www.cs.rutgers.edu/~pxk/417/notes/sockets/udp.html
  if (file_descriptor_ < 0) {
    throw std::runtime_error(
        " SpeedgoatUdpReceiverSystem: cannot create a socket.");
  }
  struct sockaddr_in myaddr;
  myaddr.sin_family = AF_INET;
  // bind the socket to any valid IP address
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr.sin_port = htons(local_port);
  if (bind(file_descriptor_, reinterpret_cast<struct sockaddr*>(&myaddr),
           sizeof(myaddr)) < 0) {
    throw std::runtime_error(
        "SpeedgoatUdpReceiverSystem: cannot bind the socket");
  }

  this->DeclareAbstractState(
      std::make_unique<Value<SpeedgoatToDrakeUdpMessage>>(
          num_positions_, num_velocities_, num_fingers_));
  this->DeclarePeriodicUnrestrictedUpdateEvent(
      kGripperUdpStatusPeriod, 0.,
      &SpeedgoatUdpReceiverSystem::ProcessMessageAndStoreToAbstractState);

  state_output_port_ = &this->DeclareVectorOutputPort(
      "state", systems::BasicVector<double>(num_positions_ + num_velocities_),
      &SpeedgoatUdpReceiverSystem::OutputStateStatus);

  witness_points_output_port_ = &this->DeclareVectorOutputPort(
      "witness point", systems::BasicVector<double>(2 * num_fingers_),
      &SpeedgoatUdpReceiverSystem::OutputWitnessPointsStatus);

  in_contact_output_port_ = &this->DeclareAbstractOutputPort(
      "in_contact_status",
      &SpeedgoatUdpReceiverSystem::MakeInContactOutputStatus,
      &SpeedgoatUdpReceiverSystem::OutputInContactStatus);
}

std::vector<bool> SpeedgoatUdpReceiverSystem::MakeInContactOutputStatus()
    const {
  std::vector<bool> in_contact(num_fingers_);
  return in_contact;
}

SpeedgoatUdpReceiverSystem::~SpeedgoatUdpReceiverSystem() {
  close(file_descriptor_);
}

systems::EventStatus
SpeedgoatUdpReceiverSystem::ProcessMessageAndStoreToAbstractState(
    const systems::Context<double>&, systems::State<double>* state) const {
  systems::AbstractValues& abstract_state = state->get_mutable_abstract_state();
  std::vector<uint8_t> buffer(abstract_state.get_value(0)
                                  .get_value<SpeedgoatToDrakeUdpMessage>()
                                  .message_size());
  struct sockaddr_in remaddr;
  socklen_t addrlen = sizeof(remaddr);
  const int recvlen =
      recvfrom(file_descriptor_, buffer.data(), buffer.size(), 0,
               reinterpret_cast<struct sockaddr*>(&remaddr), &addrlen);
  if (recvlen > 0) {
    abstract_state.get_mutable_value(0)
        .get_mutable_value<SpeedgoatToDrakeUdpMessage>()
        .Deserialize(buffer.data(), buffer.size());
    std::cout << abstract_state.get_value(0)
                     .get_value<SpeedgoatToDrakeUdpMessage>()
                     .q.transpose()
              << "\n";
  }
  return systems::EventStatus::Succeeded();
}

void SpeedgoatUdpReceiverSystem::OutputStateStatus(
    const systems::Context<double>& context,
    systems::BasicVector<double>* output) const {
  Eigen::VectorBlock<VectorX<double>> output_vec = output->get_mutable_value();
  output_vec.head(num_positions_) = context.get_abstract_state()
                                        .get_value(0)
                                        .get_value<SpeedgoatToDrakeUdpMessage>()
                                        .q;
}

void SpeedgoatUdpReceiverSystem::OutputWitnessPointsStatus(
    const systems::Context<double>& context,
    systems::BasicVector<double>* output) const {
  Eigen::VectorBlock<VectorX<double>> output_vec = output->get_mutable_value();
  for (int i = 0; i < num_fingers_; ++i) {
    output_vec.segment<2>(2 * i) = context.get_abstract_state()
                                       .get_value(0)
                                       .get_value<SpeedgoatToDrakeUdpMessage>()
                                       .p_BC.col(i);
  }
}

void SpeedgoatUdpReceiverSystem::OutputInContactStatus(
    const systems::Context<double>& context, std::vector<bool>* output) const {
  *output = context.get_abstract_state()
                .get_value(0)
                .get_value<SpeedgoatToDrakeUdpMessage>()
                .in_contact;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
