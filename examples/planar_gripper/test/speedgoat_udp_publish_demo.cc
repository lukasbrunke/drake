#include "drake/examples/planar_gripper/planar_gripper_udp_utils.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace drake {
namespace examples {
namespace planar_gripper {
int DoMain() {
  struct sockaddr_in myaddr;
  memset(reinterpret_cast<char*>(&myaddr), 0, sizeof(myaddr));
  myaddr.sin_family = AF_INET;
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  int client_port = 1101;
  myaddr.sin_port = htons(client_port);
  int file_descriptor = socket(AF_INET, SOCK_DGRAM, 0);
  if (file_descriptor < 0) {
    throw std::runtime_error("Cannot create the socket.\n");
  }
  int status =
      bind(file_descriptor, reinterpret_cast<struct sockaddr*>(&myaddr),
           sizeof(myaddr));
  if (status < 0) {
    throw std::runtime_error("Cannot bind the UDP file descriptor.");
  }

  struct sockaddr_in servaddr;
  servaddr.sin_family = AF_INET;
  int server_port = 1100;
  servaddr.sin_port = htons(server_port);
  // This corresponds to 192.168.0.1
  unsigned long server_address = 174986663;
  servaddr.sin_addr.s_addr = htonl(server_address);

  SpeedgoatToDrakeUdpMessage udp_msg(9, 9, 3);
  udp_msg.q = Eigen::Matrix<double, 9, 1>::LinSpaced(0.1, 0.9);
  udp_msg.v = Eigen::Matrix<double, 9, 1>::LinSpaced(-0.9, -0.1);
  udp_msg.p_BC << 1.1, 1.2, 1.3, 1.4, 1.5, 1.6;
  udp_msg.in_contact[0] = true;
  udp_msg.in_contact[1] = true;
  udp_msg.in_contact[2] = false;

  std::vector<uint8_t> msg;
  udp_msg.Serialize(&msg);
  status =
      sendto(file_descriptor, msg.data(), msg.size(), 0,
             reinterpret_cast<struct sockaddr*>(&servaddr), sizeof(servaddr));
  if (status < 0) {
    throw std::runtime_error("Cannot send the UDP message.");
  }
  close(file_descriptor);
  return 0;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main() { return drake::examples::planar_gripper::DoMain(); }
