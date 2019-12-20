#include "drake/examples/planar_gripper/planar_gripper_udp_utils.h"

#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/analysis/simulator.h"

namespace drake {
namespace examples {
namespace planar_gripper {
int DoMain() {
  systems::DiagramBuilder<double> builder;
  const double publish_period = 0.01;
  int num_fingers = 3;
  int client_port = 1;
  int server_port = 100;
  // This corresponds to 192.168.0.1
  unsigned long server_address = 174986663;
  auto publisher = builder.AddSystem<DesiredContactForceUdpPublisherSystem>(
      publish_period, num_fingers, client_port, server_port, server_address);
  Eigen::Matrix<double, 6, 1> val;
  val << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
  auto source = builder.AddSystem<systems::ConstantVectorSource>(val);
  builder.Connect(source->get_output_port(), publisher->get_input_port(0));
  builder.ExportOutput(publisher->get_output_port(0));
  auto diagram = builder.Build();

  auto diagram_context = diagram->CreateDefaultContext();
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(1);
  simulator.Initialize();
  simulator.AdvanceTo(10);
  return 0;
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main() { drake::examples::planar_gripper::DoMain(); }
