#include "drake/examples/planar_gripper/planar_gripper_udp_utils.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace planar_gripper {
int DoMain() {
  systems::DiagramBuilder<double> builder;
  int client_port = 1100;
  builder.AddSystem<SpeedgoatUdpReceiverSystem>(9, 9, 3, client_port);
  auto diagram = builder.Build();

  auto diagram_context = diagram->CreateDefaultContext();
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  simulator.set_target_realtime_rate(1);
  simulator.Initialize();
  simulator.AdvanceTo(20);
  return 0;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main() { return drake::examples::planar_gripper::DoMain(); }
