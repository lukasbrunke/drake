#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/finger_brick.h"
#include "drake/examples/planar_gripper/finger_brick_control.h"
#include "drake/examples/planar_gripper/planar_finger_qp.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcmt_plant_state.hpp"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/spatial_forces_to_lcm.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/constant_vector_source.h"

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 1.5,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 1e-4,
              "If greater than zero, the plant is modeled as a system with "
              "discrete updates and period equal to this time_step. "
              "If 0, the plant is modeled as a continuous system.");
DEFINE_double(theta_desired, 0, "The desired angle of the brick.");
DEFINE_double(
    gravity_accel, -9.81,
    "The acceleration due to gravity. Is only applied if is_vertical is true.");
DEFINE_bool(is_vertical, false, "If true, gravity acts along the z axis.");
DEFINE_bool(brick_only, false, "Only simulate brick (no finger).");
DEFINE_double(Kp, 60 /* 50 */, "QP controller Kp gain");
DEFINE_double(Kd, 0 /* 5 */, "QP controller Kd gain");
DEFINE_double(weight_thetaddot_error, 1, "thetaddot error weight.");
DEFINE_double(weight_f_Cb_B, 1, "Contact force magnitued penalty weight");
DEFINE_string(output_spatial_force_channel, "QP_SPATIAL_FORCE_OUTPUT",
              "Name of the LCM channel for the output spatial forces");
DEFINE_string(state_channel, "PLANT_STATE",
              "The LCM channel name for the plant states.");
namespace drake {
namespace examples {
namespace planar_gripper {
int DoMain() {
  // Construct a diagram with a QP controller. This diagram takes in the current
  // state as the input (from LCM), and outputs the desired contact force (to
  // LCM).

  systems::DiagramBuilder<double> builder;
  geometry::SceneGraph<double>& scene_graph =
      *builder.AddSystem<geometry::SceneGraph>();
  scene_graph.set_name("scene_graph");

  // Make and add the planar_gripper model.
  const std::string full_name =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_finger.sdf");
  MultibodyPlant<double>& plant =
      *builder.AddSystem<multibody::MultibodyPlant>(FLAGS_time_step);
  multibody::Parser(&plant, &scene_graph).AddModelFromFile(full_name);
  if (FLAGS_brick_only) {
    WeldFingerFrame<double>(&plant, -1);
  } else {
    WeldFingerFrame<double>(&plant);
  }

  //   Adds the object to be manipulated.
  auto object_file_name =
      FindResourceOrThrow("drake/examples/planar_gripper/1dof_brick.sdf");
  multibody::Parser(&plant, &scene_graph)
      .AddModelFromFile(object_file_name, "object");

  // Add gravity
  Vector3<double> gravity(0, 0, 0);
  if (FLAGS_is_vertical) {
    gravity(2) = FLAGS_gravity_accel;
  }
  plant.mutable_gravity_field().set_gravity_vector(gravity);

  // Now the model is complete.
  plant.Finalize();

  const double finger_brick_mu =
      GetFingerBrickFriction(plant, scene_graph).static_friction();
  const double finger_tip_radius = GetFingerTipSphereRadius(plant, scene_graph);
  const double damping =
      plant.GetJointByName<multibody::RevoluteJoint>("brick_pin_joint")
          .damping();
  const double I_B = dynamic_cast<const multibody::RigidBody<double>&>(
                         plant.GetFrameByName("brick_base").body())
                         .default_rotational_inertia()
                         .get_moments()(0);

  auto qp_controller = builder.AddSystem<PlanarFingerInstantaneousQPController>(
      &plant, FLAGS_Kp, FLAGS_Kd, FLAGS_weight_thetaddot_error,
      FLAGS_weight_f_Cb_B, finger_brick_mu, finger_tip_radius, damping, I_B);

  lcm::DrakeLcm lcm;
  // Connect planned thetaddot source to qp controller.
  auto thetaddot_planned_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(Vector1d(0));
  builder.Connect(thetaddot_planned_source->get_output_port(),
                  qp_controller->get_input_port_desired_thetaddot());

  // Connect desired theta_state source to qp controller.
  auto theta_traj_source =
      builder.AddSystem<systems::ConstantVectorSource<double>>(
          Eigen::Vector2d(FLAGS_theta_desired, 0));
  builder.Connect(theta_traj_source->get_output_port(),
                  qp_controller->get_input_port_desired_state());

  // Add an LCM subscriber for the system state.
  auto state_lcm_subscriber = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<lcmt_plant_state>(
          FLAGS_state_channel, &lcm));
  auto spatial_forces_to_lcm =
      builder.AddSystem<multibody::SpatialForcesToLcmSystem<double>>(plant);
  spatial_forces_to_lcm->set_name("qp_spatial_forces_to_lcm");

  auto spatial_forces_publisher = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_spatial_forces_for_viz>(
          FLAGS_output_spatial_force_channel, &lcm,
          1.0 / 60 /* publish period */));
  spatial_forces_publisher->set_name("qp_spatial_forces_publisher");

  builder.Connect(qp_controller->get_output_port_control(),
                  spatial_forces_to_lcm->get_input_port(0));
  builder.Connect(spatial_forces_to_lcm->get_output_port(0),
                  spatial_forces_publisher->get_input_port());

  // Connect MBP snd SG.
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));
  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());

  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

  simulator.set_publish_every_time_step(false);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);
  return 0;
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage("A simple planar gripper example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::planar_gripper::DoMain();
}
