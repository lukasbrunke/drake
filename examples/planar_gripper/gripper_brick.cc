#include "drake/examples/planar_gripper/gripper_brick.h"

#include "drake/common/find_resource.h"
#include "drake/examples/planar_gripper/planar_gripper_common.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace examples {
namespace planar_gripper {

std::string to_string(Finger finger) {
  switch (finger) {
    case Finger::kFinger1: {
      return "finger 1";
    }
    case Finger::kFinger2: {
      return "finger 2";
    }
    case Finger::kFinger3: {
      return "finger 3";
    }
    default:
      throw std::runtime_error("Finger not valid.");
  }
}

template <typename T>
void AddDrakeVisualizer(systems::DiagramBuilder<T>*,
                        const geometry::SceneGraph<T>&) {
  // Disabling visualization for non-double scalar type T.
}

template <>
void AddDrakeVisualizer<double>(
    systems::DiagramBuilder<double>* builder,
    const geometry::SceneGraph<double>& scene_graph) {
  geometry::ConnectDrakeVisualizer(builder, scene_graph);
}

template <typename T>
void InitializeDiagramSimulator(const systems::Diagram<T>&) {}

template <>
void InitializeDiagramSimulator<double>(
    const systems::Diagram<double>& diagram) {
  systems::Simulator<double>(diagram).Initialize();
}

template <typename T>
std::unique_ptr<systems::Diagram<T>> ConstructDiagram(
    multibody::MultibodyPlant<T>** plant, geometry::SceneGraph<T>** scene_graph,
    std::array<double, 3>* theta_base,
    std::array<Eigen::Vector2d, 3>* p_WFbase) {
  systems::DiagramBuilder<T> builder;
  std::tie(*plant, *scene_graph) =
      multibody::AddMultibodyPlantSceneGraph(&builder);
  const std::string gripper_path =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_gripper.sdf");
  multibody::Parser parser(*plant, *scene_graph);
  parser.AddModelFromFile(gripper_path, "gripper");
  const std::array<math::RigidTransformd, 3> X_WF =
      examples::planar_gripper::WeldGripperFrames(*plant);
  for (int i = 0; i < 3; ++i) {
    theta_base->at(i) =
        std::atan2(X_WF[i].GetAsMatrix4()(1, 1), X_WF[i].GetAsMatrix4()(0, 0));
    p_WFbase->at(i) = X_WF[i].translation().tail<2>();
  }
  const std::string brick_path =
      FindResourceOrThrow("drake/examples/planar_gripper/planar_brick.sdf");
  parser.AddModelFromFile(brick_path, "brick");
  (*plant)->WeldFrames((*plant)->world_frame(),
                       (*plant)->GetFrameByName("brick_base"),
                       math::RigidTransformd());

  (*plant)->Finalize();

  AddDrakeVisualizer<T>(&builder, **scene_graph);
  return builder.Build();
}

template <typename T>
GripperBrickHelper<T>::GripperBrickHelper() {
  diagram_ =
      ConstructDiagram<T>(&plant_, &scene_graph_, &theta_base_, &p_WFbase_);
  InitializeDiagramSimulator(*diagram_);

  const geometry::SceneGraphInspector<T>& inspector =
      scene_graph_->model_inspector();
  for (int i = 0; i < 3; ++i) {
    finger_tip_sphere_geometry_ids_[i] = inspector.GetGeometryIdByName(
        plant_->GetBodyFrameIdOrThrow(
            plant_->GetBodyByName("finger" + std::to_string(i + 1) + "_link2")
                .index()),
        geometry::Role::kProximity, "gripper::link2_pad_collision");
  }
  const geometry::Shape& fingertip_shape =
      inspector.GetShape(finger_tip_sphere_geometry_ids_[0]);
  finger_tip_radius_ =
      dynamic_cast<const geometry::Sphere&>(fingertip_shape).get_radius();
  p_L2Fingertip_ = inspector.GetPoseInFrame(finger_tip_sphere_geometry_ids_[0])
                       .translation();
  const geometry::Shape& brick_shape =
      inspector.GetShape(inspector.GetGeometryIdByName(
          plant_->GetBodyFrameIdOrThrow(
              plant_->GetBodyByName("brick_link").index()),
          geometry::Role::kProximity, "brick::box_collision"));
  brick_size_ = dynamic_cast<const geometry::Box&>(brick_shape).size();

  for (int i = 0; i < 3; ++i) {
    finger_base_position_indices_[i] =
        plant_->GetJointByName("finger" + std::to_string(i + 1) + "_BaseJoint")
            .position_start();
    finger_mid_position_indices_[i] =
        plant_->GetJointByName("finger" + std::to_string(i + 1) + "_MidJoint")
            .position_start();
    finger_link1_frames_[i] =
        &(plant_->GetFrameByName("finger" + std::to_string(i + 1) + "_link1"));
    finger_link2_frames_[i] =
        &(plant_->GetFrameByName("finger" + std::to_string(i + 1) + "_link2"));
  }
  brick_translate_y_position_index_ =
      plant_->GetJointByName("brick_translate_y_joint").position_start();
  brick_translate_z_position_index_ =
      plant_->GetJointByName("brick_translate_z_joint").position_start();
  brick_revolute_x_position_index_ =
      plant_->GetJointByName("brick_revolute_x_joint").position_start();
  brick_frame_ = &(plant_->GetFrameByName("brick_link"));
}

template <typename T>
const multibody::Frame<double>& GripperBrickHelper<T>::finger_link1_frame(
    Finger finger) const {
  switch (finger) {
    case Finger::kFinger1: {
      return *(finger_link1_frames_[0]);
    }
    case Finger::kFinger2: {
      return *(finger_link1_frames_[1]);
    }
    case Finger::kFinger3: {
      return *(finger_link1_frames_[2]);
    }
    default:
      throw std::invalid_argument("finger_link1_frame(), unknown finger.");
  }
}

template <typename T>
const multibody::Frame<double>& GripperBrickHelper<T>::finger_link2_frame(
    Finger finger) const {
  switch (finger) {
    case Finger::kFinger1: {
      return *(finger_link2_frames_[0]);
    }
    case Finger::kFinger2: {
      return *(finger_link2_frames_[1]);
    }
    case Finger::kFinger3: {
      return *(finger_link2_frames_[2]);
    }
    default:
      throw std::invalid_argument("finger_link2_frame(), unknown finger.");
  }
}

template <typename T>
int GripperBrickHelper<T>::finger_base_position_index(Finger finger) const {
  switch (finger) {
    case Finger::kFinger1:
      return finger_base_position_indices_[0];
    case Finger::kFinger2:
      return finger_base_position_indices_[1];
    case Finger::kFinger3:
      return finger_base_position_indices_[2];
    default:
      throw std::invalid_argument(
          "finger_base_position_index(): unknown finger");
  }
}

template <typename T>
int GripperBrickHelper<T>::finger_mid_position_index(Finger finger) const {
  switch (finger) {
    case Finger::kFinger1:
      return finger_mid_position_indices_[0];
    case Finger::kFinger2:
      return finger_mid_position_indices_[1];
    case Finger::kFinger3:
      return finger_mid_position_indices_[2];
    default:
      throw std::invalid_argument(
          "finger_mid_position_index(): unknown finger");
  }
}

template <typename T>
geometry::GeometryId GripperBrickHelper<T>::finger_tip_sphere_geometry_id(
    Finger finger) const {
  switch (finger) {
    case Finger::kFinger1: {
      return finger_tip_sphere_geometry_ids_[0];
    }
    case Finger::kFinger2: {
      return finger_tip_sphere_geometry_ids_[1];
    }
    case Finger::kFinger3: {
      return finger_tip_sphere_geometry_ids_[2];
    }
    default: {
      throw std::invalid_argument(
          "finger_tip_sphere_geometry_id(): unknown finger.");
    }
  }
}

template <typename T>
double GripperBrickHelper<T>::FingerBaseOrientation(Finger finger) const {
  // This returns the yaw angle to weld the finger base. Keep the values
  // synchronized with WeldGripperFrames() in planar_gripper_common.h. The
  // test in gripper_brick_test.cc guarantees that base_theta is synchronized.
  switch (finger) {
    case Finger::kFinger1: {
      return theta_base_[0];
    }
    case Finger::kFinger2: {
      return theta_base_[1];
    }
    case Finger::kFinger3: {
      return theta_base_[2];
    }
    default: {
      throw std::runtime_error(
          "GripperBrickHelper::FingerBaseOrientation: unknown finger.");
    }
  }
}

template <typename T>
Eigen::Vector2d GripperBrickHelper<T>::GetFingerBasePositionInWorldFrame(
    Finger finger) const {
  switch (finger) {
    case Finger::kFinger1: {
      return p_WFbase_[0];
    }
    case Finger::kFinger2: {
      return p_WFbase_[1];
    }
    case Finger::kFinger3: {
      return p_WFbase_[2];
    }
    default: {
      throw std::runtime_error(
          "GripperBrickHelper::GetFingerBasePositionInWorldFrame: unknown "
          "finger.");
    }
  }
}

template <typename T>
template <typename U>
Matrix2X<U>
GripperBrickHelper<T>::CalcFingerTipSpherePositionJacobianInBrickFrame(
    Finger finger, const Eigen::Ref<const VectorX<U>>& q) const {
  using std::cos;
  using std::cos;
  // The position of the finger tip in the brick frame is
  // R(θ_brick)ᵀ * (p_WFbase + l1 * [cos(θ_base + θ1); sin(θ_base + θ1)] + l2 *
  // [cos(θ_base + θ1 + θ2); sin(θ_base + θ1 + θ2)] - p_WBrick)
  Matrix2X<U> J(2, q.rows());
  const U theta_base_1 =
      FingerBaseOrientation(finger) + q(finger_base_position_index(finger));
  const U theta_base_1_2 = theta_base_1 + q(finger_mid_position_index(finger));
  const U cos_base_1 = cos(theta_base_1);
  const U sin_base_1 = sin(theta_base_1);
  const U cos_base_1_2 = cos(theta_base_1_2);
  const U sin_base_1_2 = sin(theta_base_1_2);
  const U cos_brick = cos(q(brick_revolute_x_position_index()));
  const U sin_brick = sin(q(brick_revolute_x_position_index()));
}

// Explicit instantiation
template class GripperBrickHelper<double>;
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
