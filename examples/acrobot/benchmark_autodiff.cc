#include <chrono>
#include "drake/common/eigen_types.h"
#include "drake/common/find_resource.h"
#include "drake/examples/acrobot/acrobot_plant.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/benchmarks/acrobot/make_acrobot_plant.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree.h"

#include <gflags/gflags.h>

using Eigen::Vector2d;
using drake::multibody::multibody_plant::MultibodyPlant;

namespace drake {
namespace examples {
namespace acrobot {
enum class ScalarType {
  kDouble,
  kAutoDiffXd,
  kAutoDiffUpTo73d,
};

std::string to_string(ScalarType type) {
  switch (type) {
    case ScalarType::kDouble:
      return "double";
    case ScalarType::kAutoDiffXd:
      return "AutoDiffXd";
    case ScalarType::kAutoDiffUpTo73d:
      return "AutoDiffUpTo73d";
  }
}

enum class RobotType {
  kRigidBodyTree,
  kMultibodyPlant,
  kAcrobotPlant,
};

std::string to_string(RobotType type) {
  switch (type) {
    case RobotType::kAcrobotPlant:
      return "AcrobotPlant";
    case RobotType::kRigidBodyTree:
      return "RigidBodyTree";
    case RobotType::kMultibodyPlant:
      return "MultibodyPlant";
  }
}

void GetAutoDiffXdQandV(const Eigen::Ref<const Eigen::VectorXd>& q,
                        const Eigen::Ref<const Eigen::VectorXd>& v,
                        AutoDiffVecXd* qd, AutoDiffVecXd* vd) {
  Eigen::Vector4d x;
  x << q, v;
  const AutoDiffVecXd xd = math::initializeAutoDiff(x);
  *qd = xd.head<2>();
  *vd = xd.tail<2>();
}

void GetAutoDiffUpTo73dQandV(const Eigen::Ref<const Vector2d>& q,
                             const Eigen::Ref<const Vector2d>& v,
                             VectorX<AutoDiffUpTo73d>* qd,
                             VectorX<AutoDiffUpTo73d>* vd) {
  Eigen::Vector4d x;
  x << q, v;
  VectorX<AutoDiffUpTo73d> xd(4);
  for (int i = 0; i < 4; ++i) {
    xd(i).value() = x(i);
    xd(i).derivatives().resize(4, 1);
    xd(i).derivatives() = Eigen::Vector4d::Zero();
    xd(i).derivatives()(i) = 1;
  }
  *qd = xd.head<2>();
  *vd = xd.tail<2>();
}

template <typename DerivedQ, typename DerivedV>
void EvalRigidBodyTreeGeneric(const RigidBodyTreed& tree,
                              const Eigen::MatrixBase<DerivedQ>& q,
                              const Eigen::MatrixBase<DerivedV>& v,
                              ScalarType scalar_type) {
  std::cout << "scalar type: " << to_string(scalar_type) << "\n";
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    auto cache = tree.doKinematics(q.eval(), v.eval());
    tree.massMatrix(cache);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Elapsed time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " microseconds" << std::endl;
}

template <typename T>
void EvalAcrobotPlantGeneric(const VectorX<T>& q, const VectorX<T>& v,
                             ScalarType scalar_type) {
  std::cout << "scalar type: " << to_string(scalar_type) << "\n";
  auto plant = std::make_unique<AcrobotPlant<T>>();

  auto context = plant->CreateDefaultContext();
  VectorX<T> x(q.rows() + v.rows());
  x << q, v;
  context->get_mutable_continuous_state_vector().SetFromVector(x);
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    plant->MassMatrix(*context);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Elapsed time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " microseconds" << std::endl;
}

template <typename Scalar, typename DerivedQ, typename DerivedV>
void EvalMultibodyPlantGeneric(
    const multibody::multibody_plant::MultibodyPlant<Scalar>& plant,
    const Eigen::MatrixBase<DerivedQ>& q, const Eigen::MatrixBase<DerivedV>& v,
    ScalarType scalar_type) {
  std::cout << "scalar type: " << to_string(scalar_type) << "\n";
  using T = typename DerivedQ::Scalar;
  auto context = plant.CreateDefaultContext();
  VectorX<T> x(q.rows() + v.rows());
  x << q, v;
  context->get_mutable_continuous_state_vector().SetFromVector(x);
  MatrixX<T> M(v.rows(), v.rows());
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    plant.model().CalcMassMatrixViaInverseDynamics(*context, &M);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Elapsed time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " microseconds" << std::endl;
}

int DoMain(int argc, char* argv[]) {
  auto tree = std::make_unique<RigidBodyTree<double>>();
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow("drake/examples/acrobot/Acrobot.urdf"),
      multibody::joints::kFixed, tree.get());
  auto plant = multibody::benchmarks::acrobot::MakeAcrobotPlant(
      multibody::benchmarks::acrobot::AcrobotParameters(), true);

  Eigen::VectorXd q(2);
  q << 1, 2;
  Eigen::VectorXd v(2);
  v << 3, 4;
  Eigen::VectorXd x(4);
  x << q, v;
  AutoDiffVecXd qd, vd;
  GetAutoDiffXdQandV(q, v, &qd, &vd);
  VectorX<AutoDiffUpTo73d> q73, v73;
  GetAutoDiffUpTo73dQandV(q, v, &q73, &v73);
  for (auto robot_type : {RobotType::kAcrobotPlant, RobotType::kRigidBodyTree,
                          RobotType::kMultibodyPlant}) {
    std::cout << "robot type: " << to_string(robot_type) << "\n";
    switch (robot_type) {
      case RobotType::kAcrobotPlant:
        EvalAcrobotPlantGeneric(q, v, ScalarType::kDouble);
        EvalAcrobotPlantGeneric(qd, vd, ScalarType::kAutoDiffXd);
        break;
      case RobotType::kRigidBodyTree:
        EvalRigidBodyTreeGeneric(*tree, q, v, ScalarType::kDouble);
        EvalRigidBodyTreeGeneric(*tree, qd, vd, ScalarType::kAutoDiffXd);
        EvalRigidBodyTreeGeneric(*tree, q73, v73, ScalarType::kAutoDiffUpTo73d);
        break;
      case RobotType::kMultibodyPlant:
        EvalMultibodyPlantGeneric(*plant, q, v, ScalarType::kDouble);
        EvalMultibodyPlantGeneric(*(dynamic_cast<MultibodyPlant<AutoDiffXd>*>(
                                      plant->ToAutoDiffXd().get())),
                                  qd, vd, ScalarType::kAutoDiffXd);
        break;
    }
    std::cout << "\n";
  }
  return 0;
}
}  // namespace acrobot
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::examples::acrobot::DoMain(argc, argv);
}
