#include <chrono>
#include "drake/common/eigen_types.h"
#include "drake/common/find_resource.h"
#include "drake/examples/acrobot/acrobot_plant.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree.h"

#include <gflags/gflags.h>

using Eigen::Vector2d;

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
void EvalRigidBodyTreeFunGeneric(const RigidBodyTreed& tree,
                                 const Eigen::MatrixBase<DerivedQ>& q,
                                 const Eigen::MatrixBase<DerivedV>& v) {
  auto cache = tree.doKinematics(q.eval(), v.eval());
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    tree.massMatrix(cache);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "Elapsed time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " microseconds" << std::endl;
}

template <typename T>
void EvalAcrobotPlantGeneric(const VectorX<T>& q, const VectorX<T>& v) {
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

void EvalAcrobotPlant(const Eigen::Ref<const Eigen::Vector2d>& q,
                      const Eigen::Ref<const Eigen::Vector2d>& v) {
  for (auto scalar_type : {ScalarType::kDouble, ScalarType::kAutoDiffXd,
                           ScalarType::kAutoDiffUpTo73d}) {
    std::cout << "scalar type: " << to_string(scalar_type) << "\n";
    switch (scalar_type) {
      case ScalarType::kDouble: {
        EvalAcrobotPlantGeneric<double>(q, v);
        break;
      }
      case ScalarType::kAutoDiffXd: {
        AutoDiffVecXd qd, vd;
        GetAutoDiffXdQandV(q, v, &qd, &vd);
        EvalAcrobotPlantGeneric<AutoDiffXd>(qd, vd);
        break;
      }
      case ScalarType::kAutoDiffUpTo73d: {
        VectorX<AutoDiffUpTo73d> qd, vd;
        GetAutoDiffUpTo73dQandV(q, v, &qd, &vd);
        EvalAcrobotPlantGeneric<AutoDiffUpTo73d>(qd, vd);
        break;
      }
    }
  }
}

void EvalRigidBodyTreeFun(const RigidBodyTreed& tree,
                          const Eigen::Ref<const Eigen::VectorXd>& q,
                          const Eigen::Ref<const Eigen::VectorXd>& v) {
  for (auto scalar_type : {ScalarType::kDouble, ScalarType::kAutoDiffXd,
                           ScalarType::kAutoDiffUpTo73d}) {
    std::cout << "scalar type: " << to_string(scalar_type) << "\n";
    switch (scalar_type) {
      case ScalarType::kDouble: {
        EvalRigidBodyTreeFunGeneric(tree, q, v);
        break;
      }
      case ScalarType::kAutoDiffXd: {
        AutoDiffVecXd qd, vd;
        GetAutoDiffXdQandV(q, v, &qd, &vd);
        EvalRigidBodyTreeFunGeneric(tree, qd, vd);
        break;
      }
      case ScalarType::kAutoDiffUpTo73d: {
        VectorX<AutoDiffUpTo73d> qd, vd;
        GetAutoDiffUpTo73dQandV(q, v, &qd, &vd);
        EvalRigidBodyTreeFunGeneric(tree, qd, vd);
        break;
      }
    }
  }
}

int DoMain(int argc, char* argv[]) {
  auto tree = std::make_unique<RigidBodyTree<double>>();
  parsers::urdf::AddModelInstanceFromUrdfFileToWorld(
      FindResourceOrThrow("drake/examples/acrobot/Acrobot.urdf"),
      multibody::joints::kFixed, tree.get());

  Eigen::VectorXd q(2);
  q << 1, 2;
  Eigen::VectorXd v(2);
  v << 3, 4;
  Eigen::VectorXd x(4);
  x << q, v;
  for (auto robot_type :
       {RobotType::kAcrobotPlant, RobotType::kRigidBodyTree}) {
    std::cout << "robot type: " << to_string(robot_type) << "\n";
    switch (robot_type) {
      case RobotType::kAcrobotPlant:
        EvalAcrobotPlant(q, v);
        break;
      case RobotType::kRigidBodyTree:
        EvalRigidBodyTreeFun(*tree, q, v);
        break;
      case RobotType::kMultibodyPlant:
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
