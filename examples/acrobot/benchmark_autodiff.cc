#include <chrono>
#include "drake/common/find_resource.h"
#include "drake/common/eigen_types.h"
#include "drake/examples/acrobot/acrobot_plant.h"
#include "drake/math/autodiff.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/multibody/rigid_body_tree.h"

#include <gflags/gflags.h>

namespace drake {
namespace examples {
namespace acrobot {
DEFINE_int32(scalar_type, 0,
             "0 for double, 1 for AutoDiffXd, 2 for AutoDiffUpTo73d");
DEFINE_bool(use_acrobot_plant, false, "Set to true if using AcrobotPlant, otherwise use RigidBodyTree");

template <typename Derived>
void EvalRigidBodyTreeFun(const RigidBodyTreed& tree, Derived& cache) {
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

template<typename T>
void EvalAcrobotPlant(const VectorX<T>& q, const VectorX<T>& v) {
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

template <typename Derived, typename T>
void EvalFun(const RigidBodyTreed& tree, Derived& cache, const VectorX<T>& q, const VectorX<T>& v, bool use_acrobot_plant) {
  if (use_acrobot_plant) {
    EvalAcrobotPlant(q, v);
  } else {
    EvalRigidBodyTreeFun(tree, cache);
  }
}

int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
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
  switch (FLAGS_scalar_type) {
    case 0: {
      auto cache = tree->doKinematics(q, v, true);
      EvalFun(*tree, cache, q, v, FLAGS_use_acrobot_plant);
      break;
    }
    case 1: {
      const AutoDiffVecXd xd = math::initializeAutoDiff(x);
      const AutoDiffVecXd qd = xd.head<2>();
      const AutoDiffVecXd vd = xd.tail<2>();

      auto cache = tree->doKinematics(qd, vd, true);
      EvalFun(*tree, cache, q, v, FLAGS_use_acrobot_plant);
      break;
    }
    case 2: {
      VectorX<AutoDiffUpTo73d> xd(4);
      for (int i = 0; i < 4; ++i) {
        xd(i).value() = x(i);
        xd(i).derivatives().resize(4, 1);
        xd(i).derivatives() = Eigen::Vector4d::Zero();
        xd(i).derivatives()(i) = 1;
      }
      const VectorX<AutoDiffUpTo73d> qd = xd.head<2>();
      const VectorX<AutoDiffUpTo73d> vd = xd.tail<2>();

      auto cache = tree->doKinematics(qd, vd, true);
      EvalFun(*tree, cache, q, v, FLAGS_use_acrobot_plant);
      break;
    }
    default:
      throw std::logic_error("unknow scalar_type");
  }
  return 0;
}
}
}
}

int main(int argc, char* argv[]) {
  return drake::examples::acrobot::DoMain(argc, argv);
}
