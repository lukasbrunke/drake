#include "drake/manipulation/planner/quasi_dynamic_object_contact_planning.h"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/manipulation/planner/test/block_test_util.h"
#include "drake/solvers/gurobi_solver.h"

using drake::solvers::MatrixDecisionVariable;
using drake::solvers::VectorDecisionVariable;
using drake::symbolic::Expression;

namespace drake {
namespace manipulation {
namespace planner {
namespace {
class QuasiDynamicBlockFlipTest
    : public ::testing::TestWithParam<std::tuple<int, int, double>> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuasiDynamicBlockFlipTest)

  QuasiDynamicBlockFlipTest()
      : block_(),
        num_pushers_(std::get<0>(GetParam())),
        nT_(std::get<1>(GetParam())),
        dt_(std::get<2>(GetParam())),
        problem_(nT_, dt_, block_.mass(), block_.I_B(), block_.center_of_mass(),
                 block_.p_BV(), num_pushers_, block_.Q(), 0.5, M_PI, false),
        f_WV_(nT_) {
    SetUpBlockFlipTest(block_, num_pushers_, nT_, &problem_, &f_WV_);
    problem_.AddQuasiDynamicConstraint();
  }

  ~QuasiDynamicBlockFlipTest() = default;

 protected:
  Block block_;
  int num_pushers_;
  int nT_;
  double dt_;
  QuasiDynamicObjectContactPlanning problem_;
  std::vector<MatrixDecisionVariable<3, Eigen::Dynamic>> f_WV_;
};
}  // namespace
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
