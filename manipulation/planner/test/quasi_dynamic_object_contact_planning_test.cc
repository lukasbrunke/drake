#include "drake/manipulation/planner/quasi_dynamic_object_contact_planning.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/manipulation/planner/test/block_test_util.h"

namespace drake {
namespace manipulation {
namespace planner {
GTEST_TEST(QuasiDynamicObjectContactPlanningTest, TestInterpolation) {
  Block block;
  const int nT = 2;
  const double dt = 0.1;
  const int num_pushers = 0;
  QuasiDynamicObjectContactPlanning problem(
      nT, dt, block.mass(), block.I_B(), block.center_of_mass(), block.p_BV(),
      num_pushers, block.Q(), false);

  auto p_WB0_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), problem.p_WB()[0]);
  auto p_WB1_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), problem.p_WB()[1]);
  auto R_WB0_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), problem.R_WB()[0]);
  auto R_WB1_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), problem.R_WB()[1]);
  auto v_B0_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), problem.v_B().col(0));
  auto v_B1_constraint = problem.get_mutable_prog()->AddBoundingBoxConstraint(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), problem.v_B().col(0));
  auto omega_B0_constraint =
      problem.get_mutable_prog()->AddBoundingBoxConstraint(
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          problem.omega_B().col(0));
  auto omega_B1_constraint =
      problem.get_mutable_prog()->AddBoundingBoxConstraint(
          Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
          problem.omega_B().col(1));

  auto CheckFeasibility = [&p_WB0_constraint, &p_WB1_constraint,
                           &R_WB0_constraint, &R_WB1_constraint,
                           &v_B0_constraint, &v_B1_constraint,
                           &omega_B0_constraint, &omega_B1_constraint](
      solvers::MathematicalProgram* prog,
      const Eigen::Ref<const Eigen::Vector3d>& p_WB0,
      const Eigen::Ref<const Eigen::Vector3d>& p_WB1,
      const Eigen::Ref<const Eigen::Matrix3d>& R_WB0,
      const Eigen::Ref<const Eigen::Matrix3d>& R_WB1,
      const Eigen::Ref<const Eigen::Vector3d>& v_B0,
      const Eigen::Ref<const Eigen::Vector3d>& v_B1,
      const Eigen::Ref<const Eigen::Vector3d>& omega_B0,
      const Eigen::Ref<const Eigen::Vector3d>& omega_B1, bool feasible) {
    p_WB0_constraint.evaluator()->UpdateLowerBound(p_WB0);
    p_WB0_constraint.evaluator()->UpdateUpperBound(p_WB0);
    p_WB1_constraint.evaluator()->UpdateLowerBound(p_WB1);
    p_WB1_constraint.evaluator()->UpdateUpperBound(p_WB1);
    Eigen::Matrix<double, 9, 1> R_WB0_flat, R_WB1_flat;
    R_WB0_flat << R_WB0.col(0), R_WB0.col(1), R_WB0.col(2);
    R_WB1_flat << R_WB1.col(0), R_WB1.col(1), R_WB1.col(2);
    R_WB0_constraint.evaluator()->UpdateLowerBound(R_WB0_flat);
    R_WB0_constraint.evaluator()->UpdateUpperBound(R_WB0_flat);
    R_WB1_constraint.evaluator()->UpdateLowerBound(R_WB1_flat);
    R_WB1_constraint.evaluator()->UpdateUpperBound(R_WB1_flat);
  };
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
