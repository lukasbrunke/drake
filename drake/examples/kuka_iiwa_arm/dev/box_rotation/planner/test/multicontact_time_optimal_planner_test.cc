#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/planner/multicontact_time_optimal_planner.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/gurobi_solver.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
std::vector<ContactFacet> ConstructContactFacets(double box_size, double mu) {
  std::vector<ContactFacet> contact_facets;
  contact_facets.reserve(6);

  Eigen::Matrix<double, 3, 4> friction_cone_edges;
  friction_cone_edges << 1, 0, -1, 0, 0, 1, 0, -1, 1, 1, 1, 1;
  friction_cone_edges.topRows<2>() * mu;

  Eigen::Matrix<double, 3, 8> box_vertices;
  // clang-format off
  box_vertices << 1, 1, 1, 1, -1, -1, -1, -1,
      1, 1, -1, -1, 1, 1, -1, -1,
      1, -1, 1, -1, 1, -1, 1, -1;
  // clang-format on
  box_vertices *= box_size / 2;

  // Construct contact facets
  Eigen::Matrix<double, 3, 4> top_vertices;
  Eigen::Matrix<double, 3, 4> bottom_vertices;
  Eigen::Matrix<double, 3, 4> left_vertices;
  Eigen::Matrix<double, 3, 4> right_vertices;
  Eigen::Matrix<double, 3, 4> front_vertices;
  Eigen::Matrix<double, 3, 4> back_vertices;
  int top_vertex_count = 0;
  int bottom_vertex_count = 0;
  int left_vertex_count = 0;
  int right_vertex_count = 0;
  int front_vertex_count = 0;
  int back_vertex_count = 0;
  for (int i = 0; i < 8; ++i) {
    if (box_vertices(2, i) > 0) {
      top_vertices.col(top_vertex_count++) = box_vertices.col(i);
    } else {
      bottom_vertices.col(bottom_vertex_count++) = box_vertices.col(i);
    }
    if (box_vertices(0, i) > 0) {
      front_vertices.col(front_vertex_count++) = box_vertices.col(i);
    } else {
      back_vertices.col(back_vertex_count++) = box_vertices.col(i);
    }
    if (box_vertices(1, i) < 0) {
      left_vertices.col(left_vertex_count++) = box_vertices.col(i);
    } else {
      right_vertices.col(right_vertex_count++) = box_vertices.col(i);
    }
  }
  contact_facets.emplace_back(top_vertices, friction_cone_edges);
  contact_facets.emplace_back(
      bottom_vertices,
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix() *
          friction_cone_edges);
  contact_facets.emplace_back(
      front_vertices,
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY()).toRotationMatrix() *
          friction_cone_edges);
  contact_facets.emplace_back(
      back_vertices,
      Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitY()).toRotationMatrix() *
          friction_cone_edges);
  contact_facets.emplace_back(
      left_vertices,
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitX()).toRotationMatrix() *
          friction_cone_edges);
  contact_facets.emplace_back(
      right_vertices,
      Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitX()).toRotationMatrix() *
          friction_cone_edges);
  return contact_facets;
}

GTEST_TEST(TestContactFacet, WrenchConeEdgeTest) {
  double box_size = 0.5;
  double mu = 0.4;
  auto contact_facets = ConstructContactFacets(box_size, mu);
  for (int i = 0; i < 6; ++i) {
    auto wrench_cone_edges = contact_facets[i].CalcWrenchConeEdges();
    EXPECT_EQ(wrench_cone_edges.size(), 4);
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(wrench_cone_edges[j].cols(),
                contact_facets[i].NumFrictionConeEdges());
      for (int k = 0; k < contact_facets[i].NumFrictionConeEdges(); ++k) {
        Eigen::Matrix<double, 6, 1> wrench_expected;
        wrench_expected.topRows<3>() =
            contact_facets[i].friction_cone_edges().col(k);
        wrench_expected.bottomRows<3>() =
            contact_facets[i].vertices().col(j).cross(
                wrench_expected.topRows<3>());
        EXPECT_TRUE(CompareMatrices(wrench_cone_edges[j].col(k),
                                    wrench_expected, 1E-10,
                                    MatrixCompareType::absolute));
      }
    }
  }
}

Eigen::Matrix3d ConstructInertiaMatrix(double mass, double box_size) {
  Eigen::Matrix3d inertia =
      (mass / 6 * box_size * box_size * Eigen::Vector3d::Ones()).asDiagonal();
  return inertia;
}

class MultiContactTimeOptimalPlannerTest : public ::testing::Test {
 public:
  MultiContactTimeOptimalPlannerTest()
      : mass_(2),
        box_size_(0.5),
        mu_(0.4),
        nT_(15),
        num_arms_(3),
        planner_(mass_, ConstructInertiaMatrix(mass_, box_size_),
                 ConstructContactFacets(box_size_, mu_), nT_, num_arms_) {}

  const solvers::VectorXDecisionVariable& theta() const {return planner_.theta();}

  const Eigen::VectorXd& s() const {return planner_.s();}

 protected:
  double mass_;
  double box_size_;
  double mu_;
  int nT_;
  int num_arms_;
  MultiContactTimeOptimalPlanner planner_;
};

TEST_F(MultiContactTimeOptimalPlannerTest, TestRotation) {
  std::vector<Eigen::Isometry3d> box_poses(nT_);
  box_poses[0].translation() = Eigen::Vector3d::Zero();
  box_poses[nT_ - 1].translation() = Eigen::Vector3d(0, 0, 0.1);
  box_poses[0].linear().setIdentity();
  box_poses[nT_ - 1].linear() =
      (Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()))
          .toRotationMatrix();
  for (int i = 1; i < nT_ - 1; ++i) {
    box_poses[i].translation() =
        (box_poses[nT_ - 1].translation() - box_poses[0].translation()) /
        (nT_ - 1) * i;
    box_poses[i].linear() =
        Eigen::Quaterniond(box_poses[0].linear())
            .slerp(static_cast<double>(i) / (nT_ - 1),
                   Eigen::Quaterniond(box_poses[nT_ - 1].linear()))
            .toRotationMatrix();
  }
  planner_.SetObjectPoseSequence(box_poses);

  planner_.AddCost(planner_.t_bar().cast<symbolic::Expression>().sum());

  solvers::GurobiSolver gurobi_solver;
  if (gurobi_solver.available()) {
    planner_.SetSolverOption(solvers::GurobiSolver::id(), "OutputFlag", 1);
    solvers::SolutionResult sol_result = gurobi_solver.Solve(planner_);

    EXPECT_EQ(sol_result, solvers::SolutionResult::kSolutionFound);

    auto t_bar_sol = planner_.GetSolution(planner_.t_bar());
    auto theta_sol = planner_.GetSolution(theta());
    EXPECT_TRUE((theta_sol.array() >= 0).all());
    // Now check if t_bar is equal to the interval.
    Eigen::VectorXd delta_t_expected =
        (2 * (s().bottomRows(nT_ - 1) - s().topRows(nT_ - 1)).array() /
         (theta_sol.topRows(nT_ - 1).array().sqrt() +
          theta_sol.bottomRows(nT_ - 1).array().sqrt()))
            .matrix();
    EXPECT_TRUE(CompareMatrices(delta_t_expected, t_bar_sol, 1E-2,
                                MatrixCompareType::absolute));
  }
}
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
