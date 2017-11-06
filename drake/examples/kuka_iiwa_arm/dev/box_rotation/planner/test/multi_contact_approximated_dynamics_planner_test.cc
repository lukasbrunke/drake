#include "drake/examples/kuka_iiwa_arm/dev/box_rotation/planner/multi_contact_approximated_dynamics_planner.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace box_rotation {
namespace {
std::vector<ContactFacet> ConstructContactFacets(double box_size, double mu) {
  std::vector<ContactFacet> contact_facets;
  contact_facets.reserve(6);

  constexpr int num_edges_per_cone = 5;
  Eigen::Matrix<double, 3, num_edges_per_cone> friction_cone_edges;
  Eigen::Matrix<double, 1, num_edges_per_cone> cone_theta =
      Eigen::Matrix<double, 1, num_edges_per_cone + 1>::LinSpaced(
          num_edges_per_cone + 1, 0, 2 * M_PI)
          .leftCols<num_edges_per_cone>();
  friction_cone_edges.row(0) = cone_theta.array().cos().matrix();
  friction_cone_edges.row(1) = cone_theta.array().sin().matrix();
  friction_cone_edges.row(2) =
      Eigen::Matrix<double, 1, num_edges_per_cone>::Ones();
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

class MultiContactApproximatedDynamicsPlannerTest : public ::testing::Test {
 public:
  MultiContactApproximatedDynamicsPlannerTest()
      : planner_(2, ConstructInertiaMatrix(2, 0.5),
                 ConstructContactFacets(0.5, 0.4), 10, 3) {}

  void GenerateBoxTrajectory(
      const std::vector<Eigen::MatrixXd>& contact_wrench_weight,
      const Eigen::Isometry3d& box_pose0,
      const Eigen::Ref<const Eigen::Vector3d>& com_vel0,
      const Eigen::Ref<const Eigen::Vector3d>& omega_BpB0,
      const Eigen::Ref<const Eigen::Vector3d>& omega_dot_BpB0) {
    DRAKE_ASSERT(contact_wrench_weight.size() ==
                 planner_.contact_facets().size());
    for (int i = 0; i < static_cast<int>(planner_.contact_facets().size());
         ++i) {
      planner_.AddBoundingBoxConstraint(contact_wrench_weight[i],
                                        contact_wrench_weight[i],
                                        planner_.contact_wrench_weight_[i]);
    }
    const auto& contact_facets = planner_.contact_facets();
    for (int i = 0; i < planner_.nT(); ++i) {
      Vector6<double> contact_wrench = Vector6<double>::Zero();
      for (int j = 0; j < static_cast<int>(contact_facets.size()); ++j) {
        for (int k = 0; k < contact_facets[j].NumVertices(); ++k) {
          contact_wrench += contact_facets[j].CalcWrenchConeEdges()[k] *
                            contact_wrench_weight[j].block(
                                k * contact_facets[j].NumFrictionConeEdges(), i,
                                contact_facets[j].NumFrictionConeEdges(), 1);
        }
      }
      planner_.AddBoundingBoxConstraint(contact_wrench, contact_wrench,
                                        planner_.total_contact_wrench_.col(i));
    }
  }

  void CheckConstructor(
      const std::vector<Eigen::MatrixXd>& contact_wrench_weight,
      const Eigen::Isometry3d& box_pose0,
      const Eigen::Ref<const Eigen::Vector3d>& com_vel0,
      const Eigen::Ref<const Eigen::Vector3d>& omega_BpB0,
      const Eigen::Ref<const Eigen::Vector3d>& omega_dot_BpB0) {
    // For each time sample, there are 3 non-convex quadratic constraint on the
    // dynamics, coming from R_WB * total_contact_force
    const int num_non_convex_quadratic_constraints = 3 * planner_.nT();
    EXPECT_EQ(planner_.non_convex_quadratic_constraints_.size(),
              num_non_convex_quadratic_constraints);
    GenerateBoxTrajectory(contact_wrench_weight, box_pose0, com_vel0,
                          omega_BpB0, omega_dot_BpB0);
  }

 protected:
  MultiContactApproximatedDynamicsPlanner planner_;
};

void SetActiveContactWrenchWeight(
    std::vector<Eigen::MatrixXd>* contact_wrench_weight, int time_index,
    const std::vector<int>& active_facets) {
  for (int facet_idx : active_facets) {
    contact_wrench_weight->at(facet_idx).col(time_index) =
        Eigen::VectorXd::Random(contact_wrench_weight->at(facet_idx).rows())
            .array()
            .abs()
            .matrix();
  }
}

TEST_F(MultiContactApproximatedDynamicsPlannerTest, TestConstruction) {
  // Test if the total contact wrench at each time sample is computed correctly,
  // from the weights of each friction cone edges.
  std::vector<Eigen::MatrixXd> contact_wrench_weight(
      planner_.contact_facets().size());

  for (int i = 0; i < static_cast<int>(planner_.contact_facets().size()); ++i) {
    contact_wrench_weight[i].resize(
        planner_.contact_facets()[i].NumFrictionConeEdges() *
            planner_.contact_facets()[i].NumVertices(),
        planner_.nT());
    contact_wrench_weight[i].setZero();
  }
  SetActiveContactWrenchWeight(&contact_wrench_weight, 0, {0, 1, 3});
  const Eigen::Isometry3d box_pose0 = Eigen::Isometry3d::Identity();
  const Eigen::Vector3d com_vel0(0.1, 0.2, 0.3);
  const Eigen::Vector3d omega_BpB0(0.3, 0.2, 1);
  const Eigen::Vector3d omega_dot_BpB0(0.1, -0.2, 2);
  CheckConstructor(contact_wrench_weight, box_pose0, com_vel0, omega_BpB0,
                   omega_dot_BpB0);
}
}  // namespace
}  // namespace box_rotation
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake
