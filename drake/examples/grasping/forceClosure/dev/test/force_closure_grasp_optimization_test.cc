#include "drake/examples/grasping/forceClosure/dev/force_closure_grasp_optimization.h"

#include <gtest/gtest.h>
#include <drake/solvers/gurobi_solver.h>

#include "drake/common/eigen_matrix_compare.h"

namespace drake {
namespace examples {
namespace grasping {
namespace forceClosure {
namespace test {
class TestBoxGrasp : public ::testing::Test {
 public:
  TestBoxGrasp() {
    Eigen::Matrix<double, 3, 4> facet_vertices;
    int dirs[2] = {-1, 1};
    for (int dir : dirs) {
      for (int axis = 0; axis < 3; ++axis) {
        int axis1 = (axis + 1) % 3;
        int axis2 = (axis + 2) % 3;
        facet_vertices.row(axis) = box_len_[axis] / 2 * dir * Eigen::RowVector4d::Ones();
        facet_vertices.row(axis1) =
            box_len_[axis1] / 2 * dir * Eigen::RowVector4d(1, 1, -1, -1);
        facet_vertices.row(axis2) =
            box_len_[axis2] / 2 * dir * Eigen::RowVector4d(1, -1, 1, -1);

        Eigen::Vector3d facet_normal(0, 0, 0);
        facet_normal(axis) = dir;
        box_facets_.emplace_back(facet_vertices, facet_normal);
      }
    }
  }

  bool pt_on_facet(const Eigen::Vector3d& pt, int facet_idx, double tol = 1E-6) const {
    for (int i = 0; i < 3; ++i) {
      if (pt(i) < -box_len_[i] / 2 -tol || pt(i) > box_len_[i] / 2 + tol) {
        return false;
      }
    }
    return std::abs(box_facets_[facet_idx].facet_normal().dot(pt) - box_facets_[facet_idx].offset()) < tol;
  }

 protected:
  double box_len_[3] = {0.2, 0.3, 0.4};
  std::vector<ContactFacet> box_facets_;
};

TEST_F(TestBoxGrasp, TestPolytopeSurface) {
  std::vector<std::vector<int>> facet_idx_per_contact;
  facet_idx_per_contact.push_back({0, 1, 2});
  facet_idx_per_contact.push_back({1, 2, 3, 4});
  facet_idx_per_contact.push_back({5, 0, 2});

  ForceClosureGraspOptimization grasp_optimization(3, box_facets_, facet_idx_per_contact, ForceClosureMetric::kQ1Norm);

  auto prog = grasp_optimization.prog();

  solvers::GurobiSolver gurobi_solver;
  solvers::SolutionResult result = gurobi_solver.Solve(*prog);
  EXPECT_EQ(result, solvers::SolutionResult::kSolutionFound);

  auto contact_on_facet = grasp_optimization.contact_on_facet_value();
  auto facet_vertices_weights = grasp_optimization.facet_vertices_weights_value();
  auto contact_pos = grasp_optimization.contact_pos(contact_on_facet, facet_vertices_weights);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(contact_on_facet[i].sum(), 1, 1E-6);
    std::cout<<"contact " << i << " : " << contact_pos.col(i).transpose() << std::endl;
    for (int j = 0; j < static_cast<int>(facet_idx_per_contact[i].size()); ++j) {
      int facet_idx = facet_idx_per_contact[i][j];
      std::cout << "contact " << i << " on facet " << facet_idx << " : " << contact_on_facet[i](j) << std::endl;
      EXPECT_NEAR(facet_vertices_weights[i][j].sum(), contact_on_facet[i](j), 1E-6);
      std::cout << "vertices weights for contact " << i << " on facet " << facet_idx << " : " << facet_vertices_weights[i][j].transpose() << std::endl;
      if (contact_on_facet[i](j) > 1 - 1E-5) {
        EXPECT_TRUE(pt_on_facet(contact_pos.col(i), facet_idx));
      }
    }
  }
}
}
}
}
}
}