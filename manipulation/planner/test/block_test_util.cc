#include "drake/manipulation/planner/test/block_test_util.h"

#include "drake/manipulation/planner/friction_cone.h"
namespace drake {
namespace manipulation {
namespace planner {
Block::Block() {
  // clang-format off
    p_BV_ << 1, 1, 1, 1, -1, -1, -1, -1,
             1, 1, -1, -1, 1, 1, -1, -1,
             1, -1, 1, -1, 1, -1, 1, -1;
  // clang-format on
  p_BV_.topRows<2>() *= width() / 2;
  p_BV_.row(2) *= height() / 2;

  // Add body contact point at the middle of each block edge, and the center
  // of each facet.
  Q_.reserve(18);

  auto AddFacetCenterContactPoint = [this](
      const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
      const Eigen::Ref<const Eigen::Vector3d>& n_B) {
    this->Q_.emplace_back(
        p_BQ, GenerateLinearizedFrictionConeEdges<6>(n_B, this->mu()));
  };
  // center of top facet
  AddFacetCenterContactPoint(Eigen::Vector3d(0, 0, height() / 2),
                             -Eigen::Vector3d::UnitZ());
  // center of bottom facet
  AddFacetCenterContactPoint(Eigen::Vector3d(0, 0, -height() / 2),
                             Eigen::Vector3d::UnitZ());
  // center of the positive x facet
  AddFacetCenterContactPoint(Eigen::Vector3d(width() / 2, 0, 0),
                             -Eigen::Vector3d::UnitX());
  // center of the negative x facet
  AddFacetCenterContactPoint(Eigen::Vector3d(-width() / 2, 0, 0),
                             Eigen::Vector3d::UnitX());
  // center of the positive y facet
  AddFacetCenterContactPoint(Eigen::Vector3d(0, width() / 2, 0),
                             -Eigen::Vector3d::UnitY());
  // center of the negative y facet
  AddFacetCenterContactPoint(Eigen::Vector3d(0, -width() / 2, 0),
                             Eigen::Vector3d::UnitY());

  auto AddEdgeMiddleContactPoint = [this](
      const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
      const Eigen::Ref<const Eigen::Vector3d>& n1_B,
      const Eigen::Ref<const Eigen::Vector3d>& n2_B) {
    const auto e1_B = GenerateLinearizedFrictionConeEdges<6>(n1_B, this->mu());
    const auto e2_B = GenerateLinearizedFrictionConeEdges<6>(n2_B, this->mu());
    // TODO(hongkai.dai): find the convex hull of (e1_B, e2_B)
    Eigen::Matrix<double, 3, 12> e_B;
    e_B.leftCols<6>() = e1_B;
    e_B.rightCols<6>() = e2_B;
    this->Q_.emplace_back(p_BQ, e_B);
  };

  // The order is
  // 6:  0 + +
  // 7:  0 + -
  // 8:  0 - +
  // 9:  0 - -
  // 10: + 0 +
  // 11: - 0 +
  // 12: + 0 -
  // 13: - 0 -
  // 14: + + 0
  // 15: + - 0
  // 16: - + 0
  // 17: - - 0
  for (int dim0 = 0; dim0 < 3; ++dim0) {
    const int dim1 = (dim0 + 1) % 3;
    const int dim2 = (dim1 + 1) % 3;
    for (double multiplier1 : {1, -1}) {
      for (double multiplier2 : {1, -1}) {
        Eigen::Vector3d p_BQ;
        p_BQ(dim0) = 0;
        p_BQ(dim1) = dimension()(dim1) / 2 * multiplier1;
        p_BQ(dim2) = dimension()(dim2) / 2 * multiplier2;
        Eigen::Vector3d n1_B = Eigen::Vector3d::Zero();
        n1_B(dim1) = -multiplier1;
        Eigen::Vector3d n2_B = Eigen::Vector3d::Zero();
        n2_B(dim2) = -multiplier2;
        AddEdgeMiddleContactPoint(p_BQ, n1_B, n2_B);
      }
    }
  }
}

void VisualizeBlock(dev::RemoteTreeViewerWrapper* viewer,
                    const Eigen::Ref<const Eigen::Matrix3d>& R_WB,
                    const Eigen::Ref<const Eigen::Vector3d>& p_WB,
                    const Block& block) {
  Eigen::Affine3d tf;
  tf.linear() = R_WB;
  tf.translation() = p_WB;
  const Eigen::Vector4d color_blue(0.3, 0.3, 1, 0.9);
  viewer->PublishGeometry(
      DrakeShapes::Box({block.width(), block.width(), block.height()}), tf,
      color_blue, {"block"});
}

void VisualizeForce(dev::RemoteTreeViewerWrapper* viewer,
                    const Eigen::Ref<const Eigen::Vector3d>& p_WP,
                    const Eigen::Ref<const Eigen::Vector3d>& f_WP,
                    double normalizer, const std::string& path,
                    const Eigen::Ref<const Eigen::Vector4d>& color) {
  const Eigen::Vector3d f_WP_normalized = f_WP / normalizer;
  const std::vector<double> color_red = {1, 0.2, 0.2};
  if (f_WP_normalized.norm() > 1E-3) {
    viewer->PublishArrow(p_WP, p_WP + f_WP_normalized, {path}, 0.01, 0.02,
                         0.01, color);
  } else {
    viewer->PublishPointCloud(p_WP, {path}, {color_red});
  }
}

void VisualizeTable(dev::RemoteTreeViewerWrapper* viewer) {
  const Eigen::Vector4d color_gray(0.9, 0.9, 0.9, 0.3);
  Eigen::Affine3d tf_table;
  tf_table.setIdentity();
  tf_table.translation()(2) = -0.01;
  viewer->PublishGeometry(DrakeShapes::Box({1, 1, 0.02}), tf_table, color_gray,
                          {"table"});
}

void AllVerticesAboveTable(const Block& block, ObjectContactPlanning* problem) {
  // Constrain that all vertices of the block is above the table.
  for (int knot = 0; knot < problem->nT(); ++knot) {
    for (int i = 0; i < 8; ++i) {
      problem->get_mutable_prog()->AddLinearConstraint(
          (problem->p_WB()[knot] +
           problem->R_WB()[knot] * block.p_BV().col(i))(2) >= 0);
    }
  }
}

solvers::MatrixDecisionVariable<3, Eigen::Dynamic> SetTableContactVertices(
    const Block& block, const std::vector<int>& vertex_indices, double mu_table,
    int knot, double distance_big_M, ObjectContactPlanning* problem) {
  problem->SetContactVertexIndices(knot, vertex_indices,
                                   block.mass() * kGravity * 2);

  // Add the big-M constraint for the contact distance and the binary variable.
  // p_WV_z <= M * (1 - b) where b is the binary variable indicating contact.
  for (int i = 0; i < static_cast<int>(vertex_indices.size()); ++i) {
    const int vertex = vertex_indices[i];
    problem->get_mutable_prog()->AddLinearConstraint(
        problem->p_WV(knot, vertex)(2) <=
        distance_big_M * (1 - problem->vertex_contact_flag()[knot](i)));
  }

  // Compute the vertex contact force in the world frame f_WV.
  const int num_vertices = static_cast<int>(vertex_indices.size());
  const auto f_WV =
      problem->get_mutable_prog()->NewContinuousVariables<3, Eigen::Dynamic>(
          3, num_vertices, "f_WV");
  std::array<Eigen::VectorXd, 3> phi_f;
  phi_f[0] =
      Eigen::Matrix<double, 5, 1>::LinSpaced(-1, 1) * block.mass() * kGravity;
  phi_f[1] =
      Eigen::Matrix<double, 5, 1>::LinSpaced(-1, 1) * block.mass() * kGravity;
  phi_f[2] =
      Eigen::Matrix<double, 3, 1>::LinSpaced(0, 1) * block.mass() * kGravity;
  const auto table_friction_edges = GenerateLinearizedFrictionConeEdges<8>(
      Eigen::Vector3d::UnitZ(), mu_table);
  for (int i = 0; i < num_vertices; ++i) {
    problem->CalcContactForceInWorldFrame(problem->f_BV()[knot].col(i),
                                          f_WV.col(i), knot, false, phi_f);
    // Add friction cone constraint on f_WV
    // AddFrictionConeConstraint(mu_table, Vector3d::UnitZ(),
    //                          f_WV.col(i).cast<Expression>(),
    //                          problem.get_mutable_prog());
    AddLinearizedFrictionConeConstraint(table_friction_edges, f_WV.col(i),
                                        problem->get_mutable_prog());
  }

  return f_WV;
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
