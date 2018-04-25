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

  // Set body inertia.
  I_B_ = 1.0 / 12 * mass() *
         Eigen::Vector3d(width() * width() + height() * height(),
                         width() * width() + height() * height(),
                         2 * width() * width())
             .asDiagonal();

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
    viewer->PublishArrow(p_WP, p_WP + f_WP_normalized, {path}, 0.01, 0.02, 0.01,
                         color);
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

void SetUpBlockFlipTest(
    const Block& block, int num_pushers, int nT, ObjectContactPlanning* problem,
    std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>>* f_WV) {
  AllVerticesAboveTable(block, problem);
  f_WV->resize(nT);

  const double mu_table = 1;

  const Eigen::Vector3d p_WB0(0, 0, block.height() / 2);
  problem->get_mutable_prog()->AddBoundingBoxConstraint(p_WB0, p_WB0,
                                                        problem->p_WB()[0]);
  // Initially, the bottom vertices are on the table.
  (*f_WV)[0] = SetTableContactVertices(block, block.bottom_vertex_indices(),
                                       mu_table, 0, 0, problem);
  problem->get_mutable_prog()->AddBoundingBoxConstraint(
      1, 1, problem->vertex_contact_flag()[0]);

  // Finally, the positive x vertices are on the table.
  (*f_WV)[nT - 1] = SetTableContactVertices(
      block, block.positive_x_vertex_indices(), mu_table, nT - 1, 0, problem);
  problem->get_mutable_prog()->AddBoundingBoxConstraint(
      1, 1, problem->vertex_contact_flag()[nT - 1]);

  // For all the points in between the first and last knots, the candidate
  // table contact vertices are bottom and positive x vertices.
  for (int knot = 1; knot < nT - 1; ++knot) {
    (*f_WV)[knot] = SetTableContactVertices(
        block, block.bottom_and_positive_x_vertex_indices(), mu_table, knot,
        block.mass() * kGravity * 1.1, problem);

    // At least one vertex on the table, at most 4 vertices on the table
    problem->get_mutable_prog()->AddLinearConstraint(
        Eigen::RowVectorXd::Ones(
            block.bottom_and_positive_x_vertex_indices().size()),
        1, 4, problem->vertex_contact_flag()[knot]);
  }

  // Choose all body contact points except those on the bottom or the positive
  // x facet.
  std::vector<int> pusher_contact_point_indices;
  for (int i = 0; i < static_cast<int>(block.Q().size()); ++i) {
    if (block.Q()[i].p_BQ()(0) <= 1E-10 && block.Q()[i].p_BQ()(2) >= -1E-10) {
      pusher_contact_point_indices.push_back(i);
    }
  }
  for (int knot = 0; knot < nT; ++knot) {
    problem->SetPusherContactPointIndices(knot, pusher_contact_point_indices,
                                          block.mass() * kGravity);
  }
  // Pusher remain in static contact, no sliding is allowed.
  for (int interval = 0; interval < nT - 1; ++interval) {
    problem->AddPusherStaticContactConstraint(interval);
  }

  // Add non-sliding contact constraint on the vertex.
  std::vector<solvers::VectorXDecisionVariable> b_non_sliding(nT);
  auto AddVertexNonSlidingConstraintAtKnot = [&b_non_sliding](
      ObjectContactPlanning* problem, int knot,
      const std::vector<int>& common_vertex_indices, double distance_big_M) {
    const int num_vertex_knot = common_vertex_indices.size();
    b_non_sliding[knot].resize(num_vertex_knot);
    for (int i = 0; i < num_vertex_knot; ++i) {
      b_non_sliding[knot](i) =
          (problem->AddVertexNonSlidingConstraint(
               knot, common_vertex_indices[i], Eigen::Vector3d::UnitX(),
               Eigen::Vector3d::UnitY(), distance_big_M))
              .value();
    }
  };

  AddVertexNonSlidingConstraintAtKnot(problem, 0, block.bottom_vertex_indices(),
                                      0.1);
  AddVertexNonSlidingConstraintAtKnot(problem, nT - 2,
                                      block.positive_x_vertex_indices(), 0.1);
  for (int interval = 1; interval < nT - 2; ++interval) {
    AddVertexNonSlidingConstraintAtKnot(
        problem, interval, block.bottom_and_positive_x_vertex_indices(), 0.1);
  }

  // Bound the maximal angle difference in each interval.
  const double max_angle_difference = M_PI / 4;
  for (int interval = 0; interval < nT - 1; ++interval) {
    problem->AddOrientationDifferenceUpperBoundLinearApproximation(
        interval, max_angle_difference);
    problem->AddOrientationDifferenceUpperBoundBilinearApproximation(
        interval, max_angle_difference);
  }
  //// The block moves less than 10cms in each direction within an interval.
  // for (int interval = 0; interval < nT - 1; ++interval) {
  //  const Vector3<Expression> delta_p_WB =
  //      problem->p_WB()[interval + 1] - problem->p_WB()[interval];
  //  problem->get_mutable_prog()->AddLinearConstraint(delta_p_WB(0), -0.1,
  //  0.1);
  //  problem->get_mutable_prog()->AddLinearConstraint(delta_p_WB(1), -0.1,
  //  0.1);
  //  problem->get_mutable_prog()->AddLinearConstraint(delta_p_WB(2), -0.1,
  //  0.1);
  //}
}

void SetBlockFlipSolution(
    const ObjectContactPlanning& problem, const Block& block,
    const std::vector<solvers::MatrixDecisionVariable<3, Eigen::Dynamic>>& f_WV,
    bool print_flag, BlockFlipSolution* sol) {
  sol->R_WB_sol.reserve(problem.nT());
  sol->p_WB_sol.reserve(problem.nT());
  sol->p_WV_sol.reserve(problem.nT());
  sol->p_WQ_sol.reserve(problem.nT());
  sol->f_BV_sol.reserve(problem.nT());
  sol->f_BQ_sol.reserve(problem.nT());
  sol->f_WV_sol.reserve(problem.nT());
  sol->f_WQ_sol.reserve(problem.nT());
  sol->b_V_contact_sol.reserve(problem.nT());
  sol->b_Q_contact_sol.reserve(problem.nT());
  for (int knot = 0; knot < problem.nT(); ++knot) {
    sol->p_WB_sol.push_back(problem.prog().GetSolution(problem.p_WB()[knot]));
    sol->R_WB_sol.push_back(problem.prog().GetSolution(problem.R_WB()[knot]));
    sol->f_BV_sol.push_back(problem.prog().GetSolution(problem.f_BV()[knot]));
    sol->f_WV_sol.push_back(problem.prog().GetSolution(f_WV[knot]));
    const int num_vertices_knot = problem.contact_vertex_indices()[knot].size();
    sol->p_WV_sol.emplace_back(3, num_vertices_knot);
    for (int i = 0; i < num_vertices_knot; ++i) {
      sol->p_WV_sol[knot].col(i) =
          sol->p_WB_sol[knot] +
          sol->R_WB_sol[knot] *
              block.p_BV().col(problem.contact_vertex_indices()[knot][i]);
    }
    const int num_Q_points = problem.contact_Q_indices()[knot].size();
    sol->p_WQ_sol.emplace_back(3, num_Q_points);
    for (int i = 0; i < num_Q_points; ++i) {
      const int Q_index = problem.contact_Q_indices()[knot][i];
      sol->p_WQ_sol[knot].col(i) =
          sol->p_WB_sol[knot] + sol->R_WB_sol[knot] * block.Q()[Q_index].p_BQ();
    }
    sol->b_V_contact_sol.push_back(
        problem.prog().GetSolution(problem.vertex_contact_flag()[knot]));
    sol->f_BQ_sol.push_back(problem.prog().GetSolution(problem.f_BQ()[knot]));
    sol->f_WQ_sol.push_back(sol->R_WB_sol[knot] * sol->f_BQ_sol[knot]);
    sol->b_Q_contact_sol.push_back(
        problem.prog().GetSolution(problem.b_Q_contact()[knot]));
    if (print_flag) {
      std::cout << "knot " << knot << std::endl;
      std::cout << "p_WB[" << knot << "]\n " << sol->p_WB_sol[knot].transpose()
                << std::endl;
      std::cout << "R_WB[" << knot << "]\n " << sol->R_WB_sol[knot]
                << std::endl;
      std::cout << "b_V_contact[" << knot << "]\n "
                << sol->b_V_contact_sol[knot].transpose() << std::endl;
      std::cout << "f_BV_sol[" << knot << "]\n"
                << sol->f_BV_sol[knot] << std::endl;
      std::cout << "p_WV_sol[" << knot << "]\n"
                << sol->p_WV_sol[knot] << std::endl;

      std::cout << "b_Q_contact[" << knot << "]\n "
                << sol->b_Q_contact_sol[knot].transpose() << std::endl;
      std::cout << "f_BQ_sol[" << knot << "]\n"
                << sol->f_BQ_sol[knot] << std::endl;
    }
  }
}

void VisualizeResult(const ObjectContactPlanning& problem, const Block& block,
                     const BlockFlipSolution& sol) {
  // Now visualize the result.
  dev::RemoteTreeViewerWrapper viewer;

  const Eigen::Vector4d color_red(1, 0, 0, 0.9);
  const Eigen::Vector4d color_green(0, 1, 0, 0.9);

  // VisualizeTable(&viewer);

  const double viewer_force_normalizer = block.mass() * kGravity * 5;
  for (int knot = 0; knot < problem.nT(); ++knot) {
    VisualizeBlock(&viewer, sol.R_WB_sol[knot], sol.p_WB_sol[knot], block);
    // Visualize vertex contact force.
    for (int i = 0;
         i < static_cast<int>(problem.contact_vertex_indices()[knot].size());
         ++i) {
      VisualizeForce(&viewer, sol.p_WV_sol[knot].col(i),
                     sol.R_WB_sol[knot] * sol.f_BV_sol[knot].col(i),
                     viewer_force_normalizer, "f_WV" + std::to_string(i),
                     color_red);
    }
    // Visualize pusher contact force.
    for (int i = 0;
         i < static_cast<int>(problem.contact_Q_indices()[knot].size()); ++i) {
      VisualizeForce(&viewer, sol.p_WQ_sol[knot].col(i),
                     sol.R_WB_sol[knot] * sol.f_BQ_sol[knot].col(i),
                     viewer_force_normalizer, "f_WQ" + std::to_string(i),
                     color_green);
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }
}
}  // namespace planner
}  // namespace manipulation
}  // namespace drake
