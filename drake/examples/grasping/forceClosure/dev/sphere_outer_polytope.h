#pragma once

#include <vector>

#include <Eigen/Core>

#include "drake/common/monomial.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace examples {
namespace grasping {
/**
 * Find an outer polytopic approximation of the unit sphere in `kDim` dimensions.
 * The polytope has `kNumFacet` facets.
 * We want the vertices to be evenly distributed. To do so, we maximize the sum
 * of the squared pairwise distance between the facet normals.
 * @tparam kDim The dimension of the space.
 * @tparam kNumFacet Number of vertices.
 * @return The pair (A, b), such that the outer polytope is described as
 * {x | A * x <= b}
 */
template <size_t kDim, size_t kNumFacet>
std::pair<Eigen::Matrix<double, kNumFacet, kDim>, Eigen::Matrix<double, kNumFacet, 1>> SphereInnerPolytopeApproximation() {
  // Formulate the program
  // max Σᵢ,ⱼ(aᵢ - aᵢ)²
  // s.t |aᵢ| <= 1
  solvers::MathematicalProgram prog;
  auto A = prog.NewContinuousVariables<kNumFacet, kDim>("A");
  auto b = prog.NewContinuousVariables<kNumFacet>("b");
  // dist_sum = Σᵢ,ⱼ(aᵢ - aᵢ)²
  symbolic::Expression dist_sum = 0;
  for (int i = 0; i < kNumFacet; ++i) {
    for(int j = i + 1; j < kNumFacet; ++j) {
      dist_sum = (A.row(i) - A.row(j)).squaredNorm();
    }
  }
  // Σᵢ,ⱼ(aᵢ - aᵢ)² as vᵀ * Q * v where v is the column vector
  // as the concatenation of aᵢ
  const auto& map = symbolic::internal::DecomposePolynomialInternal(dist_sum);
  std::unordered_map<symbolic::Variable::Id, int> map_var_id_to_index;
  map_var_id_to_index.reserve(kNumFacet * kDim);
  for (int j = 0; j < kDim; ++j) {
    for (int i = 0; i < kNumFacet; ++i) {
      map_var_id_to_index.emplace(A(i, j).get_id(), j * kNumFacet + i);
    }
  }
  Eigen::Matrix<double, kDim * kNumFacet, kDim * kNumFacet> Q;
  Q.setZero();
  for (const auto& p : map) {
    const auto& p_powers = p.first.get_powers();
    if (p_powers.size() == 1) {
      if (p_powers.begin()->second == 2) {
        int idx = map_var_id_to_index[p_powers.begin()->first];
        Q(idx, idx) += p.second;
      } else {
        throw runtime_error("Should be a x^2 term.");
      }
    } else if (p_powers.size() == 2) {
      std::array<int, 2> idx;
      auto it = p_powers.begin();
      idx[0] = it->first;
      if (it->second != 1) {
        throw std::runtime_error("Should be a x*y term.");
      }
      ++it;
      idx[1] = it->first;
      if (it->second != 1) {
        throw std::runtime_error("Should be a x*y term.");
      }
      Q(idx[0], idx[1]) += p.second / 2;
      Q(idx[1], idx[0]) += p.second / 2;
    }
  }

};
}
}
}