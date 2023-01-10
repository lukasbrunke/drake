#include "drake/geometry/optimization/dev/cspace_free_polytope.h"

#include <chrono>
#include <future>
#include <limits>
#include <list>
#include <optional>
#include <set>
#include <string>
#include <thread>

#include "drake/common/symbolic/monomial_util.h"
#include "drake/common/symbolic/polynomial.h"
#include "drake/geometry/optimization/dev/collision_geometry.h"
#include "drake/geometry/optimization/dev/separating_plane.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperellipsoid.h"
#include "drake/multibody/rational/rational_forward_kinematics.h"
#include "drake/multibody/rational/rational_forward_kinematics_internal.h"
#include "drake/multibody/tree/multibody_tree_system.h"
#include "drake/multibody/tree/revolute_mobilizer.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace optimization {
const double kInf = std::numeric_limits<double>::infinity();

namespace {
// Returns true if all the bodies on the kinematics chain from `start` to `end`
// are welded together (namely all the mobilizer in between are welded).
[[nodiscard]] bool ChainIsWeld(const multibody::MultibodyPlant<double>& plant,
                               multibody::BodyIndex start,
                               multibody::BodyIndex end) {
  const std::vector<multibody::internal::MobilizerIndex> mobilizers =
      multibody::internal::FindMobilizersOnPath(plant, start, end);
  if (mobilizers.size() == 0) {
    return true;
  }
  const multibody::internal::MultibodyTree<double>& tree =
      multibody::internal::GetInternalTree(plant);
  for (const auto& mobilizer : mobilizers) {
    if (tree.get_mobilizer(mobilizer).num_positions() != 0) {
      return false;
    }
  }
  return true;
}

// For all the geometries `link1_geometries` on link1, and `link2_geometries` on
// link2, return them as pairs if they are not filtered.
[[nodiscard]] std::vector<
    std::pair<const CollisionGeometry*, const CollisionGeometry*>>
GetLinkCollisionPairs(
    const multibody::MultibodyPlant<double>& plant,
    const SceneGraph<double>& scene_graph, multibody::BodyIndex link1,
    multibody::BodyIndex link2,
    const std::vector<std::unique_ptr<CollisionGeometry>>& link1_geometries,
    const std::vector<std::unique_ptr<CollisionGeometry>>& link2_geometries) {
  std::vector<std::pair<const CollisionGeometry*, const CollisionGeometry*>>
      ret;
  if (ChainIsWeld(plant, link1, link2)) {
    // Two links cannot collide if there are only welded joints between them as
    // they cannot move relative to each other. Return an empty vector.
    return ret;
  }
  const auto& model_inspector = scene_graph.model_inspector();
  for (const auto& geometry1 : link1_geometries) {
    for (const auto& geometry2 : link2_geometries) {
      if (!model_inspector.CollisionFiltered(geometry1->id(),
                                             geometry2->id())) {
        ret.emplace_back(geometry1.get(), geometry2.get());
      }
    }
  }
  return ret;
}

struct BodyPair {
  BodyPair(multibody::BodyIndex m_body1, multibody::BodyIndex m_body2)
      : body1{m_body1}, body2{m_body2} {}
  bool operator==(const BodyPair& other) const {
    return body1 == other.body1 && body2 == other.body2;
  }

  multibody::BodyIndex body1;
  multibody::BodyIndex body2;
};

struct BodyPairHash {
  size_t operator()(const BodyPair& p) const { return p.body1 * 100 + p.body2; }
};

/**
 The monomials in the numerator of body point position has the form ∏ᵢ pow(tᵢ,
 dᵢ) * ∏ⱼ pow(xⱼ, cⱼ), where tᵢ is the configuration variable tᵢ = tan(Δθᵢ/2)
 for the revolute joint, dᵢ = 0, 1, 2, on the kinematic chain between the pair
 of bodies; xⱼ is the prismatic joint displacement on the same kinematic chain,
 cⱼ = 0 or 1. When we use SeparatingPlaneOrder = kAffine, the monomial basis
 would include all the monomials in the form ∏ᵢ pow(sᵢ, mᵢ) with mᵢ = 0 or 1.

 @param[in/out] map_body_to_monomial_basis stores all the monomials basis
 already computed.
 @param[out] monomial_basis Te monomial basis for this pair of body.
 */
void FindMonomialBasis(
    const multibody::RationalForwardKinematics& rational_forward_kin,
    const SortedPair<multibody::BodyIndex>& body_pair,
    std::unordered_map<SortedPair<multibody::BodyIndex>,
                       VectorX<symbolic::Monomial>>* map_body_to_monomial_basis,
    VectorX<symbolic::Monomial>* monomial_basis) {
  auto it = map_body_to_monomial_basis->find(body_pair);
  if (it == map_body_to_monomial_basis->end()) {
    const std::vector<multibody::internal::MobilizerIndex> mobilizer_indices =
        multibody::internal::FindMobilizersOnPath(rational_forward_kin.plant(),
                                                  body_pair.first(),
                                                  body_pair.second());
    const auto& tree =
        multibody::internal::GetInternalTree(rational_forward_kin.plant());
    symbolic::Variables s_set;
    for (const auto& mobilizer_index : mobilizer_indices) {
      const auto& mobilizer = tree.get_mobilizer(mobilizer_index);
      if ((mobilizer.can_rotate() && !mobilizer.can_translate()) ||
          (mobilizer.can_translate() && !mobilizer.can_rotate())) {
        // This is a revolute or prismatic joint.
        s_set.insert(
            rational_forward_kin
                .s()[rational_forward_kin
                         .map_mobilizer_to_s_index()[mobilizer_index]]);
      } else if (mobilizer.num_velocities() > 0) {
        throw std::runtime_error(
            "FindMonomialBasis: we only support revolute, prismatic or weld "
            "mobilizers.");
      }
    }
    if (s_set.empty()) {
      // No s variable. The monomial basis is just [1].
      monomial_basis->resize(1);
      (*monomial_basis)(0) = symbolic::Monomial();
    } else {
      *monomial_basis = symbolic::CalcMonomialBasisOrderUpToOne(s_set);
    }
    it = map_body_to_monomial_basis->emplace_hint(it, body_pair,
                                                  *monomial_basis);
  } else {
    *monomial_basis = it->second;
  }
}

// TODO(hongkai.dai): move this function to a header file for general usage.
template <typename T>
void SymmetricMatrixFromLowerTriangularPart(
    int rows, const Eigen::Ref<const VectorX<T>>& lower_triangle,
    MatrixX<T>* mat) {
  mat->resize(rows, rows);
  DRAKE_DEMAND(lower_triangle.rows() == rows * (rows + 1) / 2);
  int count = 0;
  for (int j = 0; j < rows; ++j) {
    (*mat)(j, j) = lower_triangle(count++);
    for (int i = j + 1; i < rows; ++i) {
      (*mat)(i, j) = lower_triangle(count);
      (*mat)(j, i) = lower_triangle(count);
      count++;
    }
  }
}

void AddPsdConstraint(solvers::MathematicalProgram* prog,
                      const MatrixX<symbolic::Variable>& X) {
  DRAKE_DEMAND(X.rows() == X.cols());
  if (X.rows() == 1) {
    prog->AddBoundingBoxConstraint(0, kInf, X(0, 0));
  } else if (X.rows() == 2) {
    prog->AddRotatedLorentzConeConstraint(
        Vector3<symbolic::Variable>(X(0, 0), X(1, 1), X(0, 1)));
  } else {
    prog->AddPositiveSemidefiniteConstraint(X);
  }
}

void AddSosPolynomial(
    solvers::MathematicalProgram* prog,
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& gram_lower,
    symbolic::Polynomial* poly, MatrixX<symbolic::Variable>* gram) {
  SymmetricMatrixFromLowerTriangularPart<symbolic::Variable>(
      monomial_basis.rows(), gram_lower, gram);
  AddPsdConstraint(prog, *gram);
  *poly =
      symbolic::CalcPolynomialWLowerTriangularPart(monomial_basis, gram_lower);
}

// Add the constraint that poly = p0(monomial_basis_array[0], gram[0]) +
// p1(monomial_basis_array[1], gram[1]) + p2(monomial_basis_array[2], gram[2]),
// where p0, p1, p2 are all sos polynomials.
void AddSosPolynomial(
    solvers::MathematicalProgram* prog,
    const std::array<VectorX<symbolic::Monomial>, 3>& monomial_basis_array,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& gram_lower,
    symbolic::Polynomial* poly,
    std::array<MatrixX<symbolic::Variable>, 3>* gram) {
  int gram_total_size = 0;
  for (int i = 0; i < 3; ++i) {
    gram_total_size += monomial_basis_array[i].rows() *
                       (monomial_basis_array[i].rows() + 1) / 2;
  }
  DRAKE_DEMAND(gram_lower.rows() == gram_total_size);

  int gram_lower_count = 0;
  *poly = symbolic::Polynomial();
  for (int i = 0; i < 3; ++i) {
    const int gram_lower_size = monomial_basis_array[i].rows() *
                                (monomial_basis_array[i].rows() + 1) / 2;
    SymmetricMatrixFromLowerTriangularPart<symbolic::Variable>(
        monomial_basis_array[i].rows(),
        gram_lower.segment(gram_lower_count, gram_lower_size), &((*gram)[i]));
    AddPsdConstraint(prog, (*gram)[i]);
    *poly += symbolic::CalcPolynomialWLowerTriangularPart(
        monomial_basis_array[i],
        gram_lower.segment(gram_lower_count, gram_lower_size));
    gram_lower_count += gram_lower_size;
  }
}

// Computes the Lagrangian, and add the constraints on the Lagrangian Gram
// matrix to the program. If it is redundant, then set the Lagrangian to zero
// and constrain the Gram matrix to zero; otherwise constrain the Gram matrix to
// be PSD.
void AddLagrangian(
    solvers::MathematicalProgram* prog,
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& gram_lower,
    bool redundant, symbolic::Polynomial* lagrangian,
    MatrixX<symbolic::Variable>* lagrangian_gram) {
  SymmetricMatrixFromLowerTriangularPart<symbolic::Variable>(
      monomial_basis.rows(), gram_lower, lagrangian_gram);
  DRAKE_DEMAND(gram_lower.rows() ==
               monomial_basis.rows() * (monomial_basis.rows() + 1) / 2);
  if (redundant) {
    // Lagrangian is 0.
    *lagrangian = symbolic::Polynomial();
    prog->AddBoundingBoxConstraint(0, 0, gram_lower);
  } else {
    *lagrangian = symbolic::CalcPolynomialWLowerTriangularPart(monomial_basis,
                                                               gram_lower);
    AddPsdConstraint(prog, *lagrangian_gram);
  }
}

void AddLagrangian(
    solvers::MathematicalProgram* prog,
    const std::array<VectorX<symbolic::Monomial>, 3>& monomial_basis,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& gram_lower,
    bool redundant, symbolic::Polynomial* lagrangian,
    std::array<MatrixX<symbolic::Variable>, 3>* lagrangian_gram) {
  int gram_var_count = 0;
  for (int i = 0; i < 3; ++i) {
    const int gram_lower_size =
        monomial_basis[i].rows() * (monomial_basis[i].rows() + 1) / 2;
    SymmetricMatrixFromLowerTriangularPart<symbolic::Variable>(
        monomial_basis[i].rows(),
        gram_lower.segment(gram_var_count, gram_lower_size),
        &((*lagrangian_gram)[i]));
    gram_var_count += gram_lower_size;
  }
  DRAKE_DEMAND(gram_lower.rows() == gram_var_count);

  if (redundant) {
    // Lagrangian is 0.
    *lagrangian = symbolic::Polynomial();
    prog->AddBoundingBoxConstraint(0, 0, gram_lower);
  } else {
    *lagrangian = symbolic::Polynomial();
    gram_var_count = 0;
    for (int i = 0; i < 3; ++i) {
      const int gram_lower_size =
          monomial_basis[i].rows() * (monomial_basis[i].rows() + 1) / 2;
      *lagrangian += symbolic::CalcPolynomialWLowerTriangularPart(
          monomial_basis[i],
          gram_lower.segment(gram_var_count, gram_lower_size));
      AddPsdConstraint(prog, (*lagrangian_gram)[i]);
      gram_var_count += gram_lower_size;
    }
  }
}

// Checks if a future has completed execution.
// This function is taken from monte_carlo.cc. It will be used in the "thread
// pool" implementation (which doesn't use the openMP).
template <typename T>
bool IsFutureReady(const std::future<T>& future) {
  // future.wait_for() is the only method to check the status of a future
  // without waiting for it to complete.
  const std::future_status status =
      future.wait_for(std::chrono::milliseconds(1));
  return (status == std::future_status::ready);
}

solvers::MathematicalProgramResult SolveWithBackoff(
    solvers::MathematicalProgram* prog, std::optional<double> backoff_scale,
    const std::optional<solvers::SolverOptions>& solver_options,
    const solvers::SolverId& solver_id) {
  DRAKE_DEMAND(prog->quadratic_costs().size() == 0);
  auto solver = solvers::MakeSolver(solver_id);
  solvers::MathematicalProgramResult result;
  solver->Solve(*prog, std::nullopt, solver_options, &result);
  if (!result.is_success()) {
    drake::log()->warn("Failed before backoff.");
  }
  if (backoff_scale.has_value() && !(prog->linear_costs().empty())) {
    DRAKE_DEMAND(prog->linear_costs().size() == 1);
    const double cost_val = result.get_optimal_cost();
    const double cost_upper_bound =
        cost_val > 0 ? (1 + backoff_scale.value()) * cost_val
                     : (1 - backoff_scale.value()) * cost_val;
    prog->AddLinearConstraint(
        prog->linear_costs()[0].evaluator()->a(), -kInf,
        cost_upper_bound - prog->linear_costs()[0].evaluator()->b(),
        prog->linear_costs()[0].variables());
    prog->RemoveCost(prog->linear_costs()[0]);
    solver->Solve(*prog, std::nullopt, solver_options, &result);
    if (!result.is_success()) {
      drake::log()->info("Failed in backoff.");
    }
  }
  return result;
}

void AddMonomialBasisWithY(
    const SortedPair<multibody::BodyIndex>& body_pair,
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Vector3<symbolic::Variable>& y_slack,
    CspaceFreePolytope::Formulation formulation,
    std::unordered_map<SortedPair<multibody::BodyIndex>,
                       VectorX<symbolic::Monomial>>*
        map_body_to_monomial_basis_w_y,
    std::unordered_map<SortedPair<multibody::BodyIndex>,
                       std::array<VectorX<symbolic::Monomial>, 3>>*
        map_body_to_monomial_basis_w_y_array) {
  switch (formulation) {
    case CspaceFreePolytope::Formulation::kFormulation1: {
      auto it = map_body_to_monomial_basis_w_y_array->find(body_pair);
      if (it == map_body_to_monomial_basis_w_y_array->end()) {
        // monomial_basis_w_y_array[i] =
        // [       monomial_basis]
        // [y(i) * monomial_basis]
        std::array<VectorX<symbolic::Monomial>, 3> monomial_basis_w_y_array;
        for (int i = 0; i < 3; ++i) {
          monomial_basis_w_y_array[i].resize(2 * monomial_basis.rows());
          monomial_basis_w_y_array[i].head(monomial_basis.rows()) =
              monomial_basis;
          for (int j = 0; j < monomial_basis.rows(); ++j) {
            monomial_basis_w_y_array[i](monomial_basis.rows() + j) =
                monomial_basis(j) * symbolic::Monomial(y_slack(i));
          }
        }
        map_body_to_monomial_basis_w_y_array->emplace_hint(
            it, body_pair, monomial_basis_w_y_array);
      }
      break;
    }
    case CspaceFreePolytope::Formulation::kFormulation2: {
      auto it = map_body_to_monomial_basis_w_y->find(body_pair);
      if (it == map_body_to_monomial_basis_w_y->end()) {
        // monomial_basis_w_y =
        // [       monomial_basis]
        // [y(0) * monomial_basis]
        // [y(1) * monomial_basis]
        // [y(2) * monomial_basis]
        VectorX<symbolic::Monomial> monomial_basis_w_y(4 *
                                                       monomial_basis.rows());
        monomial_basis_w_y.head(monomial_basis.rows()) = monomial_basis;
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < monomial_basis.rows(); ++j) {
            monomial_basis_w_y(monomial_basis.rows() * (i + 1) + j) =
                monomial_basis(j) * symbolic::Monomial(y_slack(i));
          }
        }
        map_body_to_monomial_basis_w_y->emplace_hint(it, body_pair,
                                                     monomial_basis_w_y);
      }
      break;
    }
  }
}

template <typename MonomialBasisType>
struct LagrangianGram;

template <>
struct LagrangianGram<VectorX<symbolic::Monomial>> {
  using type = MatrixX<symbolic::Variable>;
};

template <>
struct LagrangianGram<std::array<VectorX<symbolic::Monomial>, 3>> {
  using type = std::array<MatrixX<symbolic::Variable>, 3>;
};

template <typename MonomialBasisType>
typename LagrangianGram<MonomialBasisType>::type InitializeGram(
    const MonomialBasisType& monomial_basis) {
  typename LagrangianGram<MonomialBasisType>::type gram;
  if constexpr (std::is_same_v<MonomialBasisType,
                               VectorX<symbolic::Monomial>>) {
    gram.resize(monomial_basis.rows(), monomial_basis.rows());
  } else if constexpr (std::is_same_v<
                           MonomialBasisType,
                           std::array<VectorX<symbolic::Monomial>, 3>>) {
    for (int i = 0; i < 3; ++i) {
      gram[i].resize(monomial_basis[i].rows(), monomial_basis[i].rows());
    }
  }
  return gram;
}

template <typename MonomialBasisType>
int GetGramVarSize(const MonomialBasisType& monomial_basis) {
  int num_gram_vars = 0;
  if constexpr (std::is_same_v<MonomialBasisType,
                               VectorX<symbolic::Monomial>>) {
    num_gram_vars = monomial_basis.rows() * (monomial_basis.rows() + 1) / 2;
  } else if constexpr (std::is_same_v<
                           MonomialBasisType,
                           std::array<VectorX<symbolic::Monomial>, 3>>) {
    for (int i = 0; i < 3; ++i) {
      num_gram_vars +=
          monomial_basis[i].rows() * (monomial_basis[i].rows() + 1) / 2;
    }
  }
  return num_gram_vars;
}

template <typename MonomialBasisType>
void AddRationalsNonnegative(
    solvers::MathematicalProgram* prog,
    const SeparatingPlane<symbolic::Variable>& plane,
    const VectorX<symbolic::Polynomial>& d_minus_Cs,
    const VectorX<symbolic::Polynomial>& s_minus_s_lower,
    const VectorX<symbolic::Polynomial>& s_upper_minus_s,
    const std::unordered_set<int>& C_redundant_indices,
    const std::unordered_set<int>& s_lower_redundant_indices,
    const std::unordered_set<int>& s_upper_redundant_indices,
    const std::vector<symbolic::RationalFunction>& rationals,
    multibody::BodyIndex geometry_body,
    const std::unordered_map<SortedPair<multibody::BodyIndex>,
                             MonomialBasisType>& monomial_basis_map,
    std::vector<CspaceFreePolytope::SeparatingPlaneLagrangians>*
        lagrangians_vec) {
  const SortedPair<multibody::BodyIndex> body_pair(plane.expressed_body,
                                                   geometry_body);
  if (!rationals.empty()) {
    const MonomialBasisType& monomial_basis = monomial_basis_map.at(body_pair);
    const int s_size = s_minus_s_lower.rows();
    const int num_lagrangians =
        rationals.size() * (1 + d_minus_Cs.rows() + 2 * s_size);
    const int num_gram_vars_per_lagrangian = GetGramVarSize(monomial_basis);
    const int num_gram_vars = num_lagrangians * num_gram_vars_per_lagrangian;
    auto gram_vars = prog->NewContinuousVariables(num_gram_vars, "Gram");

    int gram_var_count = 0;
    lagrangians_vec->reserve(rationals.size());

    typename LagrangianGram<MonomialBasisType>::type gram =
        InitializeGram(monomial_basis);
    for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
      lagrangians_vec->emplace_back(d_minus_Cs.rows(), s_size);
      auto& lagrangians = lagrangians_vec->back();
      for (int j = 0; j < d_minus_Cs.rows(); ++j) {
        AddLagrangian(
            prog, monomial_basis,
            gram_vars.segment(gram_var_count, num_gram_vars_per_lagrangian),
            C_redundant_indices.count(j) > 0, &(lagrangians.polytope(j)),
            &gram);
        gram_var_count += num_gram_vars_per_lagrangian;
      }
      for (int j = 0; j < s_size; ++j) {
        AddLagrangian(
            prog, monomial_basis,
            gram_vars.segment(gram_var_count, num_gram_vars_per_lagrangian),
            s_lower_redundant_indices.count(j), &(lagrangians.s_lower(j)),
            &gram);
        gram_var_count += num_gram_vars_per_lagrangian;
        AddLagrangian(
            prog, monomial_basis,
            gram_vars.segment(gram_var_count, num_gram_vars_per_lagrangian),
            s_upper_redundant_indices.count(j) > 0, &(lagrangians.s_upper(j)),
            &gram);
        gram_var_count += num_gram_vars_per_lagrangian;
      }
      const symbolic::Polynomial poly =
          rationals[i].numerator() - lagrangians.polytope.dot(d_minus_Cs) -
          lagrangians.s_lower.dot(s_minus_s_lower) -
          lagrangians.s_upper.dot(s_upper_minus_s);
      symbolic::Polynomial poly_sos;
      AddSosPolynomial(
          prog, monomial_basis,
          gram_vars.segment(gram_var_count, num_gram_vars_per_lagrangian),
          &poly_sos, &gram);
      gram_var_count += num_gram_vars_per_lagrangian;
      prog->AddEqualityConstraintBetweenPolynomials(poly, poly_sos);
    }
    DRAKE_DEMAND(gram_var_count == num_gram_vars);
  }
}

// Each matrix sos will add Lagrangian multipliers for s-s_lower and
// s_upper-s (if search_s_bounds_lagrangian=true), together with one
// sos that rational.numerator() - λ(s)ᵀ * (d - C*s) - λ_lower(s)ᵀ *
// (s - s_lower) -λ_upper(s)ᵀ * (s_upper - s) is sos
int GetGramVarSizeGivenLagrangian(const int num_rationals, int s_size,
                                  bool search_s_bounds_lagrangians,
                                  int num_gram_vars_per_lagrangian) {
  const int num_sos =
      num_rationals * (1 + (search_s_bounds_lagrangians ? 2 * s_size : 0));
  return num_gram_vars_per_lagrangian * num_sos;
}

// @param gram_vars The gram_vars declared for the entire program.
// @param[in/out] gram_var_count The number of gram variables used at the
// beginning/end of calling this function.
// TODO(hongkai.di): move this function to InitializePolytopeSearchProgram as a
// templated lambda.
template <typename MonomialBasisType>
void AddRationalsNonnegativeGivenLagrangian(
    solvers::MathematicalProgram* prog,
    const VectorX<symbolic::Polynomial>& d_minus_Cs,
    const VectorX<symbolic::Polynomial>& s_minus_s_lower,
    const VectorX<symbolic::Polynomial>& s_upper_minus_s,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& gram_vars, int s_size,
    bool search_s_bounds_lagrangians,
    const std::vector<symbolic::RationalFunction>& rationals,
    const std::unordered_map<SortedPair<multibody::BodyIndex>,
                             MonomialBasisType>& monomial_basis_map,
    multibody::BodyIndex expressed_body, multibody::BodyIndex geometry_body,
    const std::vector<CspaceFreePolytope::SeparatingPlaneLagrangians>&
        lagrangians_vec,
    int* gram_var_count,
    std::vector<CspaceFreePolytope::SeparatingPlaneLagrangians>*
        new_lagrangians_vec) {
  DRAKE_DEMAND(lagrangians_vec.size() == rationals.size());
  if (!rationals.empty()) {
    const MonomialBasisType& monomial_basis = monomial_basis_map.at(
        SortedPair<multibody::BodyIndex>(expressed_body, geometry_body));
    const int num_gram_vars_per_lagrangian = GetGramVarSize(monomial_basis);
    typename LagrangianGram<MonomialBasisType>::type gram =
        InitializeGram(monomial_basis);
    VectorX<symbolic::Polynomial> s_lower_lagrangians(s_size);
    VectorX<symbolic::Polynomial> s_upper_lagrangians(s_size);
    // Add Lagrangian multipliers for joint limits.
    for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
      if (search_s_bounds_lagrangians) {
        for (int j = 0; j < s_size; ++j) {
          AddSosPolynomial(
              prog, monomial_basis,
              gram_vars.segment(*gram_var_count, num_gram_vars_per_lagrangian),
              &(s_lower_lagrangians(j)), &gram);
          *gram_var_count += num_gram_vars_per_lagrangian;
          AddSosPolynomial(
              prog, monomial_basis,
              gram_vars.segment(*gram_var_count, num_gram_vars_per_lagrangian),
              &(s_upper_lagrangians(j)), &gram);
          *gram_var_count += num_gram_vars_per_lagrangian;
        }
      } else {
        s_lower_lagrangians = lagrangians_vec[i].s_lower;
        s_upper_lagrangians = lagrangians_vec[i].s_upper;
      }

      if (new_lagrangians_vec != nullptr) {
        new_lagrangians_vec->emplace_back(d_minus_Cs.rows(), s_size);
        (*new_lagrangians_vec)[i].polytope = lagrangians_vec[i].polytope;
        (*new_lagrangians_vec)[i].s_lower = s_lower_lagrangians;
        (*new_lagrangians_vec)[i].s_upper = s_upper_lagrangians;
      }
      const symbolic::Polynomial poly =
          rationals[i].numerator() -
          lagrangians_vec[i].polytope.dot(d_minus_Cs) -
          s_lower_lagrangians.dot(s_minus_s_lower) -
          s_upper_lagrangians.dot(s_upper_minus_s);
      symbolic::Polynomial poly_sos;
      AddSosPolynomial(
          prog, monomial_basis,
          gram_vars.segment(*gram_var_count, num_gram_vars_per_lagrangian),
          &poly_sos, &gram);
      *gram_var_count += num_gram_vars_per_lagrangian;
      prog->AddEqualityConstraintBetweenPolynomials(poly, poly_sos);
    }
  }
}
}  // namespace

CspaceFreePolytope::CspaceFreePolytope(
    const multibody::MultibodyPlant<double>* plant,
    const geometry::SceneGraph<double>* scene_graph,
    SeparatingPlaneOrder plane_order,
    const Eigen::Ref<const Eigen::VectorXd>& q_star, const Options& options)
    : rational_forward_kin_(plant),
      scene_graph_{*scene_graph},
      link_geometries_{GetCollisionGeometries(*plant, *scene_graph)},
      plane_order_{plane_order},
      s_set_{rational_forward_kin_.s()},
      q_star_{q_star},
      formulation_{options.formulation} {
  // Create separating planes.
  // collision_pairs maps each pair of body to the pair of collision geometries
  // on that pair of body.
  std::map<SortedPair<multibody::BodyIndex>,
           std::vector<
               std::pair<const CollisionGeometry*, const CollisionGeometry*>>>
      collision_pairs;
  int num_collision_pairs = 0;
  for (const auto& [link1, geometries1] : link_geometries_) {
    for (const auto& [link2, geometries2] : link_geometries_) {
      if (link1 < link2) {
        auto it = collision_pairs.emplace_hint(
            collision_pairs.end(),
            SortedPair<multibody::BodyIndex>(link1, link2),
            GetLinkCollisionPairs(rational_forward_kin_.plant(), scene_graph_,
                                  link1, link2, geometries1, geometries2));
        num_collision_pairs += static_cast<int>(it->second.size());
      }
    }
  }
  separating_planes_.reserve(num_collision_pairs);
  const symbolic::Monomial monomial_one{};
  for (const auto& [link_pair, geometry_pairs] : collision_pairs) {
    for (const auto& geometry_pair : geometry_pairs) {
      Vector3<symbolic::Polynomial> a;
      symbolic::Polynomial b;
      VectorX<symbolic::Variable> plane_decision_vars;
      switch (plane_order) {
        case SeparatingPlaneOrder::kAffine: {
          const VectorX<symbolic::Variable> s_for_plane =
              rational_forward_kin_.s();
          plane_decision_vars.resize(4 * s_for_plane.rows() + 4);
          for (int i = 0; i < plane_decision_vars.rows(); ++i) {
            plane_decision_vars(i) =
                symbolic::Variable(fmt::format("plane_var{}", i));
          }
          CalcPlane<symbolic::Variable, symbolic::Variable,
                    symbolic::Polynomial>(plane_decision_vars, s_for_plane,
                                          plane_order_, &a, &b);
        }
      }
      const multibody::BodyIndex expressed_body =
          multibody::internal::FindBodyInTheMiddleOfChain(
              rational_forward_kin_.plant(), link_pair.first(),
              link_pair.second());
      separating_planes_.emplace_back(a, b, geometry_pair.first,
                                      geometry_pair.second, expressed_body,
                                      plane_order_, plane_decision_vars);

      map_geometries_to_separating_planes_.emplace(
          SortedPair<geometry::GeometryId>(geometry_pair.first->id(),
                                           geometry_pair.second->id()),
          static_cast<int>(separating_planes_.size()) - 1);
    }
  }

  y_slack_.resize(3);
  for (int i = 0; i < 3; ++i) {
    y_slack_(i) = symbolic::Variable("y" + std::to_string(i));
  }

  s_lower_ = rational_forward_kin_.ComputeSValue(
      rational_forward_kin_.plant().GetPositionLowerLimits(), q_star_);
  s_upper_ = rational_forward_kin_.ComputeSValue(
      rational_forward_kin_.plant().GetPositionUpperLimits(), q_star_);
  CalcSBoundsPolynomial();

  this->GenerateRationals();

  this->CalcMonomialBasis();
}

void CspaceFreePolytope::GenerateRationals() {
  // There can be multiple geometries on the same pair, hence the body pose will
  // be reused. We use this map to store the body pose to avoid redundant
  // computation.
  std::unordered_map<
      BodyPair,
      multibody::RationalForwardKinematics::Pose<symbolic::Polynomial>,
      BodyPairHash>
      body_pair_to_X_AB_multilinear;
  for (int plane_index = 0;
       plane_index < static_cast<int>(separating_planes_.size());
       ++plane_index) {
    const auto& separating_plane = separating_planes_[plane_index];
    // Compute X_AB for both side of the geometries.
    std::vector<symbolic::RationalFunction> positive_side_rationals;
    std::vector<symbolic::RationalFunction> negative_side_rationals;
    std::vector<symbolic::RationalFunction> positive_side_psd_mat;
    std::vector<symbolic::RationalFunction> negative_side_psd_mat;
    for (const PlaneSide plane_side :
         {PlaneSide::kPositive, PlaneSide::kNegative}) {
      const CollisionGeometry* link_geometry =
          separating_plane.geometry(plane_side);

      const BodyPair expressed_to_link(separating_plane.expressed_body,
                                       link_geometry->body_index());
      auto it = body_pair_to_X_AB_multilinear.find(expressed_to_link);
      if (it == body_pair_to_X_AB_multilinear.end()) {
        it = body_pair_to_X_AB_multilinear.emplace_hint(
            it, expressed_to_link,
            rational_forward_kin_.CalcBodyPoseAsMultilinearPolynomial(
                q_star_, link_geometry->body_index(),
                separating_plane.expressed_body));
      }
      const multibody::RationalForwardKinematics::Pose<symbolic::Polynomial>&
          X_AB_multilinear = it->second;
      auto& rationals = plane_side == PlaneSide::kPositive
                            ? positive_side_rationals
                            : negative_side_rationals;
      auto& psd_mat = plane_side == PlaneSide::kPositive
                          ? positive_side_psd_mat
                          : negative_side_psd_mat;
      link_geometry->OnPlaneSide(separating_plane.a, separating_plane.b,
                                 X_AB_multilinear, rational_forward_kin_,
                                 plane_side, y_slack_, &rationals, &psd_mat);
    }
    plane_geometries_.emplace_back(
        positive_side_rationals, negative_side_rationals, positive_side_psd_mat,
        negative_side_psd_mat, plane_index);
    DRAKE_DEMAND(plane_geometries_[plane_index].plane_index == plane_index);
  }
}

// Find the redundant inequalities in C * s <= d, s_lower <= s <= s_upper.
void CspaceFreePolytope::FindRedundantInequalities(
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
    const Eigen::VectorXd& s_lower, const Eigen::VectorXd& s_upper,
    double tighten, std::unordered_set<int>* C_redundant_indices,
    std::unordered_set<int>* s_lower_redundant_indices,
    std::unordered_set<int>* s_upper_redundant_indices) const {
  C_redundant_indices->clear();
  s_lower_redundant_indices->clear();
  s_upper_redundant_indices->clear();
  // We aggregate the constraint {C*s<=d, s_lower <= s <= s_upper} as C̅s ≤
  // d̅
  const int ns = s_lower.rows();
  Eigen::MatrixXd C_bar(C.rows() + 2 * ns, ns);
  Eigen::VectorXd d_bar(d.rows() + 2 * ns);
  C_bar << C, Eigen::MatrixXd::Identity(ns, ns),
      -Eigen::MatrixXd::Identity(ns, ns);
  d_bar << d, s_upper, -s_lower;
  const HPolyhedron hpolyhedron(C_bar, d_bar);
  const std::set<int> redundant_indices = hpolyhedron.FindRedundant(-tighten);
  C_redundant_indices->reserve(redundant_indices.size());
  s_lower_redundant_indices->reserve(redundant_indices.size());
  s_upper_redundant_indices->reserve(redundant_indices.size());
  for (const int index : redundant_indices) {
    if (index < C.rows()) {
      C_redundant_indices->emplace_hint(C_redundant_indices->end(), index);
    } else if (index < C.rows() + ns) {
      s_upper_redundant_indices->emplace_hint(s_upper_redundant_indices->end(),
                                              index - C.rows());
    } else {
      s_lower_redundant_indices->emplace_hint(s_lower_redundant_indices->end(),
                                              index - C.rows() - ns);
    }
  }
}

void CspaceFreePolytope::CalcSBoundsPolynomial() {
  const int s_size = rational_forward_kin_.s().rows();
  s_minus_s_lower_.resize(s_size);
  s_upper_minus_s_.resize(s_size);
  const symbolic::Monomial monomial_one{};
  const auto& s = rational_forward_kin_.s();
  for (int i = 0; i < s_size; ++i) {
    s_minus_s_lower_(i) = symbolic::Polynomial(symbolic::Polynomial::MapType(
        {{symbolic::Monomial(s(i)), 1}, {monomial_one, -s_lower_(i)}}));
    s_upper_minus_s_(i) = symbolic::Polynomial(symbolic::Polynomial::MapType(
        {{monomial_one, s_upper_(i)}, {symbolic::Monomial(s(i)), -1}}));
  }
}

template <typename T>
VectorX<symbolic::Polynomial> CspaceFreePolytope::CalcDminusCs(
    const Eigen::Ref<const MatrixX<T>>& C,
    const Eigen::Ref<const VectorX<T>>& d) const {
  // Now build the polynomials d(i) - C.row(i) * s
  const auto& s = rational_forward_kin_.s();
  DRAKE_DEMAND(C.rows() == d.rows() && C.cols() == static_cast<int>(s.size()));
  const symbolic::Monomial monomial_one{};
  symbolic::Polynomial::MapType d_minus_Cs_poly_map;
  VectorX<symbolic::Monomial> s_monomials(s.rows());
  for (int i = 0; i < s.rows(); ++i) {
    s_monomials(i) = symbolic::Monomial(s(i));
  }
  VectorX<symbolic::Polynomial> d_minus_Cs(d.rows());
  for (int i = 0; i < C.rows(); ++i) {
    for (int j = 0; j < static_cast<int>(s.rows()); ++j) {
      auto it = d_minus_Cs_poly_map.find(s_monomials(j));
      if (it == d_minus_Cs_poly_map.end()) {
        d_minus_Cs_poly_map.emplace_hint(it, s_monomials(j), -C(i, j));
      } else {
        it->second = -C(i, j);
      }
    }
    auto it = d_minus_Cs_poly_map.find(monomial_one);
    if (it == d_minus_Cs_poly_map.end()) {
      d_minus_Cs_poly_map.emplace_hint(it, monomial_one, d(i));
    } else {
      it->second = d(i);
    }
    d_minus_Cs(i) = symbolic::Polynomial(d_minus_Cs_poly_map);
  }
  return d_minus_Cs;
}

CspaceFreePolytope::SeparatingPlaneLagrangians
CspaceFreePolytope::SeparatingPlaneLagrangians::GetSolution(
    const solvers::MathematicalProgramResult& result) const {
  CspaceFreePolytope::SeparatingPlaneLagrangians ret(this->polytope.rows(),
                                                     this->s_lower.rows());
  for (int i = 0; i < this->polytope.rows(); ++i) {
    ret.polytope(i) = result.GetSolution(this->polytope(i));
  }
  for (int i = 0; i < this->s_lower.rows(); ++i) {
    ret.s_lower(i) = result.GetSolution(this->s_lower(i));
    ret.s_upper(i) = result.GetSolution(this->s_upper(i));
  }
  return ret;
}

void CspaceFreePolytope::CalcMonomialBasis() {
  for (int plane_index = 0;
       plane_index < static_cast<int>(separating_planes_.size());
       ++plane_index) {
    const auto& plane = separating_planes_[plane_index];
    for (const auto collision_geometry :
         {plane.positive_side_geometry, plane.negative_side_geometry}) {
      const SortedPair<multibody::BodyIndex> body_pair(
          plane.expressed_body, collision_geometry->body_index());
      VectorX<symbolic::Monomial> monomial_basis;
      FindMonomialBasis(rational_forward_kin_, body_pair,
                        &map_body_to_monomial_basis_, &monomial_basis);

      // Now consider the monomial basis with y_slack variable, used for the
      // matrix-sos condition for non-polytopic collision geometries.
      switch (collision_geometry->type()) {
        case GeometryType::kPolytope:
          break;
        case GeometryType::kSphere:
        case GeometryType::kCapsule: {
          AddMonomialBasisWithY(body_pair, monomial_basis, y_slack_,
                                formulation_, &map_body_to_monomial_basis_w_y_,
                                &map_body_to_monomial_basis_w_y_array_);
          break;
        }
        case GeometryType::kCylinder: {
          throw std::runtime_error(
              "CalcMonomialBasis: do not support cylindrical geometry yet.");
        }
      }
    }
  }
}

CspaceFreePolytope::SeparationCertificateProgram
CspaceFreePolytope::ConstructPlaneSearchProgram(
    const PlaneSeparatesGeometries& plane_geometries,
    const VectorX<symbolic::Polynomial>& d_minus_Cs,
    const std::unordered_set<int>& C_redundant_indices,
    const std::unordered_set<int>& s_lower_redundant_indices,
    const std::unordered_set<int>& s_upper_redundant_indices) const {
  SeparationCertificateProgram ret;
  ret.prog->AddIndeterminates(rational_forward_kin_.s());
  const auto& plane = separating_planes_[plane_geometries.plane_index];
  ret.prog->AddDecisionVariables(plane.decision_variables);

  AddRationalsNonnegative(ret.prog.get(), plane, d_minus_Cs, s_minus_s_lower_,
                          s_upper_minus_s_, C_redundant_indices,
                          s_lower_redundant_indices, s_upper_redundant_indices,
                          plane_geometries.positive_side_rationals,
                          plane.positive_side_geometry->body_index(),
                          this->map_body_to_monomial_basis_,
                          &ret.certificate.positive_side_rational_lagrangians);
  AddRationalsNonnegative(ret.prog.get(), plane, d_minus_Cs, s_minus_s_lower_,
                          s_upper_minus_s_, C_redundant_indices,
                          s_lower_redundant_indices, s_upper_redundant_indices,
                          plane_geometries.negative_side_rationals,
                          plane.negative_side_geometry->body_index(),
                          this->map_body_to_monomial_basis_,
                          &ret.certificate.negative_side_rational_lagrangians);
  if (!plane_geometries.positive_side_psd_mat.empty() ||
      !plane_geometries.negative_side_psd_mat.empty()) {
    ret.prog->AddIndeterminates(y_slack_);
  }

  switch (formulation_) {
    case Formulation::kFormulation1: {
      AddRationalsNonnegative(
          ret.prog.get(), plane, d_minus_Cs, s_minus_s_lower_, s_upper_minus_s_,
          C_redundant_indices, s_lower_redundant_indices,
          s_upper_redundant_indices, plane_geometries.positive_side_psd_mat,
          plane.positive_side_geometry->body_index(),
          this->map_body_to_monomial_basis_w_y_array_,
          &ret.certificate.positive_side_psd_mat_lagrangians);
      AddRationalsNonnegative(
          ret.prog.get(), plane, d_minus_Cs, s_minus_s_lower_, s_upper_minus_s_,
          C_redundant_indices, s_lower_redundant_indices,
          s_upper_redundant_indices, plane_geometries.negative_side_psd_mat,
          plane.negative_side_geometry->body_index(),
          this->map_body_to_monomial_basis_w_y_array_,
          &ret.certificate.negative_side_psd_mat_lagrangians);
      break;
    }
    case Formulation::kFormulation2: {
      AddRationalsNonnegative(
          ret.prog.get(), plane, d_minus_Cs, s_minus_s_lower_, s_upper_minus_s_,
          C_redundant_indices, s_lower_redundant_indices,
          s_upper_redundant_indices, plane_geometries.positive_side_psd_mat,
          plane.positive_side_geometry->body_index(),
          this->map_body_to_monomial_basis_w_y_,
          &ret.certificate.positive_side_psd_mat_lagrangians);
      AddRationalsNonnegative(
          ret.prog.get(), plane, d_minus_Cs, s_minus_s_lower_, s_upper_minus_s_,
          C_redundant_indices, s_lower_redundant_indices,
          s_upper_redundant_indices, plane_geometries.negative_side_psd_mat,
          plane.negative_side_geometry->body_index(),
          this->map_body_to_monomial_basis_w_y_,
          &ret.certificate.negative_side_psd_mat_lagrangians);
      break;
    }
  }

  return ret;
}

CspaceFreePolytope::SeparationCertificateResult
CspaceFreePolytope::SeparationCertificate::GetSolution(
    int plane_index, const Vector3<symbolic::Polynomial>& a,
    const symbolic::Polynomial& b,
    const VectorX<symbolic::Variable>& plane_decision_vars,
    const solvers::MathematicalProgramResult& result) const {
  CspaceFreePolytope::SeparationCertificateResult ret;
  ret.plane_index = plane_index;

  auto set_lagrangians =
      [&result](
          const std::vector<CspaceFreePolytope::SeparatingPlaneLagrangians>&
              lagrangians_vec,
          std::vector<CspaceFreePolytope::SeparatingPlaneLagrangians>*
              lagrangians_result) {
        lagrangians_result->reserve(lagrangians_vec.size());
        for (const auto& lagrangians : lagrangians_vec) {
          lagrangians_result->push_back(lagrangians.GetSolution(result));
        }
      };
  set_lagrangians(this->positive_side_rational_lagrangians,
                  &ret.positive_side_rational_lagrangians);
  set_lagrangians(this->negative_side_rational_lagrangians,
                  &ret.negative_side_rational_lagrangians);
  set_lagrangians(this->positive_side_psd_mat_lagrangians,
                  &ret.positive_side_psd_mat_lagrangians);
  set_lagrangians(this->negative_side_psd_mat_lagrangians,
                  &ret.negative_side_psd_mat_lagrangians);
  for (int i = 0; i < 3; ++i) {
    ret.a(i) = result.GetSolution(a(i));
  }
  ret.b = result.GetSolution(b);

  ret.plane_decision_var_vals = result.GetSolution(plane_decision_vars);
  return ret;
}

std::vector<std::optional<CspaceFreePolytope::SeparationCertificateResult>>
CspaceFreePolytope::FindSeparationCertificateGivenPolytope(
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const FindSeparationCertificateGivenPolytopeOptions& options) const {
  const VectorX<symbolic::Polynomial> d_minus_Cs = this->CalcDminusCs(C, d);
  std::unordered_set<int> C_redundant_indices;
  std::unordered_set<int> s_lower_redundant_indices;
  std::unordered_set<int> s_upper_redundant_indices;
  this->FindRedundantInequalities(
      C, d, this->s_lower_, this->s_upper_, 0., &C_redundant_indices,
      &s_lower_redundant_indices, &s_upper_redundant_indices);
  if (!options.ignore_redundant_C) {
    C_redundant_indices.clear();
  }

  // Stores the indices in separating_planes_ that don't appear in
  // ignored_collision_pairs.
  std::vector<int> active_plane_indices;
  active_plane_indices.reserve(separating_planes_.size());
  for (int i = 0; i < static_cast<int>(separating_planes_.size()); ++i) {
    if (ignored_collision_pairs.count(SortedPair<geometry::GeometryId>(
            separating_planes_[i].positive_side_geometry->id(),
            separating_planes_[i].negative_side_geometry->id())) == 0) {
      active_plane_indices.push_back(i);
    }
  }

  std::vector<std::optional<bool>> is_success(active_plane_indices.size(),
                                              std::nullopt);
  std::vector<std::optional<SeparationCertificateResult>> ret(
      active_plane_indices.size(), std::nullopt);

  // This lambda function formulates and solves a small SOS program for each
  // pair of geometries.
  auto solve_small_sos = [this, &d_minus_Cs, &C_redundant_indices,
                          &s_lower_redundant_indices,
                          &s_upper_redundant_indices, &active_plane_indices,
                          &options, &is_success, &ret](int plane_count) {
    const int plane_index = active_plane_indices[plane_count];
    auto certificate_program = this->ConstructPlaneSearchProgram(
        this->plane_geometries_[plane_index], d_minus_Cs, C_redundant_indices,
        s_lower_redundant_indices, s_upper_redundant_indices);
    solvers::MathematicalProgramResult result;
    solvers::MakeSolver(options.solver_id)
        ->Solve(*certificate_program.prog, std::nullopt, options.solver_options,
                &result);
    if (result.is_success()) {
      ret[plane_count].emplace(certificate_program.certificate.GetSolution(
          plane_index, separating_planes_[plane_index].a,
          separating_planes_[plane_index].b,
          separating_planes_[plane_index].decision_variables, result));
      is_success[plane_count].emplace(true);
    } else {
      ret[plane_count].reset();
      is_success[plane_count].emplace(false);
    }
    return plane_count;
  };

  const int num_threads =
      options.num_threads > 0
          ? options.num_threads
          : static_cast<int>(std::thread::hardware_concurrency());
  // We implement the "thread pool" idea here, by following
  // MonteCarloSimulationParallel class. This implementation doesn't use openMP
  // library.
  std::list<std::future<int>> active_operations;
  // Keep track of how many SOS have been dispatched already.
  int sos_dispatched = 0;
  // If any SOS is infeasible, then we don't dispatch any more SOS and report
  // failure.
  bool stop_dispatching = false;
  while ((active_operations.size() > 0 ||
          (sos_dispatched < static_cast<int>(active_plane_indices.size()) &&
           !stop_dispatching))) {
    // Check for completed operations.
    for (auto operation = active_operations.begin();
         operation != active_operations.end();) {
      if (IsFutureReady(*operation)) {
        // This call to future.get() is necessary to propagate any exception
        // thrown during SOS setup/solve.
        const int plane_count = operation->get();
        if (options.verbose) {
          drake::log()->info("SOS {}/{} completed, is_success {}", plane_count,
                             active_plane_indices.size(),
                             is_success[plane_count].value());
        }
        if (!(is_success[plane_count].value()) &&
            options.terminate_at_failure) {
          stop_dispatching = true;
        }
        // Erase returned iterator to the next node in the list.
        operation = active_operations.erase(operation);
      } else {
        // Advance to next node in the list.
        ++operation;
      }
    }

    // Dispatch new SOS.
    while (static_cast<int>(active_operations.size()) < num_threads &&
           sos_dispatched < static_cast<int>(active_plane_indices.size()) &&
           !stop_dispatching) {
      active_operations.emplace_back(std::async(
          std::launch::async, std::move(solve_small_sos), sos_dispatched));
      if (options.verbose) {
        drake::log()->info("SOS {}/{} dispatched", sos_dispatched,
                           active_plane_indices.size());
      }
      ++sos_dispatched;
    }

    // Wait a bit before checking for completion.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  if (std::all_of(is_success.begin(), is_success.end(),
                  [](std::optional<bool> flag) {
                    return flag.has_value() && flag.value();
                  })) {
    if (options.verbose) {
      drake::log()->info("Found Lagrangian multipliers and separating planes");
    }
  } else {
    if (options.verbose) {
      std::string bad_pairs;
      const auto& inspector = scene_graph_.model_inspector();
      for (int plane_count = 0;
           plane_count < static_cast<int>(active_plane_indices.size());
           ++plane_count) {
        const int plane_index = active_plane_indices[plane_count];
        if (is_success[plane_count].has_value() &&
            !(is_success[plane_count].value())) {
          bad_pairs.append(fmt::format(
              "({}, {})\n",
              inspector.GetName(
                  separating_planes_[plane_index].positive_side_geometry->id()),
              inspector.GetName(separating_planes_[plane_index]
                                    .negative_side_geometry->id())));
        }
      }

      drake::log()->warn(fmt::format(
          "Cannot find Lagrangian multipliers and separating planes for \n{}",
          bad_pairs));
    }
  }
  return ret;
}

bool CspaceFreePolytope::FindSeparationCertificateGivenPolytope(
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const CspaceFreePolytope::FindSeparationCertificateGivenPolytopeOptions&
        options,
    std::unordered_map<SortedPair<geometry::GeometryId>,
                       CspaceFreePolytope::SeparationCertificateResult>*
        certificates) const {
  const auto d_minus_Cs = this->CalcDminusCs<double>(C, d);

  std::unordered_set<int> C_redundant_indices;
  std::unordered_set<int> s_lower_redundant_indices;
  std::unordered_set<int> s_upper_redundant_indices;
  this->FindRedundantInequalities(
      C, d, s_lower_, s_upper_, 0., &C_redundant_indices,
      &s_lower_redundant_indices, &s_upper_redundant_indices);

  const auto certificates_vec = this->FindSeparationCertificateGivenPolytope(
      ignored_collision_pairs, C, d, options);

  certificates->clear();
  bool is_success = true;
  for (const auto& certificate : certificates_vec) {
    if (certificate.has_value()) {
      const auto& plane = separating_planes_[certificate->plane_index];
      certificates->emplace(
          SortedPair<geometry::GeometryId>(plane.positive_side_geometry->id(),
                                           plane.negative_side_geometry->id()),
          std::move(certificate.value()));
    } else {
      is_success = false;
    }
  }
  return is_success;
}

int CspaceFreePolytope::GetGramVarSizeForPolytopeSearchProgram(
    const CspaceFreePolytope::IgnoredCollisionPairs& ignored_collision_pairs,
    bool search_s_bounds_lagrangians) const {
  int ret = 0;

  // Count the gram size for rational >= 0 in C-space polytope.
  auto count_gram_per_plane_side = [this, search_s_bounds_lagrangians, &ret](
                                       int num_rationals,
                                       multibody::BodyIndex expressed_body,
                                       multibody::BodyIndex collision_body) {
    const auto& monomial_basis = this->map_body_to_monomial_basis_.at(
        SortedPair<multibody::BodyIndex>(expressed_body, collision_body));
    const int s_size = this->rational_forward_kin_.s().rows();
    ret += GetGramVarSizeGivenLagrangian(num_rationals, s_size,
                                         search_s_bounds_lagrangians,
                                         GetGramVarSize(monomial_basis));
  };

  auto count_psd_mat_gram_per_plane_side =
      [this, search_s_bounds_lagrangians, &ret](
          int num_psd_mat, multibody::BodyIndex expressed_body,
          multibody::BodyIndex collision_body) {
        if (num_psd_mat > 0) {
          const SortedPair<multibody::BodyIndex> body_pair(expressed_body,
                                                           collision_body);
          int num_gram_vars_per_lagrangian = 0;
          switch (this->formulation_) {
            case Formulation::kFormulation1: {
              num_gram_vars_per_lagrangian = GetGramVarSize(
                  this->map_body_to_monomial_basis_w_y_array_.at(body_pair));
              break;
            }
            case Formulation::kFormulation2: {
              num_gram_vars_per_lagrangian = GetGramVarSize(
                  this->map_body_to_monomial_basis_w_y_.at(body_pair));
              break;
            }
          }
          const int s_size = this->rational_forward_kin_.s().rows();
          ret += GetGramVarSizeGivenLagrangian(num_psd_mat, s_size,
                                               search_s_bounds_lagrangians,
                                               num_gram_vars_per_lagrangian);
        }
      };
  for (const auto& plane_geometries : plane_geometries_) {
    const auto& plane = separating_planes_[plane_geometries.plane_index];
    if (ignored_collision_pairs.count(SortedPair<geometry::GeometryId>(
            plane.positive_side_geometry->id(),
            plane.negative_side_geometry->id())) == 0) {
      count_gram_per_plane_side(plane_geometries.positive_side_rationals.size(),
                                plane.expressed_body,
                                plane.positive_side_geometry->body_index());
      count_gram_per_plane_side(plane_geometries.negative_side_rationals.size(),
                                plane.expressed_body,
                                plane.negative_side_geometry->body_index());
      count_psd_mat_gram_per_plane_side(
          plane_geometries.positive_side_psd_mat.size(), plane.expressed_body,
          plane.positive_side_geometry->body_index());
      count_psd_mat_gram_per_plane_side(
          plane_geometries.negative_side_psd_mat.size(), plane.expressed_body,
          plane.negative_side_geometry->body_index());
    }
  }
  return ret;
}

std::unique_ptr<solvers::MathematicalProgram>
CspaceFreePolytope::InitializePolytopeSearchProgram(
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const MatrixX<symbolic::Variable>& C, const VectorX<symbolic::Variable>& d,
    const VectorX<symbolic::Polynomial>& d_minus_Cs,
    const std::vector<std::optional<SeparationCertificateResult>>&
        certificates_vec,
    bool search_s_bounds_lagrangians, int gram_total_size,
    std::unordered_map<int, SeparationCertificate>* new_certificates) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(rational_forward_kin_.s());
  // Add the indeterminates y if we need to impose matrix sos constraint.
  if (std::any_of(plane_geometries_.begin(), plane_geometries_.end(),
                  [](const PlaneSeparatesGeometries& plane_geometries) {
                    return !plane_geometries.positive_side_psd_mat.empty() ||
                           !plane_geometries.negative_side_psd_mat.empty();
                  })) {
    prog->AddIndeterminates(y_slack_);
  }

  prog->AddDecisionVariables(Eigen::Map<const VectorX<symbolic::Variable>>(
      C.data(), C.rows() * C.cols()));
  prog->AddDecisionVariables(d);
  const auto gram_vars = prog->NewContinuousVariables(gram_total_size, "Gram");

  // plane_to_certificate_map maps the plane index to the index of certificate
  // in certificates_vec. Namely
  // certificates_vec[plane_to_certificate_map[i]]->plane_index = i
  std::unordered_map<int, int> plane_to_certificate_map;
  for (int i = 0; i < static_cast<int>(certificates_vec.size()); ++i) {
    plane_to_certificate_map.emplace(certificates_vec[i]->plane_index, i);
  }
  const int s_size = rational_forward_kin_.s().rows();
  int gram_var_count = 0;
  for (int plane_index = 0;
       plane_index < static_cast<int>(separating_planes_.size());
       ++plane_index) {
    const auto& plane = separating_planes_[plane_index];
    const SortedPair<geometry::GeometryId> geometry_pair(
        plane.positive_side_geometry->id(), plane.negative_side_geometry->id());
    if (ignored_collision_pairs.count(geometry_pair) == 0) {
      prog->AddDecisionVariables(plane.decision_variables);
      const auto& certificate =
          certificates_vec[plane_to_certificate_map.at(plane_index)];
      DRAKE_DEMAND(certificate->plane_index == plane_index);
      SeparationCertificate* new_certificate = nullptr;
      if (new_certificates != nullptr) {
        auto insertion_pair =
            new_certificates->emplace(plane_index, SeparationCertificate());
        new_certificate = &(insertion_pair.first->second);
      }
      DRAKE_DEMAND(certificate.has_value());

      // Add the constraint that positive_side_rationals are nonnegative in
      // C-space polytope.
      AddRationalsNonnegativeGivenLagrangian(
          prog.get(), d_minus_Cs, this->s_minus_s_lower_,
          this->s_upper_minus_s_, gram_vars, s_size,
          search_s_bounds_lagrangians,
          plane_geometries_[plane_index].positive_side_rationals,
          map_body_to_monomial_basis_, plane.expressed_body,
          plane.positive_side_geometry->body_index(),
          certificate->positive_side_rational_lagrangians, &gram_var_count,
          new_certificate == nullptr
              ? nullptr
              : &(new_certificate->positive_side_rational_lagrangians));
      // Add the constraint that negative_side_rationals are nonnegative in
      // C-space polytope.
      AddRationalsNonnegativeGivenLagrangian(
          prog.get(), d_minus_Cs, this->s_minus_s_lower_,
          this->s_upper_minus_s_, gram_vars, s_size,
          search_s_bounds_lagrangians,
          plane_geometries_[plane_index].negative_side_rationals,
          map_body_to_monomial_basis_, plane.expressed_body,
          plane.negative_side_geometry->body_index(),
          certificate->negative_side_rational_lagrangians, &gram_var_count,
          new_certificate == nullptr
              ? nullptr
              : &(new_certificate->negative_side_rational_lagrangians));
      switch (formulation_) {
        case Formulation::kFormulation1: {
          AddRationalsNonnegativeGivenLagrangian(
              prog.get(), d_minus_Cs, this->s_minus_s_lower_,
              this->s_upper_minus_s_, gram_vars, s_size,
              search_s_bounds_lagrangians,
              plane_geometries_[plane_index].positive_side_psd_mat,
              map_body_to_monomial_basis_w_y_array_, plane.expressed_body,
              plane.positive_side_geometry->body_index(),
              certificate->positive_side_psd_mat_lagrangians, &gram_var_count,
              new_certificate == nullptr
                  ? nullptr
                  : &(new_certificate->positive_side_psd_mat_lagrangians));
          AddRationalsNonnegativeGivenLagrangian(
              prog.get(), d_minus_Cs, this->s_minus_s_lower_,
              this->s_upper_minus_s_, gram_vars, s_size,
              search_s_bounds_lagrangians,
              plane_geometries_[plane_index].negative_side_psd_mat,
              map_body_to_monomial_basis_w_y_array_, plane.expressed_body,
              plane.negative_side_geometry->body_index(),
              certificate->negative_side_psd_mat_lagrangians, &gram_var_count,
              new_certificate == nullptr
                  ? nullptr
                  : &(new_certificate->negative_side_psd_mat_lagrangians));
          break;
        }
        case Formulation::kFormulation2: {
          AddRationalsNonnegativeGivenLagrangian(
              prog.get(), d_minus_Cs, this->s_minus_s_lower_,
              this->s_upper_minus_s_, gram_vars, s_size,
              search_s_bounds_lagrangians,
              plane_geometries_[plane_index].positive_side_psd_mat,
              map_body_to_monomial_basis_w_y_, plane.expressed_body,
              plane.positive_side_geometry->body_index(),
              certificate->positive_side_psd_mat_lagrangians, &gram_var_count,
              new_certificate == nullptr
                  ? nullptr
                  : &(new_certificate->positive_side_psd_mat_lagrangians));
          AddRationalsNonnegativeGivenLagrangian(
              prog.get(), d_minus_Cs, this->s_minus_s_lower_,
              this->s_upper_minus_s_, gram_vars, s_size,
              search_s_bounds_lagrangians,
              plane_geometries_[plane_index].negative_side_psd_mat,
              map_body_to_monomial_basis_w_y_, plane.expressed_body,
              plane.negative_side_geometry->body_index(),
              certificate->negative_side_psd_mat_lagrangians, &gram_var_count,
              new_certificate == nullptr
                  ? nullptr
                  : &(new_certificate->negative_side_psd_mat_lagrangians));
          break;
        }
      }
    }
  }
  DRAKE_DEMAND(gram_var_count == gram_total_size);
  return prog;
}

void CspaceFreePolytope::AddEllipsoidContainmentConstraint(
    solvers::MathematicalProgram* prog, const Eigen::MatrixXd& Q,
    const Eigen::VectorXd& s0, const MatrixX<symbolic::Variable>& C,
    const VectorX<symbolic::Variable>& d,
    const VectorX<symbolic::Variable>& ellipsoid_margins) const {
  DRAKE_DEMAND(Q.rows() == Q.cols());
  DRAKE_DEMAND((s0.array() <= s_upper_.array()).all());
  DRAKE_DEMAND((s0.array() >= s_lower_.array()).all());
  // Add the constraint |cᵢᵀQ|₂ ≤ dᵢ − cᵢᵀs0 − δᵢ as a Lorentz cone
  // constraint, namely [dᵢ − cᵢᵀs0 − δᵢ, cᵢᵀQ] is in the Lorentz cone. [dᵢ
  // − cᵢᵀs0 − δᵢ, cᵢᵀQ] = A_lorentz1 * [cᵢ, dᵢ, δᵢ] + b_lorentz1
  Eigen::MatrixXd A_lorentz1(Q.rows() + 1, 2 + C.cols());
  Eigen::VectorXd b_lorentz1(Q.rows() + 1);
  VectorX<symbolic::Variable> lorentz1_vars(2 + C.cols());
  for (int i = 0; i < C.rows(); ++i) {
    A_lorentz1.setZero();
    A_lorentz1.block(0, 0, 1, C.cols()) = -s0.transpose();
    A_lorentz1(0, C.cols()) = 1;
    A_lorentz1(0, C.cols() + 1) = -1;
    A_lorentz1.block(1, 0, Q.rows(), Q.cols()) = Q;
    b_lorentz1.setZero();
    lorentz1_vars << C.row(i).transpose(), d(i), ellipsoid_margins(i);
    prog->AddLorentzConeConstraint(A_lorentz1, b_lorentz1, lorentz1_vars);
  }
  // Add the constraint |cᵢ|₂ ≤ 1 as a Lorentz cone constraint that [1,
  // cᵢ] is in the Lorentz cone. [1, cᵢ] = A_lorentz2 * cᵢ + b_lorentz2
  Eigen::MatrixXd A_lorentz2 = Eigen::MatrixXd::Zero(1 + C.cols(), C.cols());
  A_lorentz2.bottomRows(C.cols()) =
      Eigen::MatrixXd::Identity(C.cols(), C.cols());
  Eigen::VectorXd b_lorentz2 = Eigen::VectorXd::Zero(1 + C.cols());
  b_lorentz2(0) = 1;
  for (int i = 0; i < C.rows(); ++i) {
    prog->AddLorentzConeConstraint(A_lorentz2, b_lorentz2,
                                   C.row(i).transpose());
  }
}

void CspaceFreePolytope::AddCspacePolytopeContainment(
    solvers::MathematicalProgram* prog, const MatrixX<symbolic::Variable>& C,
    const VectorX<symbolic::Variable>& d,
    const Eigen::MatrixXd& s_inner_pts) const {
  DRAKE_DEMAND(s_inner_pts.rows() == this->rational_forward_kin_.s().rows());
  // Check that s_inner_pts is within [s_lower_, s_upper_].
  for (int i = 0; i < s_inner_pts.rows(); ++i) {
    for (int j = 0; j < s_inner_pts.cols(); ++j) {
      if (s_inner_pts(i, j) > s_upper_(i)) {
        throw std::runtime_error(
            fmt::format("AddCspacePolytopeContainment(): s_inner_pts({}, "
                        "{})={}, larger than s_upper({})={}",
                        i, j, s_inner_pts(i, j), i, s_upper_(i)));
      }
      if (s_inner_pts(i, j) < s_lower_(i)) {
        throw std::runtime_error(
            fmt::format("AddCspacePolytopeContainment(): s_inner_pts({}, "
                        "{})={}, smaller than s_lower({})={}",
                        i, j, s_inner_pts(i, j), i, s_lower_(i)));
      }
    }
  }
  // We have the constraint C.row(i).dot(s_inner_pts.col(j)) <= d(i) for all i,
  // j. We can write this as s_inner_ptsᵀ * C.row(i)ᵀ <= [d(i);...;d(i)] We
  // repeat this constraint for each row and concantenate it into the matrix
  // form blockdiag(s_inner_ptsᵀ, ..., s_inner_ptsᵀ) * [C.row(0)ᵀ;
  // C.row(1)ᵀ;...;C.row(n-1)] - blockdiag(𝟏, 𝟏, ..., 𝟏) * d <= 0
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(
      s_inner_pts.cols() * C.rows(), (s_inner_pts.rows() + 1) * C.rows());
  VectorX<symbolic::Variable> vars(A.cols());
  for (int i = 0; i < C.rows(); ++i) {
    A.block(i * s_inner_pts.cols(), i * (s_inner_pts.rows() + 1),
            s_inner_pts.cols(), s_inner_pts.rows()) = s_inner_pts.transpose();
    A.block(i * s_inner_pts.cols(),
            i * (s_inner_pts.rows() + 1) + s_inner_pts.rows(),
            s_inner_pts.cols(), 1) = -Eigen::VectorXd::Ones(s_inner_pts.cols());
    vars.segment((s_inner_pts.rows() + 1) * i, s_inner_pts.rows()) =
        C.row(i).transpose();
    vars((s_inner_pts.rows() + 1) * i + s_inner_pts.rows()) = d(i);
  }
  prog->AddLinearConstraint(A, Eigen::VectorXd::Constant(A.rows(), -kInf),
                            Eigen::VectorXd::Zero(A.rows()), vars);
}

std::optional<CspaceFreePolytope::FindPolytopeGivenLagrangianResult>
CspaceFreePolytope::FindPolytopeGivenLagrangian(
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const MatrixX<symbolic::Variable>& C, const VectorX<symbolic::Variable>& d,
    const VectorX<symbolic::Polynomial>& d_minus_Cs,
    const std::vector<std::optional<SeparationCertificateResult>>&
        certificates_vec,
    const Eigen::MatrixXd& Q, const Eigen::VectorXd& s0,
    const VectorX<symbolic::Variable>& ellipsoid_margins, int gram_total_size,
    const FindPolytopeGivenLagrangianOptions& options,
    std::unordered_map<int, SeparationCertificateResult>* certificates_result)
    const {
  std::unordered_map<int, SeparationCertificate> new_certificates;
  auto prog = this->InitializePolytopeSearchProgram(
      ignored_collision_pairs, C, d, d_minus_Cs, certificates_vec,
      options.search_s_bounds_lagrangians, gram_total_size,
      certificates_result == nullptr ? nullptr : &new_certificates);
  prog->AddDecisionVariables(ellipsoid_margins);
  AddEllipsoidContainmentConstraint(prog.get(), Q, s0, C, d, ellipsoid_margins);
  // We know that the verified polytope has to be contained in the box s_lower
  // <= s <= s_upper. Hence there is no point to grow the polytope such that any
  // of its halfspace C.row(i) * s <= d(i) contains the entire box s_lower <= s
  // <= s_upper. Therefore an upper bound of the margin δ is the maximal
  // distance from any vertices of the box s_lower <= s <= s_upper to the
  // ellipsoid. Computing the distance from a point to the hyperellipsoid is
  // non-trivial (there is not closed-form solution). Here we use an upper bound
  // of this distance, which is the maximal distance between any two points
  // within the box.
  const double margin_upper_bound = (s_upper_ - s_lower_).norm();
  prog->AddBoundingBoxConstraint(0, margin_upper_bound, ellipsoid_margins);
  if (options.s_inner_pts.has_value()) {
    for (int i = 0; i < options.s_inner_pts->cols(); ++i) {
      DRAKE_DEMAND(
          (options.s_inner_pts->col(i).array() <= s_upper_.array()).all());
      DRAKE_DEMAND(
          (options.s_inner_pts->col(i).array() >= s_lower_.array()).all());
    }
    // Add the constraint C * s_inner_pts <= d
    AddCspacePolytopeContainment(prog.get(), C, d, options.s_inner_pts.value());
  }

  switch (options.ellipsoid_margin_cost) {
    case CspaceFreePolytope::EllipsoidMarginCost::kSum:
      // Maximize ∑ᵢ δᵢ
      prog->AddLinearCost(-Eigen::VectorXd::Ones(ellipsoid_margins.rows()), 0,
                          ellipsoid_margins);
      break;
    case CspaceFreePolytope::EllipsoidMarginCost::kGeometricMean:
      // Maximize ∏ᵢ (δᵢ + ε)
      prog->AddMaximizeGeometricMeanCost(
          Eigen::MatrixXd::Identity(ellipsoid_margins.rows(),
                                    ellipsoid_margins.rows()),
          Eigen::VectorXd::Constant(ellipsoid_margins.rows(),
                                    options.ellipsoid_margin_epsilon),
          ellipsoid_margins);
      break;
  }

  const solvers::MathematicalProgramResult result =
      SolveWithBackoff(prog.get(), options.backoff_scale,
                       options.solver_options, options.solver_id);
  if (result.is_success()) {
    CspaceFreePolytope::FindPolytopeGivenLagrangianResult ret;
    ret.C = result.GetSolution(C);
    ret.d = result.GetSolution(d);
    for (int plane_index = 0;
         plane_index < static_cast<int>(this->separating_planes_.size());
         ++plane_index) {
      const auto& plane = this->separating_planes_[plane_index];
      const SortedPair<geometry::GeometryId> geometry_pair(
          plane.positive_side_geometry->id(),
          plane.negative_side_geometry->id());
      if (ignored_collision_pairs.count(geometry_pair) == 0) {
        Vector3<symbolic::Polynomial> a;
        for (int i = 0; i < 3; ++i) {
          a(i) = result.GetSolution(plane.a(i));
        }
        ret.a.emplace(plane_index, a);
        ret.b.emplace(plane_index, result.GetSolution(plane.b));
        ret.ellipsoid_margins = result.GetSolution(ellipsoid_margins);
      }
    }

    if (certificates_result != nullptr) {
      certificates_result->clear();
      for (int plane_index = 0;
           plane_index < static_cast<int>(separating_planes_.size());
           ++plane_index) {
        const auto& plane = separating_planes_[plane_index];
        if (ignored_collision_pairs.count(SortedPair<geometry::GeometryId>(
                plane.positive_side_geometry->id(),
                plane.negative_side_geometry->id())) == 0) {
          certificates_result->emplace(
              plane_index,
              new_certificates.at(plane_index)
                  .GetSolution(
                      plane_index, separating_planes_[plane_index].a,
                      separating_planes_[plane_index].b,
                      separating_planes_[plane_index].decision_variables,
                      result));
        }
      }
    }
    return ret;
  } else {
    return std::nullopt;
  }
}

HPolyhedron CspaceFreePolytope::GetPolyhedronWithJointLimits(
    const Eigen::MatrixXd& C, const Eigen::VectorXd& d) const {
  const int s_size = rational_forward_kin_.s().rows();
  Eigen::MatrixXd A(C.rows() + 2 * s_size, s_size);
  Eigen::VectorXd b(A.rows());
  A.topRows(C.rows()) = C;
  b.head(C.rows()) = d;
  A.middleRows(C.rows(), s_size) = Eigen::MatrixXd::Identity(s_size, s_size);
  b.segment(C.rows(), s_size) = s_upper_;
  A.bottomRows(s_size) = -Eigen::MatrixXd::Identity(s_size, s_size);
  b.tail(s_size) = -s_lower_;
  return HPolyhedron(A, b);
}

void CspaceFreePolytope::SearchResult::SetSeparatingPlanes(
    std::vector<std::optional<SeparationCertificateResult>>
        certificates_result) {
  a.clear();
  b.clear();
  for (const auto& certificate : certificates_result) {
    a.emplace(certificate->plane_index, std::move(certificate->a));
    b.emplace(certificate->plane_index, std::move(certificate->b));
  }
}

std::optional<CspaceFreePolytope::SearchResult>
CspaceFreePolytope::SearchWithBilinearAlternation(
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C_init,
    const Eigen::Ref<const Eigen::VectorXd>& d_init,
    const BilinearAlternationOptions& options) const {
  DRAKE_DEMAND(C_init.rows() == d_init.rows());
  DRAKE_DEMAND(C_init.cols() == this->rational_forward_kin_.s().rows());
  std::optional<CspaceFreePolytope::SearchResult> ret{std::nullopt};
  int iter = 0;
  // When we search for the C-space polytope {s |C*s<=d, s_lower<=s<=s_upper},
  // we will require that each row of C has length <= 1. Hence to start with a
  // feasible solution, we normalize each row of C and d.
  Eigen::MatrixXd C = C_init;
  Eigen::VectorXd d = d_init;
  for (int i = 0; i < C.rows(); ++i) {
    const double C_row_norm = C.row(i).norm();
    C.row(i) = C.row(i) / C_row_norm;
    d(i) = d(i) / C_row_norm;
  }
  // Create symbolic variable for C and d
  MatrixX<symbolic::Variable> C_var(C_init.rows(), C_init.cols());
  VectorX<symbolic::Variable> d_var(d_init.rows());
  // Create symbolic variable for ellipsoid margins.
  VectorX<symbolic::Variable> ellipsoid_margins(C_init.rows());
  for (int i = 0; i < C_init.rows(); ++i) {
    d_var(i) = symbolic::Variable("d" + std::to_string(i));
    ellipsoid_margins(i) =
        symbolic::Variable("ellipsoid_margin" + std::to_string(i));
    for (int j = 0; j < C_init.cols(); ++j) {
      C_var(i, j) = symbolic::Variable(fmt::format("C({},{})", i, j));
    }
  }
  const VectorX<symbolic::Polynomial> d_minus_Cs =
      this->CalcDminusCs<symbolic::Variable>(C_var, d_var);
  const int gram_total_size_in_polytope_program =
      this->GetGramVarSizeForPolytopeSearchProgram(
          ignored_collision_pairs,
          options.find_polytope_options.search_s_bounds_lagrangians);
  // Find the inscribed ellipsoid.
  HPolyhedron cspace_polytope = this->GetPolyhedronWithJointLimits(C, d);
  Hyperellipsoid ellipsoid = cspace_polytope.MaximumVolumeInscribedEllipsoid();
  DRAKE_DEMAND(options.ellipsoid_scaling >= 0);
  DRAKE_DEMAND(options.ellipsoid_scaling <= 1);
  Eigen::MatrixXd ellipsoid_Q =
      options.ellipsoid_scaling * (ellipsoid.A().inverse());
  double prev_cost = ellipsoid_Q.determinant();
  drake::log()->info("det(Q) at the beginning is {}", prev_cost);
  while (iter < options.max_iter) {
    const std::vector<std::optional<SeparationCertificateResult>>
        certificates_result = this->FindSeparationCertificateGivenPolytope(
            ignored_collision_pairs, C, d, options.find_lagrangian_options);
    if (std::any_of(
            certificates_result.begin(), certificates_result.end(),
            [](const std::optional<SeparationCertificateResult>& certificate) {
              return !certificate.has_value();
            })) {
      drake::log()->error(
          "Cannot find the separation certificate at iteration {} given the "
          "polytope.",
          iter);
      break;
    } else {
      if (!ret.has_value()) {
        ret.emplace();
      }
      // Copy the result to ret.
      ret->C = C;
      ret->d = d;
      ret->SetSeparatingPlanes(certificates_result);
    }
    // Now fix the Lagrangian and search for C-space polytope and separating
    // planes.
    const auto polytope_result = this->FindPolytopeGivenLagrangian(
        ignored_collision_pairs, C_var, d_var, d_minus_Cs, certificates_result,
        ellipsoid_Q, ellipsoid.center(), ellipsoid_margins,
        gram_total_size_in_polytope_program, options.find_polytope_options,
        nullptr /* certificates_result */);
    if (polytope_result.has_value()) {
      C = polytope_result->C;
      d = polytope_result->d;
      ret->C = polytope_result->C;
      ret->d = polytope_result->d;
      ret->a = std::move(polytope_result->a);
      ret->b = std::move(polytope_result->b);
      ret->num_iter = iter;
      // Now find the inscribed ellipsoid.
      cspace_polytope = this->GetPolyhedronWithJointLimits(ret->C, ret->d);
      ellipsoid = cspace_polytope.MaximumVolumeInscribedEllipsoid();
      ellipsoid_Q = options.ellipsoid_scaling * (ellipsoid.A().inverse());
      const double cost = ellipsoid_Q.determinant();
      drake::log()->info("Iteration {}: det(Q)={}", iter, cost);
      if (cost - prev_cost < options.convergence_tol) {
        break;
      } else {
        prev_cost = cost;
      }
    } else {
      drake::log()->error(
          "Cannot find the separation certificate at iteration {} given the "
          "Lagrangians.",
          iter);
      break;
    }
    ++iter;
  }
  return ret;
}

std::optional<CspaceFreePolytope::SearchResult>
CspaceFreePolytope::BinarySearch(
    const IgnoredCollisionPairs& ignored_collision_pairs,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d_init,
    const Eigen::Ref<const Eigen::VectorXd>& s_center,
    const BinarySearchOptions& options) const {
  DRAKE_DEMAND(options.scale_max >= options.scale_min &&
               options.scale_min >= 0);
  DRAKE_DEMAND(((C * s_center).array() <= d_init.array()).all());
  DRAKE_DEMAND((s_center.array() >= s_lower_.array()).all());
  DRAKE_DEMAND((s_center.array() <= s_upper_.array()).all());
  CspaceFreePolytope::SearchResult ret;

  const Eigen::ArrayXd C_row_norm = C.rowwise().norm().array();
  if ((C_row_norm == 0).any()) {
    throw std::runtime_error(
        "C contains rows with all 0 entries. Please remove these rows.");
  }

  // Determines if we can certify the scaled C-space polytope {s |C*s<=d,
  // s_lower<=s<=s_upper} is collision free or not. Also update `ret` if eps is
  // feasible.
  auto is_scale_feasible = [this, &ignored_collision_pairs, &C, &d_init,
                            &s_center, &options, &ret](double scale) {
    // (d - C*s_center) / |C| = scale * (d_init - C*s_center) / |C|, hence d =
    // scale * d_init + (1-scale) * C * s_center.
    const Eigen::VectorXd d = scale * d_init + (1 - scale) * C * s_center;
    const auto certificates_result =
        this->FindSeparationCertificateGivenPolytope(
            ignored_collision_pairs, C, d, options.find_lagrangian_options);
    if (std::any_of(
            certificates_result.begin(), certificates_result.end(),
            [](const std::optional<SeparationCertificateResult>& certificate) {
              return !certificate.has_value();
            })) {
      return false;
    } else {
      ret.C = C;
      ret.d = d;
      ret.SetSeparatingPlanes(std::move(certificates_result));
      return true;
    }
  };

  if (!is_scale_feasible(options.scale_min)) {
    drake::log()->error(
        "CspaceFreePolytope::BinarySearch(): scale_min={} is infeasible.",
        options.scale_min);
    return std::nullopt;
  }
  if (is_scale_feasible(options.scale_max)) {
    drake::log()->info(
        "CspaceFreePolytope::BinarySearch(): scale_max={} is feasible.",
        options.scale_max);
    ret.num_iter = 0;
    return ret;
  }
  double scale_min = options.scale_min;
  double scale_max = options.scale_max;
  int iter = 0;
  while (scale_max - scale_min > options.convergence_tol &&
         iter < options.max_iter) {
    const double scale = (scale_max + scale_min) / 2;
    if (is_scale_feasible(scale)) {
      drake::log()->info(
          "CspaceFreePolytope::BinarySearch(): scale={} is feasible", scale);
      scale_min = scale;
    } else {
      drake::log()->info(
          "CspaceFreePolytope::BinarySearch(): scale={} is infeasible", scale);
      scale_max = scale;
    }
    ++iter;
  }
  ret.num_iter = iter;
  return ret;
}

std::map<multibody::BodyIndex, std::vector<std::unique_ptr<CollisionGeometry>>>
GetCollisionGeometries(const multibody::MultibodyPlant<double>& plant,
                       const geometry::SceneGraph<double>& scene_graph) {
  std::map<multibody::BodyIndex,
           std::vector<std::unique_ptr<CollisionGeometry>>>
      ret;
  const auto& inspector = scene_graph.model_inspector();

  for (multibody::BodyIndex body_index{0}; body_index < plant.num_bodies();
       ++body_index) {
    const std::optional<geometry::FrameId> frame_id =
        plant.GetBodyFrameIdIfExists(body_index);
    if (frame_id.has_value()) {
      const auto geometry_ids =
          inspector.GetGeometries(frame_id.value(), geometry::Role::kProximity);
      for (const auto& geometry_id : geometry_ids) {
        auto collision_geometry = std::make_unique<CollisionGeometry>(
            &(inspector.GetShape(geometry_id)), body_index, geometry_id,
            inspector.GetPoseInFrame(geometry_id));
        auto body_it = ret.find(body_index);
        if (body_it == ret.end()) {
          std::vector<std::unique_ptr<CollisionGeometry>> body_geometries;
          body_geometries.push_back(std::move(collision_geometry));
          ret.emplace_hint(body_it, body_index, std::move(body_geometries));
        } else {
          body_it->second.push_back(std::move(collision_geometry));
        }
      }
    }
  }
  return ret;
}

// Explicit instantiation
template VectorX<symbolic::Polynomial> CspaceFreePolytope::CalcDminusCs<double>(
    const Eigen::Ref<const Eigen::MatrixXd>&,
    const Eigen::Ref<const Eigen::VectorXd>&) const;
template VectorX<symbolic::Polynomial>
CspaceFreePolytope::CalcDminusCs<symbolic::Variable>(
    const Eigen::Ref<const MatrixX<symbolic::Variable>>&,
    const Eigen::Ref<const VectorX<symbolic::Variable>>&) const;
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
