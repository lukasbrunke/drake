#include "drake/geometry/optimization/cspace_free_box.h"

#include <map>
#include <memory>

#include "drake/common/fmt_eigen.h"
#include "drake/geometry/optimization/cspace_free_internal.h"

namespace drake {
namespace geometry {
namespace optimization {

CspaceFreeBox::SeparationCertificateResult
CspaceFreeBox::SeparationCertificate::GetSolution(
    int plane_index, const Vector3<symbolic::Polynomial>& a,
    const symbolic::Polynomial& b,
    const VectorX<symbolic::Variable>& plane_decision_vars,
    const solvers::MathematicalProgramResult& result) const {
  CspaceFreeBox::SeparationCertificateResult ret{};
  ret.plane_index = plane_index;

  auto set_lagrangians =
      [&result](const std::vector<CspaceFreeBox::SeparatingPlaneLagrangians>&
                    lagrangians_vec,
                std::vector<CspaceFreeBox::SeparatingPlaneLagrangians>*
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
  for (int i = 0; i < 3; ++i) {
    ret.a(i) = result.GetSolution(a(i));
  }
  ret.b = result.GetSolution(b);

  ret.plane_decision_var_vals = result.GetSolution(plane_decision_vars);
  return ret;
}

CspaceFreeBox::CspaceFreeBox(const multibody::MultibodyPlant<double>* plant,
                             const geometry::SceneGraph<double>* scene_graph,
                             SeparatingPlaneOrder plane_order,
                             const Options& options)
    : CspaceFreePolytopeBase(plant, scene_graph, plane_order, options) {}

CspaceFreeBox::~CspaceFreeBox() {}

CspaceFreeBox::SeparatingPlaneLagrangians
CspaceFreeBox::SeparatingPlaneLagrangians::GetSolution(
    const solvers::MathematicalProgramResult& result) const {
  CspaceFreeBox::SeparatingPlaneLagrangians ret(this->s_box_lower_.rows());
  for (int i = 0; i < this->s_box_lower_.rows(); ++i) {
    ret.s_box_lower_(i) = result.GetSolution(this->s_box_lower_(i));
    ret.s_box_upper_(i) = result.GetSolution(this->s_box_upper_(i));
  }
  return ret;
}

void CspaceFreeBox::ComputeSBox(
    const Eigen::Ref<const Eigen::VectorXd>& q_box_lower,
    const Eigen::Ref<const Eigen::VectorXd>& q_box_upper,
    Eigen::VectorXd* s_box_lower, Eigen::VectorXd* s_box_upper,
    Eigen::VectorXd* q_star) const {
  if ((q_box_lower.array() > q_box_upper.array()).any()) {
    throw std::runtime_error(
        fmt::format("CspaceFreeBox: q_box_lower={} has some entries larger "
                    "than q_box_upper={}.",
                    fmt_eigen(q_box_lower.transpose()),
                    fmt_eigen(q_box_upper.transpose())));
  }
  const auto& plant = this->rational_forward_kin().plant();
  const Eigen::VectorXd q_position_lower = plant.GetPositionLowerLimits();
  const Eigen::VectorXd q_position_upper = plant.GetPositionUpperLimits();
  if ((q_box_lower.array() > q_position_upper.array()).any()) {
    throw std::runtime_error(fmt::format(
        "CspaceFreeBox: q_box_lower={} has some entries larger the the robot "
        "position upper limit={}.",
        fmt_eigen(q_box_lower.transpose()),
        fmt_eigen(q_position_upper.transpose())));
  }
  if ((q_box_upper.array() < q_position_lower.array()).any()) {
    throw std::runtime_error(fmt::format(
        "CspaceFreeBox: q_box_upper={} has some entries smaller than the robot "
        "position lower limit={}.",
        fmt_eigen(q_box_upper.transpose()),
        fmt_eigen(q_position_lower.transpose())));
  }
  const Eigen::VectorXd q_lower =
      q_box_lower.array().max(q_position_lower.array()).matrix();
  const Eigen::VectorXd q_upper =
      q_box_upper.array().min(q_position_upper.array()).matrix();
  *q_star = 0.5 * (q_lower + q_upper);
  *s_box_lower = this->rational_forward_kin().ComputeSValue(q_lower, *q_star);
  *s_box_upper = this->rational_forward_kin().ComputeSValue(q_upper, *q_star);
}

void CspaceFreeBox::GeneratePolynomialsToCertify(
    const Eigen::Ref<const Eigen::VectorXd>& s_box_lower,
    const Eigen::Ref<const Eigen::VectorXd>& s_box_upper,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const IgnoredCollisionPairs& ignored_collision_pairs,
    PolynomialsToCertify* certify_polynomials) const {
  this->CalcSBoundsPolynomial(s_box_lower, s_box_upper,
                              &(certify_polynomials->s_minus_s_box_lower),
                              &(certify_polynomials->s_box_upper_minus_s));

  std::map<int, const CSpaceSeparatingPlane<symbolic::Variable>*>
      separating_planes_map;
  for (int i = 0; i < static_cast<int>(separating_planes().size()); ++i) {
    const auto& plane = separating_planes()[i];
    if (ignored_collision_pairs.count(SortedPair<geometry::GeometryId>(
            plane.positive_side_geometry->id(),
            plane.negative_side_geometry->id())) == 0) {
      separating_planes_map.emplace(i, &plane);
    }
  }

  internal::GenerateRationals(separating_planes_map, y_slack(), q_star,
                              rational_forward_kin(),
                              &(certify_polynomials->plane_geometries));
}

CspaceFreeBox::SeparationCertificateProgram
CspaceFreeBox::ConstructPlaneSearchProgram(
    const PlaneSeparatesGeometries& plane_geometries,
    const VectorX<symbolic::Polynomial>& s_minus_s_lower,
    const VectorX<symbolic::Polynomial>& s_upper_minus_s) const {
  SeparationCertificateProgram ret;
  ret.plane_index = plane_geometries.plane_index;
  ret.prog->AddIndeterminates(rational_forward_kin().s());
  const auto& plane = separating_planes()[plane_geometries.plane_index];
  ret.prog->AddDecisionVariables(plane.decision_variables);

  // First count the total size of the gram matrix variables.
  int gram_var_count = 0;
  auto count_gram = [this, &s_minus_s_lower, &s_upper_minus_s](
                        const symbolic
                        : RationalFunction& rational,
                          const std::array<VectorX<symbolic::Monomial>, 4>&
                              monomial_basis_array) {
    // Each rational >= 0 requires the Lagrangian multiplier for s-s_lower and
    // s_upper - s.
    const int
  }
}
}  // namespace optimization
}  // namespace geometry
}  // namespace drake
