#include "drake/multibody/rational_forward_kinematics/sweeping_volume_outer_approximation.h"

#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace multibody {
SweepingVolumeOuterApproximation::SweepingVolumeOuterApproximation(
    const MultibodyPlant<double>& plant,
    const std::vector<std::shared_ptr<const ConvexPolytope>>& link_polytopes,
    const Eigen::Ref<const Eigen::VectorXd>& q_star)
    : rational_forward_kin_{plant},
      link_polytopes_{link_polytopes},
      q_star_{q_star} {
  const int num_link_polytopes = static_cast<int>(link_polytopes_.size());
  p_WV_poly_.resize(num_link_polytopes);
  const std::vector<RationalForwardKinematics::Pose<symbolic::Polynomial>>
      X_WB = rational_forward_kin_.CalcLinkPosesAsMultilinearPolynomials(
          q_star, plant.world_body().index());

  for (int i = 0; i < num_link_polytopes; ++i) {
    link_polytope_id_to_index_.emplace(link_polytopes_[i]->get_id(), i);
    p_WV_poly_[i].reserve(link_polytopes_[i]->p_BV().cols());
    const BodyIndex link_index = link_polytopes_[i]->body_index();
    for (int j = 0; j < link_polytopes_[i]->p_BV().cols(); ++j) {
      p_WV_poly_[i].push_back(X_WB[link_index].R_AB *
                                  link_polytopes_[i]->p_BV().col(j) +
                              X_WB[link_index].p_AB);
    }
  }
}

/**
 * We can find an upper bound on max nᵀp_WQ s.t q_lower <= q <= q_upper through
 * the SOS program
 * min d
 * s.t d - nᵀp_WQ - l_lower * (t - t_lower) - l_upper(t_upper - t) is SOS
 *     l_lower, l_upper is SOS
 */
double SweepingVolumeOuterApproximation::FindSweepingVolumeMaximalProjection(
    ConvexGeometry::Id link_polytope_id,
    const Eigen::Ref<const Eigen::Vector3d>& n_W,
    const Eigen::Ref<const Eigen::VectorXd>& q_lower,
    const Eigen::Ref<const Eigen::VectorXd>& q_upper,
    const SweepingVolumeOuterApproximation::VerificationOption&
        verification_option) const {
  DRAKE_DEMAND(q_lower.rows() == rational_forward_kin_.plant().num_positions());
  // First q_lower <= q_star <= q_upper
  DRAKE_DEMAND((q_lower.array() <= q_star_.array()).all());
  DRAKE_DEMAND((q_upper.array() >= q_star_.array()).all());
  DRAKE_DEMAND(((q_upper - q_star_).array() <= M_PI).all());
  DRAKE_DEMAND(((q_star_ - q_lower).array() <= M_PI).all());
  const int link_polytope_index =
      link_polytope_id_to_index_.at(link_polytope_id);

  solvers::MathematicalProgram prog;
  prog.AddIndeterminates(rational_forward_kin_.t());
  auto d = prog.NewContinuousVariables<1>()(0);
  prog.AddLinearCost(d);
  const symbolic::Monomial monomial_one{};
  const symbolic::Polynomial d_poly({{monomial_one, d}});

  const VectorX<symbolic::Variable> t_on_path =
      rational_forward_kin_.FindTOnPath(
          rational_forward_kin_.plant().world_body().index(),
          link_polytopes_[link_polytope_index]->body_index());
  std::vector<symbolic::Polynomial> t_minus_t_lower, t_upper_minus_t;
  t_minus_t_lower.reserve(t_on_path.rows());
  t_upper_minus_t.reserve(t_on_path.rows());
  const Eigen::VectorXd t_lower =
      rational_forward_kin_.ComputeTValue(q_lower, q_star_);
  const Eigen::VectorXd t_upper =
      rational_forward_kin_.ComputeTValue(q_upper, q_star_);
  for (int i = 0; i < t_on_path.rows(); ++i) {
    const int t_index =
        rational_forward_kin_.t_id_to_index().at(t_on_path(i).get_id());
    const symbolic::Monomial ti_monomial{rational_forward_kin_.t()(t_index)};
    t_minus_t_lower.emplace_back(symbolic::Polynomial::MapType{
        {ti_monomial, 1}, {monomial_one, -t_lower(t_index)}});
    t_upper_minus_t.emplace_back(symbolic::Polynomial::MapType{
        {ti_monomial, -1}, {monomial_one, t_upper(t_index)}});
  }
  const VectorX<symbolic::Monomial> monomial_basis =
      GenerateMonomialBasisWithOrderUpToOne(symbolic::Variables(t_on_path));
  for (const auto& p_WVi_poly : p_WV_poly_[link_polytope_index]) {
    const symbolic::RationalFunction d_minus_n_W_times_p_WVi =
        rational_forward_kin_.ConvertMultilinearPolynomialToRationalFunction(
            d_poly - p_WVi_poly.dot(n_W));
    symbolic::Polynomial verified_polynomial =
        d_minus_n_W_times_p_WVi.numerator();
    for (int i = 0; i < t_on_path.rows(); ++i) {
      const auto l_lower_i =
          prog.NewNonnegativePolynomial(monomial_basis,
                                        verification_option.lagrangian_type)
              .first;
      const auto l_upper_i =
          prog.NewNonnegativePolynomial(monomial_basis,
                                        verification_option.lagrangian_type)
              .first;
      verified_polynomial -= l_lower_i * t_minus_t_lower[i];
      verified_polynomial -= l_upper_i * t_upper_minus_t[i];
    }
    const auto verified_polynomial_expected =
        prog.NewNonnegativePolynomial(monomial_basis,
                                      verification_option.link_polynomial_type)
            .first;
    const auto diff = verified_polynomial_expected - verified_polynomial;
    for (const auto& term : diff.monomial_to_coefficient_map()) {
      prog.AddLinearConstraint(term.second == 0);
    }
  }

  solvers::MosekSolver solver;
  solver.set_stream_logging(true, "");
  solvers::MathematicalProgramResult result;
  solver.Solve(prog, {}, {}, &result);
  return result.get_optimal_cost();
}

}  // namespace multibody
}  // namespace drake
