#pragma once

#include <functional>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/system.h"

namespace drake {
namespace systems {
namespace analysis {

/// Sets up a linear program to search for the coefficients of a
/// Lyapunov function that satisfies the Lyapunov conditions at a set
/// of sample points.
///   ∀xᵢ, V(xᵢ) ≥ 0,
///   ∀xᵢ, V̇(xᵢ) = ∂V/∂x f(xᵢ) ≤ 0.
/// In order to provide boundary conditions to the problem, and improve
/// numerical conditioning, we additionally impose the constraint
///   V(x₀) = 0,
/// and add an objective that pushes V̇(xᵢ) towards -1 (time-to-go):
///   min ∑ |V̇(xᵢ) + 1|.
///
/// For background, and a description of this algorithm, see
/// http://underactuated.csail.mit.edu/underactuated.html?chapter=lyapunov .
/// It currently requires that the system to be optimized has only continuous
/// state and it is assumed to be time invariant.
///
/// @param system to be verified.  We currently require that the system has
/// only continuous state, and it is assumed to be time invariant.  Unlike
/// many analysis algorithms, the system does *not* need to support conversion
/// to other ScalarTypes (double is sufficient).
///
/// @param context is used only to specify any parameters of the system, and to
/// fix any input ports.  The system/context must have all inputs assigned.
///
/// @param basis_functions must define an AutoDiffXd function that takes the
/// state vector as an input argument and returns the vector of values of the
/// basis functions at that state.  The Lyapunov function will then have the
/// form
///   V(x) = ∑ pᵢ φᵢ(x),
/// where `p` is the vector to be solved for and `φ(x)` is the vector of
/// basis function evaluations returned by this function.
///
/// @param state_samples is a list of sample states (one per column) at which
/// to apply the optimization constraints and the objective.
///
/// @param V_zero_state is a particular state, x₀, where we impose the
/// condition: V(x₀) = 0.
///
/// @return params the VectorXd of parameters, p, that satisfies the Lyapunov
/// conditions described above.  The resulting Lyapunov function is
///   V(x) = ∑ pᵢ φᵢ(x),
///
/// @ingroup analysis
Eigen::VectorXd SampleBasedLyapunovAnalysis(
    const System<double>& system, const Context<double>& context,
    const std::function<VectorX<AutoDiffXd>(const VectorX<AutoDiffXd>& state)>&
        basis_functions,
    const Eigen::Ref<const Eigen::MatrixXd>& state_samples,
    const Eigen::Ref<const Eigen::VectorXd>& V_zero_state);

/// Constructs a program to find the common Lyapunov function and controller for
/// a set of linear systems
/// <pre>
/// ẋ = A[0]*x + B[0]*u
/// ẋ = A[1]*x + B[1]*u
///       ...
/// ẋ = A[n-1]*x + B[n-1]*u
/// </pre>
/// We assume that common Lyapunov function is xᵀPx with the controller u = Kx
/// The condition for the common Lyapunov function is
/// <pre>
/// P(A[i]+B[i]K) + (A[i]+B[i]K)ᵀP  ≼ 0     (1)
/// P ≽ 0
/// </pre>
/// To form (1) as an LMI, we do multiply P⁻¹ from both left and right in (1),
/// and introduce two new variables M=P⁻¹, N = KP⁻¹, with the following LMI
/// <pre>
/// A[i]M+B[i]N + MA[i]ᵀ + NᵀB[i]ᵀ ≼ 0
/// M ≽ 0
/// </pre>
/// @param A The dynamics of the i'th system is ẋ = A[i]x+B[i]u.
/// @param B The dynamics of the i'th system is ẋ = A[i]x+B[i]u.
/// @param[out] M The common Lyapunov function is xᵀM⁻¹x
/// @param[out] N The common controller is u = NM⁻¹x
std::unique_ptr<solvers::MathematicalProgram> ConstructCommonLyapunovProgram(
    const std::vector<Eigen::MatrixXd>& A,
    const std::vector<Eigen::MatrixXd>& B, MatrixX<symbolic::Variable>* M,
    MatrixX<symbolic::Variable>* N);

}  // namespace analysis
}  // namespace systems
}  // namespace drake
