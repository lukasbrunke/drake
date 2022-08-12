import numpy as np
import pickle

import clf_cbf_utils

import pydrake.systems.analysis as analysis
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
import pydrake.symbolic as sym
import pydrake.common

def Search(
    quadrotor: analysis.Quadrotor2dTrigPlant, x: np.ndarray, thrust_max: float,
    deriv_eps: float, unsafe_regions: list, x_safe: np.ndarray
) -> sym.Polynomial:
    f, G = analysis.TrigPolyDynamicsTwinQuadrotor(quadrotor, x)

    u_vertices = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0], 
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 1], 
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 0]]).T * thrust_max

    state_constraints = analysis.TwinQuadrotor2dStateEqConstraint(x)

    dynamics_denominator = None

    beta_minus = -0.0 

    beta_plus = 0.1

    dut = analysis.ControlBarrier(
        f, G, dynamics_denominator, x, beta_minus, beta_plus, unsafe_regions,
        u_vertices, state_constraints)

    h_degree = 2
    x_set = sym.Variables(x)

    #h_init = sym.Polynomial(x[5] * x[5] + x[6] * x[6] + x[7] * x[7] - 0.2)
    #with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor2d_cbf/twin_quadrotor2d_trig_cbf5.pickle", "rb") as input_file:
    with open("twin_quadrotor2d_trig_cbf7.pickle", "rb") as input_file:
        h_init = clf_cbf_utils.deserialize_polynomial(
            x_set, pickle.load(input_file)["h"])
    h_init_x_safe = h_init.EvaluateIndeterminates(x, x_safe)
    print(f"h_init(x_safe): {h_init_x_safe.squeeze()}")
    if np.any(h_init_x_safe < 0):
        h_init -= h_init_x_safe.min()
        h_init += 0.1

    with_slack_a = False

    lambda0_degree = 4
    lambda1_degree = 4
    l_degrees = [2] * 16
    hdot_eq_lagrangian_degrees = [h_degree + lambda0_degree - 2] * 2
    if with_slack_a:
        hdot_a_degree = h_degree + lambda0_degree

    t_degrees = [0]
    s_degrees = [[h_degree - 2]]
    unsafe_eq_lagrangian_degrees = [[h_degree - 2, h_degree - 2]]
    if with_slack_a:
        unsafe_a_degrees = [h_degree]
    h_x_safe_min = np.array([0.01])

    if with_slack_a:
        hdot_a_zero_tol = 3E-9
        unsafe_a_zero_tol = 1E-9
        search_options = analysis.ControlBarrier.SearchWithSlackAOptions(
            hdot_a_zero_tol, unsafe_a_zero_tol, use_zero_a=True,
            hdot_a_cost_weight=1., unsafe_a_cost_weight=[1.])
        search_options.bilinear_iterations = 100
        search_options.lagrangian_step_solver_options = mp.SolverOptions()
        search_options.lagrangian_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.barrier_step_solver_options = mp.SolverOptions()
        search_options.barrier_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.barrier_step_backoff_scale = 0.03
        search_options.lagrangian_step_backoff_scale = 0.04
        search_result = dut.SearchWithSlackA(
            h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
            hdot_eq_lagrangian_degrees, hdot_a_degree, t_degrees, s_degrees,
            unsafe_eq_lagrangian_degrees, unsafe_a_degrees, x_safe,
            h_x_safe_min, search_options)
        search_lagrangian_ret = dut.SearchLagrangian(
            search_result.h, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
            hdot_eq_lagrangian_degrees, None, t_degrees, s_degrees,
            unsafe_eq_lagrangian_degrees, [None] * len(unsafe_regions),
            search_options, backoff_scale=None)
    else:
        ellipsoids = [analysis.ControlBarrier.Ellipsoid(
            c=x_safe[:, 0], S=np.eye(12), d=0., r_degree=0,
            eq_lagrangian_degrees=[0, 0])]
        ellipsoid_options = [
            analysis.ControlBarrier.EllipsoidMaximizeOption(
                t=sym.Polynomial(), s_degree=0, backoff_scale=0.04)]
        search_options = analysis.ControlBarrier.SearchOptions()
        search_options.lagrangian_step_solver_options = mp.SolverOptions()
        search_options.lagrangian_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.barrier_step_solver_options = mp.SolverOptions()
        search_options.barrier_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.lsol_tiny_coeff_tol = 1E-6
        search_options.hsol_tiny_coeff_tol = 1E-6
        search_options.barrier_step_backoff_scale = 0.1
        x_anchor = x_safe[:, 0]
        h_x_anchor_max = h_init.EvaluateIndeterminates(x, x_anchor)[0] * 2
        search_result = dut.Search(
            h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree,
            l_degrees, hdot_eq_lagrangian_degrees, t_degrees, s_degrees,
            unsafe_eq_lagrangian_degrees, x_anchor, h_x_anchor_max,
            search_options, ellipsoids, ellipsoid_options)

    with open("twin_quadrotor2d_trig_cbf8.pickle", "wb") as handle:
        pickle.dump({
            "h": clf_cbf_utils.serialize_polynomial(search_result.h),
            "beta_plus": beta_plus, "beta_minus": beta_minus,
            "deriv_eps": deriv_eps, "thrust_max": thrust_max,
            "x_safe": x_safe}, handle)
    return search_result.h

def DoMain():
    pydrake.common.configure_logging()
    quadrotor = analysis.Quadrotor2dTrigPlant()
    x = np.empty(12, dtype=object)
    for i in range(12):
        x[i] = sym.Variable(f"x{i}")

    thrust_equilibrium = analysis.EquilibriumThrust(quadrotor)
    thrust_max = 3 * thrust_equilibrium
    deriv_eps = 0.5
    unsafe_regions = [np.array([
        sym.Polynomial(x[5] ** 2 + x[6] ** 2 - quadrotor.length() ** 2 )])]

    safe_states = [None] * 1
    safe_states[0] = [np.zeros(6), np.array([0, -1, 0, 0, 0, 0])]
    x_safe = np.empty((12, len(safe_states)))
    for i in range(len(safe_states)):
        x1 = analysis.ToQuadrotor2dTrigState(safe_states[i][0])
        x2 = analysis.ToQuadrotor2dTrigState(safe_states[i][1])
        x_safe[:, i] = np.concatenate((x1[2:], x2[:2] - x1[:2], x2[2:]))

    h_sol = Search(quadrotor, x, thrust_max, deriv_eps, unsafe_regions, x_safe)

if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        DoMain()
