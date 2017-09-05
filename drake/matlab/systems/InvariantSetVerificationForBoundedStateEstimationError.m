classdef InvariantSetVerificationForBoundedStateEstimationError
% Find the invariant set for a control-affine system
% xdot = f(x) + g(x) * y
% and the linear controller u = K * x_hat + d
% If we assume that the estimated state x_hat satisfies
% x_hat = x + w
% where w is the state estimation error, bounded in a polytope,
% then we can try to find a Lyapunov-like function V, satisfying
% Vdot <= 0 if V(x) = 1
% x in S if V(x) <= 1
% Then the set S is an outer approximation of the invariant set. 
% Our goal is to minimize the size of S.
  properties(SetAccess = protected)
    plant
    plant_taylor
    x0
    u0
    p_dynamics
    p_x
    p_u
  end
  
  methods
    function obj = InvariantSetVerificationForBoundedStateEstimationError(plant, x0, u0)
      typecheck(plant,'Manipulator');
      obj.plant = plant;
      obj.plant_taylor = taylorApprox(obj.plant, 0, x0, u0, 3);
      [obj.p_dynamics, ~] = obj.plant_taylor.getPolyDynamics(0);
      obj.p_x = obj.plant_taylor.getStateFrame().getPoly();
      obj.p_u = obj.plant_taylor.getInputFrame().getPoly();
    end
    
    function [S, l1, l2] = FindInitialFeasibleSolution(obj, rho0, V0, control_gain, control_offset, x_err_vertices)
      % Find an initial feasible solution to the problem
      % max trace(S)
      % s.t -Vdot - l1 * (rho - V0) is sos
      %      (1 - (x - x0)' * S * (x - x0)) - l2 * (rho - V) is sos
      %      l2 is sos
      prog = spotsosprog();
      prog.withIndeterminate(obj.p_x);
      [prog, S] = prog.newPSD(obj.plant.getNumContStates());
      num_x_err_vertices = size(obj.x_err_vertices, 2);
      l1 = zeros(num_x_err_vertices, 1);
      p_f_cl = obj.ClosedLoopDynamics(control_gain, control_offset, x_err_vertices);
      dVdx = diff(V, obj.p_x);
      for i = 1 : num_x_err_vertices
        Vdot = dVdx * p_f_cl(:, i);
        [prog, l1(i)] = prog.newSOSPoly(obj.p_x, 2);
        prog = prog.withSOS(-Vdot - l1(i) * (rho0 - V0));
      end
      if (deg(V0, obj.p_x) == 2) 
        [prog, l2] = prog.newPos(1, 1);
      else
        [prog, l2] = prog.newSOSPoly(obj.p_x, 2);
      end
      prog = prog.withSOS(1 - (obj.p_x - obj.x0)' * S * (obj.p_x - obj.x0) - l2 * (rho0 - V0));
      
      options = @spot_sdp_default_options;
      options.verbose = 1;
      sol = prog.minimize(-trace(S), @spot_mosek, options);
      if sol.isPrimalFeasible()
        S = double(sol.eval(S));
        l1 = sol.eval(l1);
        l2 = sol.eval(l2);
      else
        error('Infeasible rho and V');
      end
    end
    
    function p_f_cl = ClosedLoopDynamics(obj, control_gain, control_offset, x_err_vertices)
      sizecheck(control_gain, [obj.plant.getNumInputs(), obj.plant.getNumContStates()]);
      sizecheck(control_offset, [obj.plant.getNumInputs(), 1]);
      sizecheck(x_err_vertices, [obj.plant.getNumContStates(), NaN]);
      num_x_err_vertices = size(x_err_vertices, 2);
      p_f_cl = zeros(obj.plant.getNumContStates(), num_x_err_vertices);
      for i = 1 : num_x_err_vertices
        p_f_cl(:, i) = subs(obj.p_dynamics, obj.p_u, control_gain * (obj.p_x + x_err_vertices(:, i)) + control_offset);
      end
    end
  end
end