function InvariantSetVerificationForBoundedStateError(plant, control_gain, control_offset, x_err_vertices, x0, u0)
% For a control affine system
% xdot = f(x) + g(x) * u
% and the controlller u = K * x_hat + d
% If we assume that the estimated state x_hat satisfies
% x_hat = x + w
% where w is the state estimation error, bounded in a polytope, then
% we can try to find a Lyapunov function V, satisfying
% Vdot <= 0 if V(x) = 1
% x in S if V(x) <= 1
% Then the set S is an outer approximation of the invariant set for the system.
% Our goal is to minimize the size of S.
typecheck(plant, 'Manipulator');
sizecheck(x_err_vertices, [plant.getNumContStates(), NaN]);
num_x_err_vert = size(x_err_vertices, 2);
sys = taylorApprox(plant, 0, x0, u0, 3);
[p_dynamics, ~] = sys.getPolyDynamics(0);
p_x = sys.getStateFrame.getPoly;
p_u = sys.getInputFrame.getPoly;

prog = spotsosprog();
prog.withIndeterminate(p_x);
L1 = cell(num_x_err_vert, 1);
[prog, V] = prog.NewSOSPoly(p_x, 2);
for i = 1 : num_x_err_vert
  f_cl = subs(p_dynamics, p_u, control_gain * (p_x + x_err_vertices(:, i)) + control_offset);
  Vdot = diff(V, p_x) * f_cl;
  [prog, L1{i}] = prog.NewSOSPoly(p_x, 2);
  prog = prog.withSOS(-Vdot - L1{i} * (1 - V));
end
end

function InvariantSetVerificationV0(p_dynamics, p_x, p_u, control_gain, control_offset, x0, u0)
% Find an initial guess for the function V.
% This V satisfies that for a system WITHOUT state uncertainty.
% Vdot <= 0 if V(x) = 1
A = double(subs(diff(p_dynamics, p_x), p_x, x0));
B = 
prog = spotsosprog();
prog.withIndeterminate(p_x);
[prog, V] = prog.NewSOSPoly(p_x, 2);
f_cl = subs(p_dynamics, p_u, control_gain * p_x + control_offset);
Vdot = diff(V, p_x) * f_cl;

end