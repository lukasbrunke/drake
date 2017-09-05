function runInvariantSetVerificationForBoundedStateError
p = AcrobotPlant;

p = setInputLimits(p,-inf,inf);

[c, V] = balanceLQR(p);

p = InvariantSetVerificationForBoundedStateEstimationError(p, p.xG, p.uG);
x_err_vertices = 0.1 * [0, 0, 0, 0;
                        0, 0, 0, 0;
                        -1, 1, 0, 0;
                        0, 0, -1, 1];

[S, l1, l2] = p.FindInitialFeasibleSolution(0.1, V, c.D, p.uG.tau, x_err_vertices);
end