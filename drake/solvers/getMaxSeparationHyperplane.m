function [c,d] = getMaxSeparationHyperplane(x1,x2)
% solve the following problem to find the hyperplane that maximally
% separates the set x1 and x2
checkDependency('spotless');
p = spotsosprog();
n1 = size(x1,2);
n2 = size(x2,2);
[p,c] = p.newFree(3);
[p,d] = p.newFree(1);
[p,s] = p.newFree(1);
p = p.withLor([1;c]);
p = p.withPos(c'*x1+(d-s)*ones(1,n1));
p = p.withPos(-c'*x2-(s+d)*ones(1,n2));
options = spot_sdp_default_options();
options.verbose = 0;
solver_sol = p.minimize(-s,@spot_mosek,options);
c = double(solver_sol.eval(c));
d = double(solver_sol.eval(d));
end