function testTrajOptConstraint
r = RigidBodyManipulator([getDrakePath,'/examples/Atlas/urdf/atlas_minimal_contact.urdf'],struct('floating',true));
nq = r.getNumDOF();
com_cnstr = CoMMatchConstraint(r);
testNonlinearConstraintWKinsol(com_cnstr,1e-3,randn(nq,1),randn(3,1));
end

function testNonlinearConstraintWKinsol(cnstr,tol,q,not_q)
[c,dc] = evalNonlinearConstraintWKinsol(cnstr,q,not_q);
nq = cnstr.robot.getNumDOF;
[~,dc_numeric] = geval(@(x) evalNonlinearConstraintWKinsol(cnstr,x(1:nq),x(nq+1:end)),[q;not_q],struct('grad_method','numerical'));
valuecheck(dc,dc_numeric,tol);
valuecheck(dc,sparse(cnstr.iCfun,cnstr.jCvar,dc(sub2ind([cnstr.num_cnstr,cnstr.xdim],cnstr.iCfun,cnstr.jCvar)),cnstr.num_cnstr,cnstr.xdim,cnstr.nnz));
end

function [c,dc] = evalNonlinearConstraintWKinsol(cnstr,q,not_q)
kinsol = cnstr.robot.doKinematics(q,false,false);
[c,dc] = cnstr.eval(kinsol,not_q);
end