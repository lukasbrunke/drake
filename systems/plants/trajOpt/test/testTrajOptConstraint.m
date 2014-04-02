function testTrajOptConstraint
r = RigidBodyManipulator([getDrakePath,'/examples/Atlas/urdf/atlas_minimal_contact.urdf'],struct('floating',true));
nq = r.getNumDOF();
com_cnstr = CoMMatchConstraint(r);
testNonlinearConstraintWKinsol(r,com_cnstr,1e-3,randn(nq,1),randn(3,1));

l_foot = r.findLinkInd('l_foot');
l_foot_contact_pt = getContactPoints(getBody(r,l_foot));
r_foot = r.findLinkInd('r_foot');
r_foot_contact_pt = getContactPoints(getBody(r,r_foot));
l_foot_fc = FrictionConeWrenchConstraint(r,l_foot,l_foot_contact_pt,1,[0;0;1]);
fc_edge = [cos(linspace(0,2*pi,5));sin(linspace(0,2*pi,5));ones(1,5)];
r_foot_fc = LinearFrictionConeWrenchConstraint(r,r_foot,r_foot_contact_pt,fc_edge);
num_F = prod(l_foot_fc.F_size)+prod(r_foot_fc.F_size);
t = [];
sctc = SimpleCentroidalTorqueConstraint(t,nq,num_F,l_foot_fc,r_foot_fc);
testNonlinearConstraintWKinsol(r,sctc,1e-3,randn(nq,1),randn(num_F,1));
end

function testNonlinearConstraintWKinsol(robot,cnstr,tol,q,not_q)
[c,dc] = evalNonlinearConstraintWKinsol(robot,cnstr,q,not_q);
nq = robot.getNumDOF;
[~,dc_numeric] = geval(@(x) evalNonlinearConstraintWKinsol(robot,cnstr,x(1:nq),x(nq+1:end)),[q;not_q],struct('grad_method','numerical'));
valuecheck(dc,dc_numeric,tol);
valuecheck(dc,sparse(cnstr.iCfun,cnstr.jCvar,dc(sub2ind([cnstr.num_cnstr,cnstr.xdim],cnstr.iCfun,cnstr.jCvar)),cnstr.num_cnstr,cnstr.xdim,cnstr.nnz));
end

function [c,dc] = evalNonlinearConstraintWKinsol(robot,cnstr,q,not_q)
kinsol = robot.doKinematics(q,false,false);
[c,dc] = cnstr.eval(kinsol,not_q);
end