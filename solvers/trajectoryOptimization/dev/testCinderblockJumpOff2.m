function testCinderblockJumpOff2(mode)
% jump off one layer of cinderblock
% mode 0 play back
% mode 1 trajectory optimization
% Jumping with fixed contact sequence
warning('off','Drake:RigidBody:SimplifiedCollisionGeometry');
warning('off','Drake:RigidBody:NonPositiveInertiaMatrix');
warning('off','Drake:RigidBodyManipulator:UnsupportedContactPoints');
warning('off','Drake:RigidBodyManipulator:UnsupportedJointLimits');
warning('off','Drake:RigidBodyManipulator:UnsupportedVelocityLimits');
urdf = [getDrakePath,'/examples/Atlas/urdf/atlas_convex_hull.urdf'];
urdf_visual = [getDrakePath,'/examples/Atlas/urdf/atlas_minimal_contact.urdf'];
options.floating = true;
robot = RigidBodyManipulator(urdf,options);
robot_visual = RigidBodyManipulator(urdf_visual,options);
cinderblock_urdf = [getDrakePath,'/solvers/trajectoryOptimization/dev/cinderblock_single.urdf'];
cinderblock_dim = [0.397;0.195;0.145];
cinderblock_pos1 = [-0.05;cinderblock_dim(2)/2;cinderblock_dim(3)/2;0;0;0];
cinderblock_pos2 = [-0.05;-cinderblock_dim(2)/2;cinderblock_dim(3)/2;0;0;0];
cinderblock_lb = cinderblock_pos2(1:3)-cinderblock_dim/2.*[0.9;0.8;-1];
cinderblock_ub = cinderblock_pos1(1:3)+cinderblock_dim/2.*[0.9;0.8;1];
robot = robot.addRobotFromURDF(cinderblock_urdf,cinderblock_pos1(1:3),cinderblock_pos1(4:6));
robot = robot.addRobotFromURDF(cinderblock_urdf,cinderblock_pos2(1:3),cinderblock_pos2(4:6));
robot_visual = robot_visual.addRobotFromURDF(cinderblock_urdf,cinderblock_pos1(1:3),cinderblock_pos1(4:6));
robot_visual = robot_visual.addRobotFromURDF(cinderblock_urdf,cinderblock_pos2(1:3),cinderblock_pos2(4:6));
nomdata = load([getDrakePath,'/examples/Atlas/data/atlas_fp.mat']);
nq = robot.getNumPositions();
qstar = nomdata.xstar(1:nq);
kinsol_star = robot.doKinematics(qstar);
nv = robot.getNumVelocities();
vstar = zeros(nv,1);

bky_idx = robot.getBody(robot.findJointInd('back_bky')).position_num;
l_leg_hpx = robot.getBody(robot.findJointInd('l_leg_hpx')).position_num;
r_leg_hpx = robot.getBody(robot.findJointInd('r_leg_hpx')).position_num;
l_leg_kny = robot.getBody(robot.findJointInd('l_leg_kny')).position_num;
r_leg_kny = robot.getBody(robot.findJointInd('r_leg_kny')).position_num;
l_leg_aky = robot.getBody(robot.findJointInd('l_leg_aky')).position_num;
r_leg_aky = robot.getBody(robot.findJointInd('r_leg_aky')).position_num;

l_foot = robot.findLinkInd('l_foot');
r_foot = robot.findLinkInd('r_foot');
l_foot_shapes_heel = robot.getBody(l_foot).getContactShapes('heel');
l_foot_shapes_toe = robot.getBody(l_foot).getContactShapes('toe');
r_foot_shapes_heel = robot.getBody(r_foot).getContactShapes('heel');
r_foot_shapes_toe = robot.getBody(r_foot).getContactShapes('toe');
l_foot_toe = [];
l_foot_heel = [];
r_foot_toe = [];
r_foot_heel = [];
for i = 1:length(l_foot_shapes_heel)
  l_foot_heel = [l_foot_heel l_foot_shapes_heel{i}.getPoints];
end
for i = 1:length(l_foot_shapes_toe)
  l_foot_toe = [l_foot_toe l_foot_shapes_toe{i}.getPoints];
end
for i = 1:length(r_foot_shapes_heel)
  r_foot_heel = [r_foot_heel r_foot_shapes_heel{i}.getPoints];
end
for i = 1:length(r_foot_shapes_toe)
  r_foot_toe = [r_foot_toe r_foot_shapes_toe{i}.getPoints];
end
l_foot_bottom = [l_foot_toe l_foot_heel];
r_foot_bottom = [r_foot_toe r_foot_heel];

q_start = qstar;
q_start(3) = qstar(3)+cinderblock_dim(3);
lfoot_on_cinderblock = WorldPositionConstraint(robot,l_foot,l_foot_bottom,repmat([cinderblock_lb(1:2);cinderblock_ub(3)],1,4),repmat(cinderblock_ub,1,4));
rfoot_on_cinderblock = WorldPositionConstraint(robot,r_foot,r_foot_bottom,repmat([cinderblock_lb(1:2);cinderblock_ub(3)],1,4),repmat(cinderblock_ub,1,4));
qsc = QuasiStaticConstraint(robot);
qsc = qsc.addContact(l_foot,l_foot_bottom,r_foot,r_foot_bottom);
qsc = qsc.setActive(true);
qsc = qsc.setShrinkFactor(0.1);
pc = PostureConstraint(robot);
pc = pc.setJointLimits([1;2;5;l_leg_kny;r_leg_kny],[0;0;0;-inf(2,1)],[0;0;0;0.6*ones(2,1)]);
prog = InverseKinematics(robot,q_start,lfoot_on_cinderblock,rfoot_on_cinderblock,qsc,pc);
q_start = prog.solve(q_start);

kinsol_start = robot.doKinematics(q_start);
lfoot_pos_start = robot.forwardKin(kinsol_start,l_foot,[0;0;0],2);
rfoot_pos_start = robot.forwardKin(kinsol_start,r_foot,[0;0;0],2);
lfoot_toe_pos_start = robot.forwardKin(kinsol_start,l_foot,l_foot_toe,0);
rfoot_toe_pos_start = robot.forwardKin(kinsol_start,r_foot,r_foot_toe,0);
lfoot_heel_pos_start = robot.forwardKin(kinsol_start,l_foot,l_foot_heel,0);
rfoot_heel_pos_start = robot.forwardKin(kinsol_start,r_foot,r_foot_heel,0);
com_start = robot.getCOM(kinsol_start);


heel_takeoff_idx = 4;
toe_takeoff_idx = 7;
heel_land_idx = 13;
toe_land_idx = 11;
nT = 15;
mu = 1;
num_edges = 4;
FC_angles = linspace(0,2*pi,num_edges+1);FC_angles(end) = [];
g = 9.81;
FC_axis = [0;0;1];
FC_perp1 = rotx(pi/2)*FC_axis;
FC_perp2 = cross(FC_axis,FC_perp1);
FC_edge = bsxfun(@plus,FC_axis,mu*(bsxfun(@times,cos(FC_angles),FC_perp1) + ...
                                   bsxfun(@times,sin(FC_angles),FC_perp2)));
FC_edge = bsxfun(@rdivide,FC_edge,sqrt(sum(FC_edge.^2,1)));
FC_edge = FC_edge*robot.getMass*g;
l_foot_contact_wrench = struct('active_knot',[],'cw',[]);
l_foot_contact_wrench(1) = struct('active_knot',1:heel_takeoff_idx,'cw',LinearFrictionConeWrench(robot,l_foot,l_foot_heel,FC_edge));
l_foot_contact_wrench(2) = struct('active_knot',1:toe_takeoff_idx,'cw',LinearFrictionConeWrench(robot,l_foot,l_foot_toe,FC_edge));
l_foot_contact_wrench(3) = struct('active_knot',heel_land_idx:nT,'cw',LinearFrictionConeWrench(robot,l_foot,l_foot_heel,FC_edge));
l_foot_contact_wrench(4) = struct('active_knot',toe_land_idx:nT,'cw',LinearFrictionConeWrench(robot,l_foot,l_foot_toe,FC_edge));
r_foot_contact_wrench = struct('active_knot',[],'cw',[]);
r_foot_contact_wrench(1) = struct('active_knot',1:heel_takeoff_idx,'cw',LinearFrictionConeWrench(robot,r_foot,r_foot_heel,FC_edge));
r_foot_contact_wrench(2) = struct('active_knot',1:toe_takeoff_idx,'cw',LinearFrictionConeWrench(robot,r_foot,r_foot_toe,FC_edge));
r_foot_contact_wrench(3) = struct('active_knot',heel_land_idx:nT,'cw',LinearFrictionConeWrench(robot,r_foot,r_foot_heel,FC_edge));
r_foot_contact_wrench(4) = struct('active_knot',toe_land_idx:nT,'cw',LinearFrictionConeWrench(robot,r_foot,r_foot_toe,FC_edge));



tf_range = [0.5 2];
q_nom = bsxfun(@times,qstar,ones(1,nT));
Q_comddot = eye(3);
Q = eye(nq);
Q(1,1) = 0;
Q(2,2) = 0;
Q(3,3) = 0;
Q(bky_idx,bky_idx) = 100*Q(bky_idx,bky_idx);
Qv = 0.1*eye(nv);
Q_contact_force = 10/(robot.getMass*g)^2*eye(3);
cdfkp = ComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,q_nom,Q_contact_force,[l_foot_contact_wrench r_foot_contact_wrench]);

lfoot_toe_above_ground = WorldPositionConstraint(robot,l_foot,l_foot_toe,[nan(2,2);0.01*ones(1,2)],nan(3,2));
% lfoot_toe_above_ground = lfoot_toe_above_ground.generateConstraint([]);
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_toe_above_ground,num2cell(toe_takeoff_idx+1:toe_land_idx-1));

rfoot_toe_above_ground = WorldPositionConstraint(robot,r_foot,r_foot_toe,[nan(2,2);0.01*ones(1,2)],nan(3,2));
% rfoot_toe_above_ground = rfoot_toe_above_ground.generateConstraint([]);
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_toe_above_ground,num2cell(toe_takeoff_idx+1:toe_land_idx-1));

lfoot_heel_above_ground = WorldPositionConstraint(robot,l_foot,l_foot_heel,[nan(2,2);0.01*ones(1,2)],nan(3,2));
% lfoot_heel_above_ground = lfoot_heel_above_ground.generateConstraint([]);
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_heel_above_ground,num2cell(heel_takeoff_idx+1:heel_land_idx-1));

rfoot_heel_above_ground = WorldPositionConstraint(robot,r_foot,r_foot_heel,[nan(2,2);0.01*ones(1,2)],nan(3,2));
% rfoot_heel_above_ground = rfoot_heel_above_ground.generateConstraint([]);
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_heel_above_ground,num2cell(heel_takeoff_idx+1:heel_land_idx-1));

% lfoot_heel_on_ground = WorldPositionConstraint(robot,l_foot,l_foot_heel,lfoot_heel_pos_star,lfoot_heel_pos_star);
% % lfoot_heel_on_ground = lfoot_heel_on_ground.generateConstraint([]);
% sdfkp = sdfkp.addRigidBodyConstraint(lfoot_heel_on_ground,num2cell(heel_land_idx:toe_land_idx-1));
% 
% rfoot_heel_on_ground = WorldPositionConstraint(robot,r_foot,r_foot_heel,rfoot_heel_pos_star,rfoot_heel_pos_star);
% % rfoot_heel_on_ground = rfoot_heel_on_ground.generateConstraint([]);
% sdfkp = sdfkp.addRigidBodyConstraint(rfoot_heel_on_ground,num2cell(heel_land_idx:toe_land_idx-1));

land_region_lb = [cinderblock_dim(1)/2+cinderblock_pos1(1)+0.08;-inf;0];
land_region_ub = [inf;inf;0];
lfoot_toe_on_ground = WorldPositionConstraint(robot,l_foot,l_foot_toe,bsxfun(@times,land_region_lb,ones(1,size(l_foot_toe,2))),bsxfun(@times,land_region_ub,ones(1,size(l_foot_toe,2))));
% lfoot_toe_on_ground = lfoot_toe_on_ground.generateConstraint([]);
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_toe_on_ground,num2cell(toe_land_idx:nT));

rfoot_toe_on_ground = WorldPositionConstraint(robot,r_foot,r_foot_toe,bsxfun(@times,land_region_lb,ones(1,size(r_foot_toe,2))),bsxfun(@times,land_region_ub,ones(1,size(r_foot_toe,2))));
% rfoot_toe_on_ground = rfoot_toe_on_ground.generateConstraint([]);
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_toe_on_ground,num2cell(toe_land_idx:nT));

lfoot_heel_on_ground = WorldPositionConstraint(robot,l_foot,l_foot_heel,bsxfun(@times,land_region_lb,ones(1,size(l_foot_heel,2))),bsxfun(@times,land_region_ub,ones(1,size(l_foot_heel,2))));
% lfoot_on_ground = {lfoot_on_ground{1}.generateConstraint([]),lfoot_on_ground{2}.generateConstraint([])};
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_heel_on_ground,num2cell(heel_land_idx:nT));

rfoot_heel_on_ground = WorldPositionConstraint(robot,r_foot,r_foot_heel,bsxfun(@times,land_region_lb,ones(1,size(r_foot_heel,2))),bsxfun(@times,land_region_ub,ones(1,size(r_foot_heel,2))));
% rfoot_on_ground = {rfoot_on_ground{1}.generateConstraint([]),rfoot_on_ground{2}.generateConstraint([])};
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_heel_on_ground,num2cell(heel_land_idx:nT));

% fix the toe position after landing
rfoot_toe_static_on_ground = WorldFixedPositionConstraint(robot,r_foot,r_foot_toe);
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_toe_static_on_ground,{toe_land_idx:nT});
lfoot_toe_static_on_ground = WorldFixedPositionConstraint(robot,l_foot,l_foot_toe);
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_toe_static_on_ground,{toe_land_idx:nT});

lfoot_on_cinderblock = {WorldPositionConstraint(robot,l_foot,[0;0;0],lfoot_pos_start(1:3),lfoot_pos_start(1:3)),...
  WorldQuatConstraint(robot,l_foot,lfoot_pos_start(4:7),0)};
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_on_cinderblock{1},num2cell(2:heel_takeoff_idx));
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_on_cinderblock{2},num2cell(2:heel_takeoff_idx));

rfoot_on_cinderblock = {WorldPositionConstraint(robot,r_foot,[0;0;0],rfoot_pos_start(1:3),rfoot_pos_start(1:3)),...
  WorldQuatConstraint(robot,r_foot,rfoot_pos_start(4:7),0)};
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_on_cinderblock{1},num2cell(2:heel_takeoff_idx));
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_on_cinderblock{2},num2cell(2:heel_takeoff_idx));

rfoot_heel_above_cinderblock = WorldPositionConstraint(robot,r_foot,r_foot_heel,bsxfun(@times,[nan(2,1);cinderblock_dim(3)+0.005],ones(1,size(r_foot_heel,2))),nan(3,size(r_foot_heel,2)));
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_heel_above_cinderblock,num2cell(heel_takeoff_idx+1:toe_takeoff_idx));

lfoot_heel_above_cinderblock = WorldPositionConstraint(robot,l_foot,l_foot_heel,bsxfun(@times,[nan(2,1);cinderblock_dim(3)+0.005],ones(1,size(l_foot_heel,2))),nan(3,size(l_foot_heel,2)));
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_heel_above_cinderblock,num2cell(heel_takeoff_idx+1:toe_takeoff_idx));

rfoot_toe_on_cinderblock = WorldPositionConstraint(robot,r_foot,r_foot_toe,rfoot_toe_pos_start,rfoot_toe_pos_start);
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_toe_on_cinderblock,num2cell(heel_takeoff_idx+1:toe_takeoff_idx));

lfoot_toe_on_cinderblock = WorldPositionConstraint(robot,l_foot,l_foot_toe,lfoot_toe_pos_start,lfoot_toe_pos_start);
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_toe_on_cinderblock,num2cell(heel_takeoff_idx+1:toe_takeoff_idx));
% 
% sdfkp = sdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(qstar(3)+0.3,inf),sdfkp.q_inds(3,apex_knot));

cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(q_start,q_start),cdfkp.q_inds(:,1));
% cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(qstar,qstar),cdfkp.q_inds(:,end));
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(vstar,vstar),cdfkp.v_inds(:,1));
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(vstar,vstar),cdfkp.v_inds(:,end));
cdfkp = cdfkp.addBoundingBoxConstraint(ConstantConstraint(zeros(6,1)),reshape(cdfkp.comdot_inds(:,[1,nT]),[],1));

cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(0.02*ones(nT-1,1),0.1*ones(nT-1,1)),cdfkp.h_inds(:));
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(0.02*ones(toe_takeoff_idx-heel_takeoff_idx,1),0.05*ones(toe_takeoff_idx-heel_takeoff_idx,1)),cdfkp.h_inds(heel_takeoff_idx:toe_takeoff_idx-1)');
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(0.02*ones(toe_land_idx-toe_takeoff_idx,1),0.05*ones(toe_land_idx-toe_takeoff_idx,1)),cdfkp.h_inds(toe_takeoff_idx:toe_land_idx-1)');

% cdfkp = cdfkp.addCoMBounds(1:nT,bsxfun(@times,com_star-[0.5;0.5;0.5],ones(1,nT)),bsxfun(@times,com_star+[0.5;0.5;1],ones(1,nT)));

cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(...
  reshape(bsxfun(@times,[-9.81;-9.81;-9.81*1.1],ones(1,nT)),[],1),...
  reshape(bsxfun(@times,[9.81;9.81;9.81],ones(1,nT)),[],1)),cdfkp.comddot_inds(:));

% add a collision avoidance constraint between the foot and the cinderblock
lfoot_avoid_cinderblock = MinDistanceConstraint(robot,0.04,struct('body_idx',[l_foot,1]));
rfoot_avoid_cinderblock = MinDistanceConstraint(robot,0.04,struct('body_idx',[r_foot,1]));
lfoot_avoid_cinderblock_interp = generateInterpolatedMinDistanceConstraint(lfoot_avoid_cinderblock,[0.3 0.6]);
rfoot_avoid_cinderblock_interp = generateInterpolatedMinDistanceConstraint(rfoot_avoid_cinderblock,[0.3 0.6]);
lfoot_avoid_cinderblock = lfoot_avoid_cinderblock.generateConstraint();
lfoot_avoid_cinderblock = lfoot_avoid_cinderblock{1};
rfoot_avoid_cinderblock = rfoot_avoid_cinderblock.generateConstraint();
rfoot_avoid_cinderblock = rfoot_avoid_cinderblock{1};

for i = toe_takeoff_idx+2:toe_land_idx
  cdfkp = cdfkp.addNonlinearConstraint(lfoot_avoid_cinderblock,cdfkp.q_inds(:,i),cdfkp.kinsol_dataind(i));
  cdfkp = cdfkp.addNonlinearConstraint(rfoot_avoid_cinderblock,cdfkp.q_inds(:,i),cdfkp.kinsol_dataind(i));
end
for i = toe_takeoff_idx+2:toe_land_idx-1
  cdfkp = cdfkp.addNonlinearConstraint(lfoot_avoid_cinderblock_interp{1},{cdfkp.q_inds(:,i),cdfkp.q_inds(:,i+1)});
  cdfkp = cdfkp.addNonlinearConstraint(rfoot_avoid_cinderblock_interp{1},{cdfkp.q_inds(:,i),cdfkp.q_inds(:,i+1)});
end

x_seed = zeros(cdfkp.num_vars,1);
x_seed(cdfkp.h_inds) = 0.04;
x_seed(cdfkp.q_inds(:)) = reshape(bsxfun(@times,q_start,ones(1,nT)),[],1);
x_seed(cdfkp.com_inds(:)) = reshape(bsxfun(@times,com_start,ones(1,nT)),[],1);
x_seed(cdfkp.lambda_inds{1}(:)) = reshape(bsxfun(@times,1/num_edges*ones(num_edges,4,1),ones(1,1,nT)),[],1);
x_seed(cdfkp.lambda_inds{2}(:)) = reshape(bsxfun(@times,1/num_edges*ones(num_edges,4,1),ones(1,1,nT)),[],1);

% add a cost to maximize com apex_height
% cdfkp = cdfkp.addCost(FunctionHandleConstraint(-inf,inf,2,@comApexHeightCost),{cdfkp.com_inds(3,toe_takeoff_idx);cdfkp.comdot_inds(3,toe_takeoff_idx)});
% add a constraint on the com apex height
apex_height_cnstr = FunctionHandleConstraint(com_start(3)+0.1,inf,2,@comApexHeight);
apex_height_cnstr = apex_height_cnstr.setName({'com apex height'});
% cdfkp = cdfkp.addNonlinearConstraint(apex_height_cnstr,{cdfkp.com_inds(3,toe_takeoff_idx);cdfkp.comdot_inds(3,toe_takeoff_idx)});

% add a symmetric constraint
symmetry_cnstr = symmetryConstraint(robot,2:nT);
cdfkp = cdfkp.addLinearConstraint(symmetry_cnstr,reshape(cdfkp.q_inds(:,2:nT),[],1));
% no yawing on the back
bkz_idx = robot.getBody(robot.findJointInd('back_bkz')).position_num;
cdfkp = cdfkp.addBoundingBoxConstraint(ConstantConstraint(zeros(nT-1,1)),reshape(cdfkp.q_inds(bkz_idx,2:nT),[],1));
% sdfkp = sdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(zeros(nT-2,1),zeros(nT-2,1)),reshape(sdfkp.q_inds(6,2:nT-1),[],1));
% sdfkp = sdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(zeros(nT-2,1),zeros(nT-2,1)),reshape(sdfkp.q_inds(4,2:nT-1),[],1));

% no leg crossing by constraining the hpx
no_crossing_leg = MinDistanceConstraint(robot,0.08,struct('body_idx',[l_foot,r_foot]));
no_crossing_leg = no_crossing_leg.generateConstraint();
no_crossing_leg = no_crossing_leg{1};
for i = toe_takeoff_idx+1:toe_land_idx
%   cdfkp = cdfkp.addNonlinearConstraint(no_crossing_leg,cdfkp.q_inds(:,i),cdfkp.kinsol_dataind(i));
end
cdfkp = cdfkp.addRigidBodyConstraint(WorldPositionConstraint(robot,l_foot,l_foot_bottom,[nan;0;nan]*ones(1,size(l_foot_bottom,2)),nan(3,size(l_foot_bottom,2))),num2cell(2:nT));
cdfkp = cdfkp.addRigidBodyConstraint(WorldPositionConstraint(robot,r_foot,r_foot_bottom,nan(3,size(r_foot_bottom,2)),[nan;0;nan]*ones(1,size(r_foot_bottom,2))),num2cell(2:nT));
% cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-0.1*ones(nT-1,1),inf(nT-1,1)),reshape(cdfkp.q_inds(l_leg_hpx,2:nT),[],1));
% cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-inf(nT-1,1),0.1*ones(nT-1,1)),reshape(cdfkp.q_inds(r_leg_hpx,2:nT),[],1));

% no large pelvis pitch
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint((qstar(5)-0.3)*ones(nT-2,1),(qstar(5)+0.4)*ones(nT-2,1)),cdfkp.q_inds(5,2:nT-1));
% no pelvis roll
cdfkp = cdfkp.addBoundingBoxConstraint(ConstantConstraint(zeros(nT-1,1)),cdfkp.q_inds(4,2:end)');

% do not bend the ankle too much at the end
% cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint([-0.5;-0.5],inf(2,1)),cdfkp.q_inds([l_leg_aky;r_leg_aky],end));

% add a constraint on the final pelvis height
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(0.7,inf),cdfkp.q_inds(3,nT));

% foot above the cinderblock right after taking off
lfoot_above_cinderblock = WorldPositionConstraint(robot,l_foot,l_foot_bottom,[nan;nan;cinderblock_dim(3)+0.01]*ones(1,size(l_foot_bottom,2)),nan(3,size(l_foot_bottom,2)));
rfoot_above_cinderblock = WorldPositionConstraint(robot,r_foot,r_foot_bottom,[nan;nan;cinderblock_dim(3)+0.01]*ones(1,size(r_foot_bottom,2)),nan(3,size(r_foot_bottom,2)));
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_above_cinderblock,{toe_takeoff_idx+1});
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_above_cinderblock,{toe_takeoff_idx+1});

% foot cannot stretch too much outward in the y direction
lfoot_no_y_stretch = WorldPositionConstraint(robot,l_foot,l_foot_bottom,nan(3,size(l_foot_bottom,2)),[nan;0.3;nan]*ones(1,size(l_foot_bottom,2)));
rfoot_no_y_stretch = WorldPositionConstraint(robot,r_foot,r_foot_bottom,[nan;-0.3;nan]*ones(1,size(l_foot_bottom,2)),nan(3,size(l_foot_bottom,2)));
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_no_y_stretch,num2cell(toe_takeoff_idx:toe_takeoff_idx+3));
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_no_y_stretch,num2cell(toe_takeoff_idx:toe_takeoff_idx+3));

cdfkp = cdfkp.setSolverOptions('snopt','iterationslimit',1e6);
cdfkp = cdfkp.setSolverOptions('snopt','majoriterationslimit',1000);
cdfkp = cdfkp.setSolverOptions('snopt','majorfeasibilitytolerance',1e-5);
cdfkp = cdfkp.setSolverOptions('snopt','majoroptimalitytolerance',2e-4);
cdfkp = cdfkp.setSolverOptions('snopt','superbasicslimit',2000);
cdfkp = cdfkp.setSolverOptions('snopt','print','test_jump_cinderblock.out');

seed_sol = load('test_cinderblock4','-mat','x_sol');
if(mode == 0)
  jump = load('test_cinderblock2','-mat','t_sol','v_sol','q_sol','wrench_sol','com_sol','comdot_sol','comddot_sol');
  v = robot.constructVisualizer();
  xtraj = PPTrajectory(foh(jump.t_sol,[jump.q_sol;jump.v_sol]));
  xtraj = xtraj.setOutputFrame(robot.getStateFrame);
  v.playback(xtraj,struct('slider',true));
  t_samples = linspace(jump.t_sol(1),jump.t_sol(end),1000);
  com_traj = PPTrajectory(pchipDeriv(jump.t_sol,jump.com_sol,jump.comdot_sol));
  com_samples = com_traj.eval(t_samples);
  figure;
  plot(jump.t_sol,jump.com_sol(3,:),'*','Markersize',10);
  hold on;
  plot(t_samples,com_samples(3,:),'LineWidth',2);
  title('CoM height','fontsize',22);
  xlabel('time (seconds)','fontsize',22);
  ylabel('CoM height (meters)','fontsize',22);
  plot(jump.t_sol(toe_takeoff_idx),jump.com_sol(3,toe_takeoff_idx),'*','Markersize',10,'Color','r');
  text(jump.t_sol(toe_takeoff_idx)-0.01,jump.com_sol(3,toe_takeoff_idx)-0.03,'toe take off','fontsize',22);
  plot(jump.t_sol(heel_takeoff_idx),jump.com_sol(3,heel_takeoff_idx),'*','Markersize',10,'Color','r');
  text(jump.t_sol(heel_takeoff_idx)-0.02,jump.com_sol(3,heel_takeoff_idx)-0.02,'heel take off','fontsize',22);
  plot(jump.t_sol(toe_land_idx),jump.com_sol(3,toe_land_idx),'*','Markersize',10,'Color','r');
  text(jump.t_sol(toe_land_idx)+0.02,jump.com_sol(3,toe_land_idx),'toe touch down','fontsize',22);
  plot(jump.t_sol(heel_land_idx),jump.com_sol(3,heel_land_idx),'*','Markersize',10,'Color','r');
  text(jump.t_sol(heel_land_idx)-0.07,jump.com_sol(3,heel_land_idx)-0.02,'heel touch down','fontsize',22);
  set(gca,'Fontsize',18);
end
if(mode == 1)
tic
[x_sol,F,info] = cdfkp.solve(seed_sol.x_sol);
toc
[q_sol,v_sol,h_sol,t_sol,com_sol,comdot_sol,comddot_sol,H_sol,Hdot_sol,lambda_sol,wrench_sol] = parseSolution(cdfkp,x_sol);
xtraj_sol = PPTrajectory(foh(cumsum([0 h_sol]),[q_sol;v_sol]));
xtraj_sol = xtraj_sol.setOutputFrame(robot.getStateFrame);
end
keyboard;
end

function [h,dh] = comApexHeight(takeoff_height,takeoff_vel)
g = 9.81;
h = takeoff_height+takeoff_vel^2/(2*g);
dh = [1 takeoff_vel/g];
end

function [h,dh] = comApexHeightCost(takeoff_height,takeoff_vel)
scaler = -200;
[h1,dh1] = comApexHeight(takeoff_height,takeoff_vel);
h = scaler*h1;
dh = scaler*dh1;
end

function symmetry_cnstr = symmetryConstraint(robot,t_idx)
N = numel(t_idx);
num_symmetry = 10;
symmetric_matrix = zeros(num_symmetry,robot.getNumPositions);
l_arm_usy_idx = robot.getBody(robot.findJointInd('l_arm_usy')).position_num;
r_arm_usy_idx = robot.getBody(robot.findJointInd('r_arm_usy')).position_num;
symmetric_matrix(1,[l_arm_usy_idx r_arm_usy_idx]) = [1 -1];
l_arm_shx_idx = robot.getBody(robot.findJointInd('l_arm_shx')).position_num;
r_arm_shx_idx = robot.getBody(robot.findJointInd('r_arm_shx')).position_num;
symmetric_matrix(2,[l_arm_shx_idx r_arm_shx_idx]) = [1 1];
l_arm_ely_idx = robot.getBody(robot.findJointInd('l_arm_ely')).position_num;
r_arm_ely_idx = robot.getBody(robot.findJointInd('r_arm_ely')).position_num;
symmetric_matrix(3,[l_arm_ely_idx r_arm_ely_idx]) = [1 -1];
l_arm_elx_idx = robot.getBody(robot.findJointInd('l_arm_elx')).position_num;
r_arm_elx_idx = robot.getBody(robot.findJointInd('r_arm_elx')).position_num;
symmetric_matrix(4,[l_arm_elx_idx r_arm_elx_idx]) = [1 1];
l_arm_uwy_idx = robot.getBody(robot.findJointInd('l_arm_uwy')).position_num;
r_arm_uwy_idx = robot.getBody(robot.findJointInd('r_arm_uwy')).position_num;
symmetric_matrix(5,[l_arm_uwy_idx r_arm_uwy_idx]) = [1 -1];
l_leg_hpz_idx = robot.getBody(robot.findJointInd('l_leg_hpz')).position_num;
r_leg_hpz_idx = robot.getBody(robot.findJointInd('r_leg_hpz')).position_num;
symmetric_matrix(6,[l_leg_hpz_idx r_leg_hpz_idx]) = [1 1];
l_leg_hpx_idx = robot.getBody(robot.findJointInd('l_leg_hpx')).position_num;
r_leg_hpx_idx = robot.getBody(robot.findJointInd('r_leg_hpx')).position_num;
symmetric_matrix(7,[l_leg_hpx_idx r_leg_hpx_idx]) = [1 1];
l_leg_hpy_idx = robot.getBody(robot.findJointInd('l_leg_hpy')).position_num;
r_leg_hpy_idx = robot.getBody(robot.findJointInd('r_leg_hpy')).position_num;
symmetric_matrix(8,[l_leg_hpy_idx r_leg_hpy_idx]) = [1 -1];
l_leg_aky_idx = robot.getBody(robot.findJointInd('l_leg_aky')).position_num;
r_leg_aky_idx = robot.getBody(robot.findJointInd('r_leg_aky')).position_num;
symmetric_matrix(9,[l_leg_aky_idx r_leg_aky_idx]) = [1 -1];
l_leg_kny_idx = robot.getBody(robot.findJointInd('l_leg_kny')).position_num;
r_leg_kny_idx = robot.getBody(robot.findJointInd('r_leg_kny')).position_num;
symmetric_matrix(10,[l_leg_kny_idx r_leg_kny_idx]) = [1 -1];

symmetry_cnstr = LinearConstraint(zeros(num_symmetry*N,1),zeros(num_symmetry*N,1),kron(speye(N),symmetric_matrix));
cnstr_name = cell(num_symmetry*N,1);
for i = 1:N
  cnstr_name{(i-1)*num_symmetry+1} = sprintf('arm_usy_symmetry[%d]',t_idx(i));
  cnstr_name{(i-1)*num_symmetry+2} = sprintf('arm_shx_symmetry[%d]',t_idx(i));
  cnstr_name{(i-1)*num_symmetry+3} = sprintf('arm_ely_symmetry[%d]',t_idx(i));
  cnstr_name{(i-1)*num_symmetry+4} = sprintf('arm_elx_symmetry[%d]',t_idx(i));
  cnstr_name{(i-1)*num_symmetry+5} = sprintf('arm_uwy_symmetry[%d]',t_idx(i));
  cnstr_name{(i-1)*num_symmetry+6} = sprintf('leg_hpz_symmetry[%d]',t_idx(i));
  cnstr_name{(i-1)*num_symmetry+7} = sprintf('leg_hpx_symmetry[%d]',t_idx(i));
  cnstr_name{(i-1)*num_symmetry+8} = sprintf('leg_hpy_symmetry[%d]',t_idx(i));
  cnstr_name{(i-1)*num_symmetry+9} = sprintf('leg_aky_symmetry[%d]',t_idx(i));
  cnstr_name{(i-1)*num_symmetry+10} = sprintf('leg_kyy_symmetry[%d]',t_idx(i));
end
symmetry_cnstr = symmetry_cnstr.setName(cnstr_name);
end
