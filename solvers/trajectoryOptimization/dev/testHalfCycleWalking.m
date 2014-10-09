function testHalfCycleWalking
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
nomdata = load([getDrakePath,'/examples/Atlas/data/atlas_fp.mat']);
nq = robot.getNumPositions();
qstar = nomdata.xstar(1:nq);
kinsol_star = robot.doKinematics(qstar);
nv = robot.getNumVelocities();
vstar = zeros(nv,1);

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

l_leg_kny = robot.getBody(robot.findJointInd('l_leg_kny')).position_num;
r_leg_kny = robot.getBody(robot.findJointInd('r_leg_kny')).position_num;
l_arm_elx = robot.getBody(robot.findJointInd('l_arm_elx')).position_num;
r_arm_elx = robot.getBody(robot.findJointInd('r_arm_elx')).position_num;
l_arm_mwx = robot.getBody(robot.findJointInd('l_arm_mwx')).position_num;
r_arm_mwx = robot.getBody(robot.findJointInd('r_arm_mwx')).position_num;

pc = PostureConstraint(robot);
pc = pc.setJointLimits([1;2;l_leg_kny;r_leg_kny;l_arm_elx;r_arm_elx],...
    [0;0;-inf(2,1);-inf;-0.5],[0;0;0.2;0.2;0.5;inf]);

qsc = QuasiStaticConstraint(robot);
qsc = qsc.addContact(l_foot,l_foot_bottom,r_foot,r_foot_bottom);
qsc = qsc.setShrinkFactor(0.3);
qsc = qsc.setActive(true);

[qstar,info] = inverseKin(robot,qstar,qstar,...
    WorldPositionConstraint(robot,l_foot,l_foot_bottom,[nan;nan;0]*ones(1,size(l_foot_bottom,2)),[nan;nan;0]*ones(1,size(l_foot_bottom,2))),...
    WorldPositionConstraint(robot,r_foot,r_foot_bottom,[nan;nan;0]*ones(1,size(r_foot_bottom,2)),[nan;nan;0]*ones(1,size(r_foot_bottom,2))),...
    pc,qsc);
kinsol_star = robot.doKinematics(qstar,false,false);
rfoot_pos_star = robot.forwardKin(kinsol_star,r_foot,[0;0;0],2);
com_star = robot.getCOM(kinsol_star);
back_bky = robot.getBody(robot.findJointInd('back_bky')).position_num;

% start with swing leg heel already off the ground
swing_toe_off_idx = 3;
swing_heel_touch_idx = 7;
swing_toe_touch_idx = 9;
stance_heel_off_idx = 8;
nT = 10;

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

r_foot_contact_wrench = struct('active_knot',[],'cw',[]);
r_foot_contact_wrench(1) = struct('active_knot',1:stance_heel_off_idx,'cw',LinearFrictionConeWrench(robot,r_foot,r_foot_bottom,FC_edge));
r_foot_contact_wrench(2) = struct('active_knot',stance_heel_off_idx+1:nT,'cw',LinearFrictionConeWrench(robot,r_foot,r_foot_toe,FC_edge));
l_foot_contact_wrench = struct('active_knot',[],'cw',[]);
l_foot_contact_wrench(1) = struct('active_knot',1:swing_toe_off_idx,'cw',LinearFrictionConeWrench(robot,l_foot,l_foot_toe,FC_edge));
l_foot_contact_wrench(2) = struct('active_knot',swing_heel_touch_idx:swing_toe_touch_idx-1,'cw',LinearFrictionConeWrench(robot,l_foot,l_foot_heel,FC_edge));
l_foot_contact_wrench(3) = struct('active_knot',swing_toe_touch_idx:nT,'cw',LinearFrictionConeWrench(robot,l_foot,l_foot_bottom,FC_edge));

tf_range = [0.3 1];
q_nom = bsxfun(@times,qstar,ones(1,nT));
Q_comddot = eye(3);
Q = eye(nq);
Q(1,1) = 0;
Q(2,2) = 0;
Q(6,6) = 0;
Qv = 0.005*eye(nv);
% Qv(4,4) = 100*Qv(4,4);
Q_contact_force = 0/(robot.getMass*g)^2*eye(3);
cdfkp = ComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,q_nom,Q_contact_force,[l_foot_contact_wrench r_foot_contact_wrench]);

lfoot_heel_above_ground = WorldPositionConstraint(robot,l_foot,l_foot_heel,[nan;0.03;0.001]*ones(1,size(l_foot_heel,2)),nan(3,size(l_foot_heel,2)));
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_heel_above_ground,num2cell(1:swing_heel_touch_idx-1));

lfoot_toe_above_ground = WorldPositionConstraint(robot,l_foot,l_foot_toe,[nan;0.03;0.001]*ones(1,size(l_foot_toe,2)),nan(3,size(l_foot_toe,2)));
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_toe_above_ground,num2cell(swing_toe_off_idx+1:swing_toe_touch_idx-1));

lfoot_toe_on_ground = WorldPositionConstraint(robot,l_foot,l_foot_toe,[nan;nan;0]*ones(1,size(l_foot_toe,2)),[nan;nan;0]*ones(1,size(l_foot_toe,2)));
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_toe_on_ground,num2cell(1:swing_toe_off_idx));

lfoot_heel_on_ground = WorldPositionConstraint(robot,l_foot,l_foot_heel,[nan;nan;0]*ones(1,size(l_foot_heel,2)),[nan;nan;0]*ones(1,size(l_foot_heel,2)));
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_heel_on_ground,num2cell(swing_heel_touch_idx:swing_toe_touch_idx-1));

lfoot_on_ground = WorldPositionConstraint(robot,l_foot,l_foot_bottom(:,1:3),[nan;nan;0]*ones(1,3),[nan;nan;0]*ones(1,3));
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_on_ground,num2cell(swing_toe_touch_idx:nT));

lfoot_toe_fix = WorldFixedPositionConstraint(robot,l_foot,l_foot_toe);
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_toe_fix,{1:swing_toe_off_idx});
lfoot_heel_fix = WorldFixedPositionConstraint(robot,l_foot,l_foot_heel);
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_heel_fix,{swing_heel_touch_idx:swing_toe_touch_idx});
lfoot_fix = WorldFixedBodyPoseConstraint(robot,l_foot);
cdfkp = cdfkp.addRigidBodyConstraint(lfoot_fix,{swing_toe_touch_idx:nT});

rfoot_on_ground = WorldPositionConstraint(robot,r_foot,r_foot_bottom,[nan;-0.22;0]*ones(1,size(r_foot_bottom,2)),[nan;nan;0]*ones(1,size(r_foot_bottom,2)));
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_on_ground,num2cell(1:stance_heel_off_idx));

rfoot_heel_above_ground = WorldPositionConstraint(robot,r_foot,r_foot_heel,[nan;nan;0.001]*ones(1,size(r_foot_heel,2)),nan(3,size(r_foot_heel,2)));
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_heel_above_ground,num2cell(stance_heel_off_idx+1:nT));

rfoot_toe_on_ground = WorldPositionConstraint(robot,l_foot,l_foot_toe,[nan;nan;0]*ones(1,size(l_foot_toe,2)),[nan;nan;0]*ones(1,size(l_foot_toe,2)));
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_toe_on_ground,num2cell(stance_heel_off_idx+1:nT));

rfoot_fix = WorldFixedBodyPoseConstraint(robot,r_foot);
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_fix,{1:stance_heel_off_idx});
rfoot_toe_fix = WorldFixedPositionConstraint(robot,r_foot,r_foot_toe);
cdfkp = cdfkp.addRigidBodyConstraint(rfoot_toe_fix,{stance_heel_off_idx:nT});

cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(0.02*ones(nT-1,1),0.07*ones(nT-1,1)),cdfkp.h_inds(:));

% pelvis starts at origin
cdfkp = cdfkp.addBoundingBoxConstraint(ConstantConstraint([0;0]),cdfkp.q_inds(1:2,1));
% pelvis ends at certain distance away from the origin, along x-axis
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(0.35,0.6),cdfkp.q_inds(1,nT));

% half cycle periodicity constraint
cdfkp = addHalfCycleConstraint(robot,cdfkp);

% no large pelvis pitch
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-0.2*ones(nT,1),0.2*ones(nT,1)),cdfkp.q_inds(5,:)');

% add a bounds on the centroidal angular momentum
% cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(reshape(bsxfun(@times,-[0.2;0.1;0.2],ones(1,nT)),[],1),...
%     reshape(bsxfun(@times,[0.2;0.1;0.2],ones(1,nT)),[],1)),cdfkp.H_inds(:));

% back_bky should not bend too much
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-0.1*ones(nT,1),0.1*ones(nT,1)),cdfkp.q_inds(back_bky,:)');
% back_bky velocity is bounded
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-1*ones(nT,1),1*ones(nT,1)),cdfkp.v_inds(back_bky,:)');

% no large pelvis roll vibration 
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-2*ones(nT,1),2*ones(nT,1)),cdfkp.v_inds(4,:)');

% pelvis above certain height
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(0.85*ones(nT,1),inf(nT,1)),cdfkp.q_inds(3,:)');

% add a cost on centroidal angular momentum
cdfkp = cdfkp.addCost(QuadraticSumConstraint(-inf,inf,1000*eye(3),zeros(3,nT)),cdfkp.H_inds(:));

% no self collision
no_self_collision = MinDistanceConstraint(robot,0.01);
cdfkp = cdfkp.addRigidBodyConstraint(no_self_collision,num2cell(1:nT));

% Do not straighten the knees
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(0.1*ones(2*nT,1),inf(2*nT,1)),reshape(cdfkp.q_inds([l_leg_kny;r_leg_kny],:),[],1));

% wrist not bending too much
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-0.3*ones(nT,1),0.3*ones(nT,1)),cdfkp.q_inds(l_arm_mwx,:)');
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-0.3*ones(nT,1),0.3*ones(nT,1)),cdfkp.q_inds(r_arm_mwx,:)');

% do not bend the elbow too much
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-inf(nT,1),0.7*ones(nT,1)),cdfkp.q_inds(l_arm_elx,:)');
cdfkp = cdfkp.addBoundingBoxConstraint(BoundingBoxConstraint(-0.7*ones(nT,1),inf(nT,1)),cdfkp.q_inds(r_arm_elx,:)');

x_seed = zeros(cdfkp.num_vars,1);
x_seed(cdfkp.h_inds) = 0.05;
x_seed(cdfkp.q_inds(:)) = reshape(bsxfun(@times,qstar,ones(1,nT)),[],1);
x_seed(cdfkp.com_inds(:)) = reshape(bsxfun(@times,com_star,ones(1,nT)),[],1);
x_seed(cdfkp.lambda_inds{1}(:)) = reshape(bsxfun(@times,1/num_edges*ones(num_edges,4,1),ones(1,1,nT)),[],1);
x_seed(cdfkp.lambda_inds{2}(:)) = reshape(bsxfun(@times,1/num_edges*ones(num_edges,4,1),ones(1,1,nT)),[],1);

cdfkp = cdfkp.setSolverOptions('snopt','iterationslimit',1e6);
cdfkp = cdfkp.setSolverOptions('snopt','majoriterationslimit',1000);
cdfkp = cdfkp.setSolverOptions('snopt','majorfeasibilitytolerance',1e-6);
cdfkp = cdfkp.setSolverOptions('snopt','majoroptimalitytolerance',1e-5);
cdfkp = cdfkp.setSolverOptions('snopt','superbasicslimit',2000);
cdfkp = cdfkp.setSolverOptions('snopt','print','test_walk.out');

seed_sol = load('test_halfcycle_walk2','-mat','x_sol');
tic
[x_sol,F,info] = cdfkp.solve(seed_sol.x_sol);
toc
[q_sol,v_sol,h_sol,t_sol,com_sol,comdot_sol,comddot_sol,H_sol,Hdot_sol,lambda_sol,wrench_sol] = parseSolution(cdfkp,x_sol);
xtraj_sol = PPTrajectory(foh(cumsum([0 h_sol]),[q_sol;v_sol]));
xtraj_sol = xtraj_sol.setOutputFrame(robot.getStateFrame);
keyboard;
end

function cdfkp = addHalfCycleConstraint(robot,cdfkp)
nq = robot.getNumPositions;
nv = robot.getNumVelocities;
num_cnstr = nq-1;

A_position = zeros(num_cnstr,2*nq);
A_velocity = zeros(num_cnstr+1,2*nv);
position_cnstr_name = cell(num_cnstr,1);
velocity_cnstr_name = cell(num_cnstr+1,1);

cnstr_count = 1;
A_position(cnstr_count,2) = 1;
A_position(cnstr_count,nq+2) = 1;
A_velocity(cnstr_count,2) = 1;
A_velocity(cnstr_count,nv+2) = 1;
position_cnstr_name{cnstr_count} = 'base_y(1)+base_y(end)=0';
velocity_cnstr_name{cnstr_count} = 'base_ydot(1)+base_ydot(end)=0';
cnstr_count = cnstr_count+1;

A_position(cnstr_count,3) = 1;
A_position(cnstr_count,nq+3) = -1;
A_velocity(cnstr_count,3) = 1;
A_velocity(cnstr_count,nv+3) = -1;
position_cnstr_name{cnstr_count} = 'base_z(1)=baze_z(end)';
velocity_cnstr_name{cnstr_count} = 'baze_zdot(1)=baze_zdot(end)';
cnstr_count = cnstr_count+1;

A_position(cnstr_count,4) = 1;
A_position(cnstr_count,nq+4) = 1;
A_velocity(cnstr_count,4) = 1;
A_velocity(cnstr_count,nv+4) = 1;
position_cnstr_name{cnstr_count} = 'base_roll(1)+base_roll(end)=0';
velocity_cnstr_name{cnstr_count} = 'base_rolldot(1)+base_rolldot(end)=0';
cnstr_count = cnstr_count+1;

A_position(cnstr_count,5) = 1;
A_position(cnstr_count,nq+5) = -1;
A_velocity(cnstr_count,5) = 1;
A_velocity(cnstr_count,nv+5) = -1;
position_cnstr_name{cnstr_count} = 'base_pitch(1)=base_pitch(end)';
velocity_cnstr_name{cnstr_count} = 'base_pitchdot(1)=base_pitchdot(end)';
cnstr_count = cnstr_count+1;

A_position(cnstr_count,6) = 1;
A_position(cnstr_count,6+nq) = 1;
A_velocity(cnstr_count,6) = 1;
A_velocity(cnstr_count,nv+6) = 1;
position_cnstr_name{cnstr_count} = 'base_yaw(1)+base_yaw(end)=0';
velocity_cnstr_name{cnstr_count} = 'base_yawdot(1)+base_yawdot(end)=0';
cnstr_count = cnstr_count+1;

back_bkz_position = robot.getBody(robot.findJointInd('back_bkz')).position_num;
back_bkz_velocity = robot.getBody(robot.findJointInd('back_bkz')).velocity_num;
A_position(cnstr_count,back_bkz_position) = 1;
A_position(cnstr_count,nq+back_bkz_position) = 1;
A_velocity(cnstr_count,back_bkz_velocity) = 1;
A_velocity(cnstr_count,nv+back_bkz_velocity) = 1;
position_cnstr_name{cnstr_count} = 'back_bkz(1)+back_bkz(end)=0';
velocity_cnstr_name{cnstr_count} = 'back_bkzdot(1)+back_bkzdot(end)=0';
cnstr_count = cnstr_count+1;

back_bky_position = robot.getBody(robot.findJointInd('back_bky')).position_num;
back_bky_velocity = robot.getBody(robot.findJointInd('back_bky')).velocity_num;
A_position(cnstr_count,back_bky_position) = 1;
A_position(cnstr_count,nq+back_bky_position) = -1;
A_velocity(cnstr_count,back_bky_velocity) = 1;
A_velocity(cnstr_count,nv+back_bky_velocity) = -1;
position_cnstr_name{cnstr_count} = 'back_bky(1)=back_bky(end)';
velocity_cnstr_name{cnstr_count} = 'back_bkydot(1)=back_bkydot(end)';
cnstr_count = cnstr_count+1;

back_bkx_position = robot.getBody(robot.findJointInd('back_bkx')).position_num;
back_bkx_velocity = robot.getBody(robot.findJointInd('back_bkx')).velocity_num;
A_position(cnstr_count,back_bkx_position) = 1;
A_position(cnstr_count,nq+back_bkx_position) = 1;
A_velocity(cnstr_count,back_bkx_velocity) = 1;
A_velocity(cnstr_count,nv+back_bkx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'back_bkx(1)+back_bkx(end)=0';
velocity_cnstr_name{cnstr_count} = 'back_bkxdot(1)+back_bkxdot(end)=0';
cnstr_count = cnstr_count+1;

l_arm_usy_position = robot.getBody(robot.findJointInd('l_arm_usy')).position_num;
r_arm_usy_position = robot.getBody(robot.findJointInd('r_arm_usy')).position_num;
l_arm_usy_velocity = robot.getBody(robot.findJointInd('l_arm_usy')).velocity_num;
r_arm_usy_velocity = robot.getBody(robot.findJointInd('r_arm_usy')).velocity_num;
A_position(cnstr_count,l_arm_usy_position) = 1;
A_position(cnstr_count,nq+r_arm_usy_position) = -1;
A_velocity(cnstr_count,l_arm_usy_velocity) = 1;
A_velocity(cnstr_count,nv+r_arm_usy_velocity) = -1;
position_cnstr_name{cnstr_count} = 'l_arm_usy(1)=r_arm_usy(end)';
velocity_cnstr_name{cnstr_count} = 'l_arm_usydot(1)=r_arm_usydot(end)';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_arm_usy_position) = 1;
A_position(cnstr_count,nq+l_arm_usy_position) = -1;
A_velocity(cnstr_count,r_arm_usy_velocity) = 1;
A_velocity(cnstr_count,nv+l_arm_usy_velocity) = -1;
position_cnstr_name{cnstr_count} = 'r_arm_usy(1)=l_arm_usy(end)';
velocity_cnstr_name{cnstr_count} = 'r_arm_usydot(1)=l_arm_usydot(end)';
cnstr_count = cnstr_count+1;

l_arm_shx_position = robot.getBody(robot.findJointInd('l_arm_shx')).position_num;
r_arm_shx_position = robot.getBody(robot.findJointInd('r_arm_shx')).position_num;
l_arm_shx_velocity = robot.getBody(robot.findJointInd('l_arm_shx')).velocity_num;
r_arm_shx_velocity = robot.getBody(robot.findJointInd('r_arm_shx')).velocity_num;
A_position(cnstr_count,l_arm_shx_position) = 1;
A_position(cnstr_count,nq+r_arm_shx_position) = 1;
A_velocity(cnstr_count,l_arm_shx_velocity) = 1;
A_velocity(cnstr_count,nv+r_arm_shx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'l_arm_shx(1)+r_arm_shx(end)=0';
velocity_cnstr_name{cnstr_count} = 'l_arm_shxdot(1)+r_arm_shxdot(end)=0';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_arm_shx_position) = 1;
A_position(cnstr_count,nq+l_arm_shx_position) = 1;
A_velocity(cnstr_count,r_arm_shx_velocity) = 1;
A_velocity(cnstr_count,nv+l_arm_shx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'r_arm_shx(1)+l_arm_shx(end)=0';
velocity_cnstr_name{cnstr_count} = 'r_arm_shxdot(1)+l_arm_shxdot(end)=0';
cnstr_count = cnstr_count+1;

l_arm_ely_position = robot.getBody(robot.findJointInd('l_arm_ely')).position_num;
r_arm_ely_position = robot.getBody(robot.findJointInd('r_arm_ely')).position_num;
l_arm_ely_velocity = robot.getBody(robot.findJointInd('l_arm_ely')).velocity_num;
r_arm_ely_velocity = robot.getBody(robot.findJointInd('r_arm_ely')).velocity_num;
A_position(cnstr_count,l_arm_ely_position) = 1;
A_position(cnstr_count,nq+r_arm_ely_position) = -1;
A_velocity(cnstr_count,l_arm_ely_velocity) = 1;
A_velocity(cnstr_count,nv+r_arm_ely_velocity) = -1;
position_cnstr_name{cnstr_count} = 'l_arm_ely(1)=r_arm_ely(end)';
velocity_cnstr_name{cnstr_count} = 'l_arm_elydot(1)=r_arm_elydot(end)';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_arm_ely_position) = 1;
A_position(cnstr_count,nq+l_arm_ely_position) = -1;
A_velocity(cnstr_count,r_arm_ely_velocity) = 1;
A_velocity(cnstr_count,nv+l_arm_ely_velocity) = -1;
position_cnstr_name{cnstr_count} = 'r_arm_ely(1)=l_arm_ely(end)';
velocity_cnstr_name{cnstr_count} = 'r_arm_elydot(1)=l_arm_elydot(end)';
cnstr_count = cnstr_count+1;

l_arm_elx_position = robot.getBody(robot.findJointInd('l_arm_elx')).position_num;
r_arm_elx_position = robot.getBody(robot.findJointInd('r_arm_elx')).position_num;
l_arm_elx_velocity = robot.getBody(robot.findJointInd('l_arm_elx')).velocity_num;
r_arm_elx_velocity = robot.getBody(robot.findJointInd('r_arm_elx')).velocity_num;
A_position(cnstr_count,l_arm_elx_position) = 1;
A_position(cnstr_count,nq+r_arm_elx_position) = 1;
A_velocity(cnstr_count,l_arm_elx_velocity) = 1;
A_velocity(cnstr_count,nv+r_arm_elx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'l_arm_elx(1)+r_arm_elx(end)=0';
velocity_cnstr_name{cnstr_count} = 'l_arm_elxdot(1)+r_arm_elxdot(end)=0';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_arm_elx_position) = 1;
A_position(cnstr_count,nq+l_arm_elx_position) = 1;
A_velocity(cnstr_count,r_arm_elx_velocity) = 1;
A_velocity(cnstr_count,nv+l_arm_elx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'r_arm_elx(1)+l_arm_elx(end)=0';
velocity_cnstr_name{cnstr_count} = 'r_arm_elxdot(1)+l_arm_elxdot(end)=0';
cnstr_count = cnstr_count+1;

l_arm_uwy_position = robot.getBody(robot.findJointInd('l_arm_uwy')).position_num;
r_arm_uwy_position = robot.getBody(robot.findJointInd('r_arm_uwy')).position_num;
l_arm_uwy_velocity = robot.getBody(robot.findJointInd('l_arm_uwy')).velocity_num;
r_arm_uwy_velocity = robot.getBody(robot.findJointInd('r_arm_uwy')).velocity_num;
A_position(cnstr_count,l_arm_uwy_position) = 1;
A_position(cnstr_count,nq+r_arm_uwy_position) = -1;
A_velocity(cnstr_count,l_arm_uwy_velocity) = 1;
A_velocity(cnstr_count,nv+r_arm_uwy_velocity) = -1;
position_cnstr_name{cnstr_count} = 'l_arm_uwy(1)=r_arm_uwy(end)';
velocity_cnstr_name{cnstr_count} = 'l_arm_uwydot(1)=r_arm_uwydot(end)';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_arm_uwy_position) = 1;
A_position(cnstr_count,nq+l_arm_uwy_position) = -1;
A_velocity(cnstr_count,r_arm_uwy_velocity) = 1;
A_velocity(cnstr_count,nv+l_arm_uwy_velocity) = -1;
position_cnstr_name{cnstr_count} = 'r_arm_uwy(1)=l_arm_uwy(end)';
velocity_cnstr_name{cnstr_count} = 'r_arm_uwydot(1)=l_arm_uwydot(end)';
cnstr_count = cnstr_count+1;

l_arm_mwx_position = robot.getBody(robot.findJointInd('l_arm_mwx')).position_num;
r_arm_mwx_position = robot.getBody(robot.findJointInd('r_arm_mwx')).position_num;
l_arm_mwx_velocity = robot.getBody(robot.findJointInd('l_arm_mwx')).velocity_num;
r_arm_mwx_velocity = robot.getBody(robot.findJointInd('r_arm_mwx')).velocity_num;
A_position(cnstr_count,l_arm_mwx_position) = 1;
A_position(cnstr_count,nq+r_arm_mwx_position) = 1;
A_velocity(cnstr_count,l_arm_mwx_velocity) = 1;
A_velocity(cnstr_count,nv+r_arm_mwx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'l_arm_mwx(1)+r_arm_mwx(end)=0';
velocity_cnstr_name{cnstr_count} = 'l_arm_mwxdot(1)+r_arm_mwxdot(end)=0';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_arm_mwx_position) = 1;
A_position(cnstr_count,nq+l_arm_mwx_position) = 1;
A_velocity(cnstr_count,r_arm_mwx_velocity) = 1;
A_velocity(cnstr_count,nv+l_arm_mwx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'r_arm_mwx(1)+l_arm_mwx(end)=0';
velocity_cnstr_name{cnstr_count} = 'r_arm_mwxdot(1)+l_arm_mwxdot(end)=0';
cnstr_count = cnstr_count+1;


l_leg_hpz_position = robot.getBody(robot.findJointInd('l_leg_hpz')).position_num;
r_leg_hpz_position = robot.getBody(robot.findJointInd('r_leg_hpz')).position_num;
l_leg_hpz_velocity = robot.getBody(robot.findJointInd('l_leg_hpz')).velocity_num;
r_leg_hpz_velocity = robot.getBody(robot.findJointInd('r_leg_hpz')).velocity_num;
A_position(cnstr_count,l_leg_hpz_position) = 1;
A_position(cnstr_count,nq+r_leg_hpz_position) = 1;
A_velocity(cnstr_count,l_leg_hpz_velocity) = 1;
A_velocity(cnstr_count,nv+r_leg_hpz_velocity) = 1;
position_cnstr_name{cnstr_count} = 'l_leg_hpz(1)+r_leg_hpz(end)=0';
velocity_cnstr_name{cnstr_count} = 'l_leg_hpzdot(1)+r_leg_hpzdot(end)=0';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_leg_hpz_position) = 1;
A_position(cnstr_count,nq+l_leg_hpz_position) = 1;
A_velocity(cnstr_count,r_leg_hpz_velocity) = 1;
A_velocity(cnstr_count,nv+l_leg_hpz_velocity) = 1;
position_cnstr_name{cnstr_count} = 'r_leg_hpz(1)+l_leg_hpz(end)=0';
velocity_cnstr_name{cnstr_count} = 'r_leg_hpzdot(1)+l_leg_hpzdot(end) = 0';
cnstr_count = cnstr_count+1;

l_leg_hpx_position = robot.getBody(robot.findJointInd('l_leg_hpx')).position_num;
r_leg_hpx_position = robot.getBody(robot.findJointInd('r_leg_hpx')).position_num;
l_leg_hpx_velocity = robot.getBody(robot.findJointInd('l_leg_hpx')).velocity_num;
r_leg_hpx_velocity = robot.getBody(robot.findJointInd('r_leg_hpx')).velocity_num;
A_position(cnstr_count,l_leg_hpx_position) = 1;
A_position(cnstr_count,nq+r_leg_hpx_position) = 1;
A_velocity(cnstr_count,l_leg_hpx_velocity) = 1;
A_velocity(cnstr_count,nv+r_leg_hpx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'l_leg_hpx(1)+r_leg_hpx(end)=0';
velocity_cnstr_name{cnstr_count} = 'l_leg_hpxdot(1)+r_leg_hpxdot(end)=0';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_leg_hpx_position) = 1;
A_position(cnstr_count,nq+l_leg_hpx_position) = 1;
A_velocity(cnstr_count,r_leg_hpx_velocity) = 1;
A_velocity(cnstr_count,nv+l_leg_hpx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'r_leg_hpx(1)+l_leg_hpx(end)=0';
velocity_cnstr_name{cnstr_count} = 'r_leg_hpxdot(1)+l_leg_hpxdot(end)=0';
cnstr_count = cnstr_count+1;

l_leg_hpy_position = robot.getBody(robot.findJointInd('l_leg_hpy')).position_num;
r_leg_hpy_position = robot.getBody(robot.findJointInd('r_leg_hpy')).position_num;
l_leg_hpy_velocity = robot.getBody(robot.findJointInd('l_leg_hpy')).velocity_num;
r_leg_hpy_velocity = robot.getBody(robot.findJointInd('r_leg_hpy')).velocity_num;
A_position(cnstr_count,l_leg_hpy_position) = 1;
A_position(cnstr_count,nq+r_leg_hpy_position) = -1;
A_velocity(cnstr_count,l_leg_hpy_velocity) = 1;
A_velocity(cnstr_count,nv+r_leg_hpy_velocity) = -1;
position_cnstr_name{cnstr_count} = 'l_leg_hpy(1)=r_leg_hpy(end)';
velocity_cnstr_name{cnstr_count} = 'l_leg_hpydot(1)=r_leg_hpydot(end)';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_leg_hpy_position) = 1;
A_position(cnstr_count,nq+l_leg_hpy_position) = -1;
A_velocity(cnstr_count,r_leg_hpy_velocity) = 1;
A_velocity(cnstr_count,nv+l_leg_hpy_velocity) = -1;
position_cnstr_name{cnstr_count} = 'r_leg_hpy(1)=l_leg_hpy(end)';
velocity_cnstr_name{cnstr_count} = 'r_leg_hpydot(1)=l_leg_hpydot(end)';
cnstr_count = cnstr_count+1;

l_leg_kny_position = robot.getBody(robot.findJointInd('l_leg_kny')).position_num;
r_leg_kny_position = robot.getBody(robot.findJointInd('r_leg_kny')).position_num;
l_leg_kny_velocity = robot.getBody(robot.findJointInd('l_leg_kny')).velocity_num;
r_leg_kny_velocity = robot.getBody(robot.findJointInd('r_leg_kny')).velocity_num;
A_position(cnstr_count,l_leg_kny_position) = 1;
A_position(cnstr_count,nq+r_leg_kny_position) = -1;
A_velocity(cnstr_count,l_leg_kny_velocity) = 1;
A_velocity(cnstr_count,nv+r_leg_kny_velocity) = -1;
position_cnstr_name{cnstr_count} = 'l_leg_kny(1)=r_leg_kny(end)';
velocity_cnstr_name{cnstr_count} = 'l_leg_knydot(1)=r_leg_knydot(end)';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_leg_kny_position) = 1;
A_position(cnstr_count,nq+l_leg_kny_position) = -1;
A_velocity(cnstr_count,r_leg_kny_velocity) = 1;
A_velocity(cnstr_count,nv+l_leg_kny_velocity) = -1;
position_cnstr_name{cnstr_count} = 'r_leg_kny(1)=l_leg_kny(end)';
velocity_cnstr_name{cnstr_count} = 'r_leg_knydot(1)=l_leg_knydot(end)';
cnstr_count = cnstr_count+1;

l_leg_aky_position = robot.getBody(robot.findJointInd('l_leg_aky')).position_num;
r_leg_aky_position = robot.getBody(robot.findJointInd('r_leg_aky')).position_num;
l_leg_aky_velocity = robot.getBody(robot.findJointInd('l_leg_aky')).velocity_num;
r_leg_aky_velocity = robot.getBody(robot.findJointInd('r_leg_aky')).velocity_num;
A_position(cnstr_count,l_leg_aky_position) = 1;
A_position(cnstr_count,nq+r_leg_aky_position) = -1;
A_velocity(cnstr_count,l_leg_aky_velocity) = 1;
A_velocity(cnstr_count,nv+r_leg_aky_velocity) = -1;
position_cnstr_name{cnstr_count} = 'l_leg_aky(1)=r_leg_aky(end)';
velocity_cnstr_name{cnstr_count} = 'l_leg_akydot(1)=r_leg_akydot(end)';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_leg_aky_position) = 1;
A_position(cnstr_count,nq+l_leg_aky_position) = -1;
A_velocity(cnstr_count,r_leg_aky_velocity) = 1;
A_velocity(cnstr_count,nv+l_leg_aky_velocity) = -1;
position_cnstr_name{cnstr_count} = 'r_leg_aky(1)=l_leg_aky(end)';
velocity_cnstr_name{cnstr_count} = 'r_leg_akydot(1)=l_leg_akydot(end)';
cnstr_count = cnstr_count+1;

l_leg_akx_position = robot.getBody(robot.findJointInd('l_leg_akx')).position_num;
r_leg_akx_position = robot.getBody(robot.findJointInd('r_leg_akx')).position_num;
l_leg_akx_velocity = robot.getBody(robot.findJointInd('l_leg_akx')).velocity_num;
r_leg_akx_velocity = robot.getBody(robot.findJointInd('r_leg_akx')).velocity_num;
A_position(cnstr_count,l_leg_akx_position) = 1;
A_position(cnstr_count,nq+r_leg_akx_position) = 1;
A_velocity(cnstr_count,l_leg_akx_velocity) = 1;
A_velocity(cnstr_count,nv+r_leg_akx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'l_leg_akx(1)+r_leg_akx(end)=0';
velocity_cnstr_name{cnstr_count} = 'l_leg_akxdot(1)+r_leg_akxdot(end)=0';
cnstr_count = cnstr_count+1;
A_position(cnstr_count,r_leg_akx_position) = 1;
A_position(cnstr_count,nq+l_leg_akx_position) = 1;
A_velocity(cnstr_count,r_leg_akx_velocity) = 1;
A_velocity(cnstr_count,nv+l_leg_akx_velocity) = 1;
position_cnstr_name{cnstr_count} = 'r_leg_akx(1)+l_leg_akx(end)=0';
velocity_cnstr_name{cnstr_count} = 'r_leg_akxdot(1)+l_leg_akxdot(end)=0';
cnstr_count = cnstr_count+1;

neck_ay_position = robot.getBody(robot.findJointInd('neck_ay')).position_num;
neck_ay_velocity = robot.getBody(robot.findJointInd('neck_ay')).velocity_num;
A_position(cnstr_count,neck_ay_position) = 1;
A_position(cnstr_count,nq+neck_ay_position) = -1;
A_velocity(cnstr_count,neck_ay_velocity) = 1;
A_velocity(cnstr_count,nv+neck_ay_velocity) = -1;
position_cnstr_name{cnstr_count} = 'neck_ay(1)=neck_ay(end)';
velocity_cnstr_name{cnstr_count} = 'neck_aydot(1)=neck_aydot(end)';
cnstr_count = cnstr_count+1;

A_velocity(cnstr_count,1) = 1;
A_velocity(cnstr_count,nv+1) = -1;
velocity_cnstr_name{cnstr_count} = 'base_xdot(1)=base_xdot(end)';
cnstr_count = cnstr_count+1;

position_cnstr = LinearConstraint(zeros(num_cnstr,1),zeros(num_cnstr,1),A_position);
position_cnstr = position_cnstr.setName(position_cnstr_name);
cdfkp = cdfkp.addLinearConstraint(position_cnstr,[cdfkp.q_inds(:,1);cdfkp.q_inds(:,end)]);
velocity_cnstr = LinearConstraint(zeros(num_cnstr+1,1),zeros(num_cnstr+1,1),A_velocity);
velocity_cnstr = velocity_cnstr.setName(velocity_cnstr_name);
cdfkp = cdfkp.addLinearConstraint(velocity_cnstr,[cdfkp.v_inds(:,1);cdfkp.v_inds(:,end)]);

A_angular_momentum = zeros(3,6);
A_angular_momentum(1,1) = 1;
A_angular_momentum(1,4) = 1;
A_angular_momentum(2,2) = 1;
A_angular_momentum(2,5) = -1;
A_angular_momentum(3,3) = 1;
A_angular_momentum(3,6) = -1;
angular_momentum_cnstr_name = {'Hx(1)+Hx(end)=0';'Hy(1)=Hy(end)';'Hz(1)+Hz(end)=0'};
angular_momentum_cnstr = LinearConstraint(zeros(3,1),zeros(3,1),A_angular_momentum);
angular_momentum_cnstr = angular_momentum_cnstr.setName(angular_momentum_cnstr_name);
cdfkp = cdfkp.addLinearConstraint(angular_momentum_cnstr,[cdfkp.H_inds(:,1);cdfkp.H_inds(:,end)]);
end