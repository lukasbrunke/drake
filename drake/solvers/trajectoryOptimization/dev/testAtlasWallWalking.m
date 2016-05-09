function testAtlasWallWalking()
robot = RigidBodyManipulator([getDrakePath,'/examples/Atlas/urdf/atlas_minimal_contact.urdf'],struct('floating',true));
% add a wall
l_wall_pos = [4;1.05;1];
r_wall_pos = [4;-1.05;1];
wall_size = [10;0.1;2];
l_wall = RigidBodyBox(wall_size,l_wall_pos,[0;0;0]);
r_wall = RigidBodyBox(wall_size,r_wall_pos,[0;0;0]);
robot = robot.addGeometryToBody(1,l_wall);
robot = robot.addGeometryToBody(1,r_wall);
robot = robot.compile();
v = robot.constructVisualizer();
nq = robot.getNumPositions();
nv = robot.getNumVelocities();
l_foot = robot.findLinkId('l_foot');
r_foot = robot.findLinkId('r_foot');
l_toe = robot.getBody(l_foot).getTerrainContactPoints('toe');
l_heel = robot.getBody(l_foot).getTerrainContactPoints('heel');
l_foot_contact_pts = [l_toe l_heel];
r_toe = robot.getBody(r_foot).getTerrainContactPoints('toe');
r_heel = robot.getBody(r_foot).getTerrainContactPoints('heel');
r_foot_contact_pts = [r_toe r_heel];
r_hand = robot.findLinkId('r_hand');
l_hand = robot.findLinkId('l_hand');
rhand_pt = [0;-0.15;0];
lhand_pt = [0;-0.15;0];
rhand_dir = [0;-1;0];
lhand_dir = [0;-1;0];
utorso = robot.findLinkId('utorso');

xstar = load([getDrakePath,'/examples/Atlas/data/atlas_fp.mat']);
xstar = xstar.xstar;
qstar = xstar(1:nq);
lfoot_on_ground0 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(1,4);zeros(1,4);zeros(1,4)],[nan(2,4);zeros(1,4)]);
rfoot_on_ground0 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);zeros(1,4)],[nan(1,4);-0.05*ones(1,4);zeros(1,4)]);
lhand_on_wall = WorldPositionConstraint(robot,l_hand,lhand_pt,[0.2;l_wall_pos(2)-wall_size(2)/2;nan],[l_wall_pos(1)+wall_size(1)/2;l_wall_pos(2)-wall_size(2)/2;nan]);
lhand_dir0 = WorldGazeDirConstraint(robot,l_hand,lhand_dir,[0;1;0],pi/6);
utorso_upright0 = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],pi/20);
ik = InverseKinematics(robot,qstar,lfoot_on_ground0,rfoot_on_ground0,lhand_on_wall,lhand_dir0,utorso_upright0);
ik= ik.addConstraint(ConstantConstraint(qstar(1:6)),1:6);
[q0,F,info] = ik.solve(qstar);
kinsol0 = robot.doKinematics(q0);
com0 = robot.getCOM(kinsol0);

mu_ground = 1;
mu_wall = 1;
num_fc_edges = 4;
theta = linspace(0,2*pi,num_fc_edges+1);
theta = theta(1:end-1);
fc_edges0 = [cos(theta);sin(theta);ones(1,num_fc_edges)];
ground_fc_edges = [fc_edges0(1:2,:)*mu_ground;fc_edges0(3,:)];
lwall_fc_edges = rotateVectorToAlign([0;0;1],[0;-1;0])*[fc_edges0(1:2,:)*mu_wall;fc_edges0(3,:)];

r_heel_takeoff = 2;
r_toe_takeoff = 3;
l_hand_active = 3;
r_foot_land = 5;
l_hand_inactive = 6;
nT = 7;

lfoot_contact_pos0 = robot.forwardKin(kinsol0,l_foot,l_foot_contact_pts,0);
rfoot_contact_pos0 = robot.forwardKin(kinsol0,r_foot,r_foot_contact_pts,0);
lhand_contact_pos0 = robot.forwardKin(kinsol0,l_hand,lhand_pt,0);

lfoot_cw = LinearFrictionConeWrench(robot,l_foot,l_foot_contact_pts,ground_fc_edges);
rfoot_cw = LinearFrictionConeWrench(robot,r_foot,r_foot_contact_pts,ground_fc_edges);
rheel_cw = LinearFrictionConeWrench(robot,r_foot,r_heel,ground_fc_edges);
rtoe_cw = LinearFrictionConeWrench(robot,r_foot,r_toe,ground_fc_edges);
lhand_cw = LinearFrictionConeWrench(robot,l_hand,lhand_pt,lwall_fc_edges);

lfoot_contact_wrench0 = struct('active_knot',1:nT,'cw',lfoot_cw);
rheel_contact_wrench0 = struct('active_knot',1:r_heel_takeoff-1,'cw',rheel_cw);
rtoe_contact_wrench0 = struct('active_knot',1:r_toe_takeoff-1,'cw',rtoe_cw);
rfoot_contact_wrench1 = struct('active_knot',r_foot_land:nT,'cw',rfoot_cw);
lhand_contact_wrench0 = struct('active_knot',l_hand_active:l_hand_inactive,'cw',lhand_cw);

Q_comddot = eye(3);
Qv = 0.1*eye(nv);
Q = eye(nq);
Q(1,1) = 0;
Q(2,2) = 0;
Q(5,5) = 10;
Q(6,6) = 10;
T_lb = 0.2;
T_ub = 0.4;

cws_margin_cost = 10;
q_nom = repmat(qstar,1,nT);
contact_wrench_struct = [lfoot_contact_wrench0 rheel_contact_wrench0 rtoe_contact_wrench0 rfoot_contact_wrench1,lhand_contact_wrench0];
disturbance_pos = repmat(com0,1,nT)+bsxfun(@times,[0.02;0;0],0:nT-1);
Qw = eye(6);
Q_contact_force = 1e-6*eye(3);
tf_range = [T_lb T_ub];
cdfkp = ComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,q_nom,Q_contact_force,contact_wrench_struct);

% dt bound
dt_bnd = BoundingBoxConstraint(0.03*ones(nT-1,1),0.1*ones(nT-1,1));
cdfkp = cdfkp.addConstraint(dt_bnd,cdfkp.h_inds);
% lfoot on ground
lfoot_pose0 = robot.forwardKin(kinsol0,l_foot,[0;0;0],2);
lfoot_cnstr0 = {WorldPositionConstraint(robot,l_foot,[0;0;0],lfoot_pose0(1:3),lfoot_pose0(1:3));...
  WorldQuatConstraint(robot,l_foot,lfoot_pose0(4:7),0)};
cdfkp = cdfkp.addConstraint(lfoot_cnstr0{1},num2cell(1:nT));
cdfkp = cdfkp.addConstraint(lfoot_cnstr0{2},num2cell(1:nT));

% rfoot on ground
rfoot_pose0 = robot.forwardKin(kinsol0,r_foot,[0;0;0],2);
rfoot_cnstr0 = {WorldPositionConstraint(robot,r_foot,[0;0;0],rfoot_pose0(1:3),rfoot_pose0(1:3));...
  WorldQuatConstraint(robot,r_foot,rfoot_pose0(4:7),0)};
cdfkp = cdfkp.addConstraint(rfoot_cnstr0{1},num2cell(1:r_heel_takeoff-1));
cdfkp = cdfkp.addConstraint(rfoot_cnstr0{2},num2cell(1:r_heel_takeoff-1));
rtoe_cnstr0 = WorldPositionConstraint(robot,r_foot,r_toe,rfoot_contact_pos0(:,1:2),rfoot_contact_pos0(:,1:2));
cdfkp = cdfkp.addConstraint(rtoe_cnstr0,num2cell(r_heel_takeoff:r_toe_takeoff-1));

% rfoot above ground
rfoot_air_cnstr = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);0.04*ones(1,4)],nan(3,4));
cdfkp = cdfkp.addConstraint(rfoot_air_cnstr,num2cell(r_toe_takeoff:r_foot_land-1));

% rfoot land on ground
rfoot_cnstr1 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[rfoot_contact_pos0(1,:)+0.3;nan(1,4);zeros(1,4)],[nan(2,4);zeros(1,4)]);
cdfkp = cdfkp.addConstraint(rfoot_cnstr1,{r_foot_land});
rfoot_fixed = WorldFixedBodyPoseConstraint(robot,r_foot);
cdfkp = cdfkp.addConstraint(rfoot_fixed,{r_foot_land:nT});

% lhand on wall
lhand_wall_cnstr = {WorldPositionConstraint(robot,l_hand,lhand_pt,[0.2;l_wall_pos(2)-wall_size(2)/2;0],[nan;l_wall_pos(2)-wall_size(2)/2;2]);...
  WorldGazeDirConstraint(robot,l_hand,lhand_dir,[0;1;0],pi/6)};
cdfkp = cdfkp.addConstraint(lhand_wall_cnstr{1},{l_hand_active});
cdfkp = cdfkp.addConstraint(lhand_wall_cnstr{2},{l_hand_active});
lhand_fixed = WorldFixedPositionConstraint(robot,l_hand,lhand_pt);
cdfkp = cdfkp.addConstraint(lhand_fixed,{l_hand_active:l_hand_inactive});

% rhand off the wall
rhand_offwall_cnstr = WorldPositionConstraint(robot,r_hand,rhand_pt,[nan;r_wall_pos(2)+wall_size(2)/2+0.05;0],[nan;nan;2]);
cdfkp = cdfkp.addConstraint(rhand_offwall_cnstr,num2cell(1:nT));

x_init = zeros(cdfkp.num_vars,1);
x_init(cdfkp.q_inds) = reshape(repmat(q0,1,nT),[],1);
cdfkp = cdfkp.setSolverOptions('snopt','majoriterationslimit',300);
cdfkp = cdfkp.setSolverOptions('snopt','majoroptimalitytolerance',1e-4);
cdfkp = cdfkp.setSolverOptions('snopt','print','wall_walking_init.out');

[x_sol,F,info] = cdfkp.solve(x_init);
keyboard;
end