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
l_arm = robot.findPositionIndices('l_arm');
r_arm = robot.findPositionIndices('r_arm');
l_leg = robot.findPositionIndices('l_leg');
r_leg = robot.findPositionIndices('r_leg');
back = robot.findPositionIndices('bk');

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
rwall_fc_edges = rotateVectorToAlign([0;0;1],[0;1;0])*[fc_edges0(1:2,:)*mu_wall;fc_edges0(3,:)];

r_heel_takeoff = 2;
r_toe_takeoff = 3;
l_hand_active = 3;
r_foot_land = 5;
l_hand_inactive = 6;
l_heel_takeoff = 6;
r_hand_active = 7;
l_toe_takeoff = 7;
l_foot_land = 9;
r_hand_inactive = 10;
nT = 10;

lfoot_cw = LinearFrictionConeWrench(robot,l_foot,l_foot_contact_pts,ground_fc_edges);
ltoe_cw = LinearFrictionConeWrench(robot,l_foot,l_toe,ground_fc_edges);
lheel_cw = LinearFrictionConeWrench(robot,l_foot,l_heel,ground_fc_edges);
rfoot_cw = LinearFrictionConeWrench(robot,r_foot,r_foot_contact_pts,ground_fc_edges);
rheel_cw = LinearFrictionConeWrench(robot,r_foot,r_heel,ground_fc_edges);
rtoe_cw = LinearFrictionConeWrench(robot,r_foot,r_toe,ground_fc_edges);
lhand_cw = LinearFrictionConeWrench(robot,l_hand,lhand_pt,lwall_fc_edges);
rhand_cw = LinearFrictionConeWrench(robot,r_hand,rhand_pt,rwall_fc_edges);

lfoot_contact_wrench0 = struct('active_knot',1:l_heel_takeoff-1,'cw',lfoot_cw);
rheel_contact_wrench0 = struct('active_knot',1:r_heel_takeoff-1,'cw',rheel_cw);
rtoe_contact_wrench0 = struct('active_knot',1:r_toe_takeoff-1,'cw',rtoe_cw);
rfoot_contact_wrench1 = struct('active_knot',r_foot_land+1:nT,'cw',rfoot_cw);
lhand_contact_wrench0 = struct('active_knot',l_hand_active+1:l_hand_inactive-1,'cw',lhand_cw);
ltoe_contact_wrench1 = struct('active_knot',l_heel_takeoff:l_toe_takeoff-1,'cw',ltoe_cw);
lfoot_contact_wrench1 = struct('active_knot',l_foot_land+1:nT,'cw',lfoot_cw);
rhand_contact_wrench1 = struct('active_knot',r_hand_active+1:r_hand_inactive-1,'cw',rhand_cw);

Q_comddot = eye(3);
Qv = 0.1*ones(nv,1);
Qv(l_arm) = 10;
Qv(r_arm) = 10;
Qv = diag(Qv);
Q = eye(nq);
Q(1,1) = 0;
Q(2,2) = 0;
Q(5,5) = 10;
Q(6,6) = 10;
T_lb = 0.5;
T_ub = 0.8;

cws_margin_cost = 10;
q_nom = repmat(qstar,1,nT);
contact_wrench_struct = [lfoot_contact_wrench0 rheel_contact_wrench0 rtoe_contact_wrench0 rfoot_contact_wrench1,lhand_contact_wrench0 ltoe_contact_wrench1 lfoot_contact_wrench1 rhand_contact_wrench1];
disturbance_pos = repmat(com0,1,nT)+bsxfun(@times,[0.02;0;0],0:nT-1);
Qw = eye(6);
Q_contact_force = 1e-6*eye(3);
tf_range = [T_lb T_ub];
cdfkp = ComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,q_nom,Q_contact_force,contact_wrench_struct);

% dt bound
dt_bnd = BoundingBoxConstraint(0.03*ones(nT-1,1),0.1*ones(nT-1,1));
cdfkp = cdfkp.addConstraint(dt_bnd,cdfkp.h_inds);
% lfoot on ground
lfoot_cnstr0 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);zeros(1,4)],[nan(2,4);zeros(1,4)]);
cdfkp = cdfkp.addConstraint(lfoot_cnstr0,num2cell(1:l_heel_takeoff-1));
lfoot_fixed = WorldFixedBodyPoseConstraint(robot,l_foot);
cdfkp = cdfkp.addConstraint(lfoot_fixed,{1:l_heel_takeoff-1});
ltoe_fixed = WorldFixedPositionConstraint(robot,l_foot,l_toe);
cdfkp = cdfkp.addConstraint(ltoe_fixed,{l_heel_takeoff-1:l_toe_takeoff-1});
% rfoot on ground
rfoot_cnstr0 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);zeros(1,4)],[nan(2,4);zeros(1,4)]);
cdfkp = cdfkp.addConstraint(rfoot_cnstr0,num2cell(1:r_heel_takeoff-1));
rtoe_cnstr0 = WorldPositionConstraint(robot,r_foot,r_toe,[nan(2,2);zeros(1,2)],[nan(2,2);zeros(1,2)]);
cdfkp = cdfkp.addConstraint(rtoe_cnstr0,num2cell(r_heel_takeoff:r_toe_takeoff-1));
rtoe_fixed = WorldFixedPositionConstraint(robot,r_foot,r_toe);
cdfkp = cdfkp.addConstraint(rtoe_fixed,{1:r_toe_takeoff-1});

% rfoot above ground
rfoot_air_cnstr = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);0.03*ones(1,4)],nan(3,4));
cdfkp = cdfkp.addConstraint(rfoot_air_cnstr,num2cell(r_toe_takeoff:r_foot_land-1));

% rfoot land on ground
rfoot_cnstr1 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);zeros(1,4)],[nan(2,4);zeros(1,4)]);
cdfkp = cdfkp.addConstraint(rfoot_cnstr1,{r_foot_land});
rfoot_cnstr2 = RelativePositionConstraint(robot,r_foot_contact_pts,[0.3*ones(1,4);nan(2,4)],nan(3,4),r_foot,l_foot,[zeros(3,1);1;0;0;0]);
cdfkp = cdfkp.addConstraint(rfoot_cnstr2,{r_foot_land});
rfoot_fixed = WorldFixedBodyPoseConstraint(robot,r_foot);
cdfkp = cdfkp.addConstraint(rfoot_fixed,{r_foot_land:nT});

% lhand on wall
lhand_wall_cnstr = {WorldPositionConstraint(robot,l_hand,lhand_pt,[nan;l_wall_pos(2)-wall_size(2)/2;0],[nan;l_wall_pos(2)-wall_size(2)/2;2]);...
  WorldGazeDirConstraint(robot,l_hand,lhand_dir,[0;1;0],pi/6)};
cdfkp = cdfkp.addConstraint(lhand_wall_cnstr{1},{l_hand_active});
cdfkp = cdfkp.addConstraint(lhand_wall_cnstr{2},{l_hand_active});
lhand_fixed = WorldFixedPositionConstraint(robot,l_hand,lhand_pt);
cdfkp = cdfkp.addConstraint(lhand_fixed,{l_hand_active:l_hand_inactive-1});

% rhand on wall
rhand_wall_cnstr = {WorldPositionConstraint(robot,r_hand,rhand_pt,[nan;r_wall_pos(2)+wall_size(2)/2;0],[nan;r_wall_pos(2)+wall_size(2)/2;2]);...
  WorldGazeDirConstraint(robot,r_hand,rhand_dir,[0;-1;0],pi/6)};
cdfkp = cdfkp.addConstraint(rhand_wall_cnstr{1},{r_hand_active});
cdfkp = cdfkp.addConstraint(rhand_wall_cnstr{2},{r_hand_active});
rhand_fixed = WorldFixedPositionConstraint(robot,r_hand,rhand_pt);
cdfkp = cdfkp.addConstraint(rhand_fixed,{r_hand_active:r_hand_inactive-1});

% lfoot above ground
lfoot_air_cnstr = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);0.03*ones(1,4)],nan(3,4));
cdfkp = cdfkp.addConstraint(lfoot_air_cnstr,num2cell(l_toe_takeoff:l_foot_land-1));

% lfoot land on ground
lfoot_cnstr1 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);zeros(1,4)],[nan(2,4);zeros(1,4)]);
cdfkp = cdfkp.addConstraint(lfoot_cnstr1,num2cell(l_foot_land:nT));
lfoot_cnstr2 = RelativePositionConstraint(robot,l_foot_contact_pts,[0.3*ones(1,4);nan(2,4)],nan(3,4),l_foot,r_foot,[zeros(3,1);1;0;0;0]);
cdfkp = cdfkp.addConstraint(lfoot_cnstr2,{l_foot_land});
cdfkp = cdfkp.addConstraint(lfoot_fixed,{l_foot_land:nT});

% lhand off wall
lhand_offwall_cnstr = WorldPositionConstraint(robot,l_hand,lhand_pt,nan(3,1),[nan;l_wall_pos(2)-wall_size(2)/2-0.05;nan]);
cdfkp = cdfkp.addConstraint(lhand_offwall_cnstr,num2cell(1:l_hand_active-1));
% rhand_off_wall
rhand_offwall_cnstr = WorldPositionConstraint(robot,r_hand,rhand_pt,[nan;r_wall_pos(2)+wall_size(2)/2+0.05;nan],nan(3,1));
cdfkp = cdfkp.addConstraint(rhand_offwall_cnstr,num2cell(r_hand_inactive:nT));
cdfkp = cdfkp.addConstraint(rhand_offwall_cnstr,num2cell(1:r_hand_active-1));

% torso upright
torso_upright = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],pi/9);
cdfkp = cdfkp.addConstraint(torso_upright,num2cell(1:nT));

% pelvis not too low
cdfkp = cdfkp.addConstraint(BoundingBoxConstraint(0.65*ones(nT,1),0.9*ones(nT,1)),cdfkp.q_inds(3,:));

% not too much pitch on pelvis
cdfkp = cdfkp.addConstraint(BoundingBoxConstraint(-pi/9*ones(nT,1),pi/9*ones(nT,1)),cdfkp.q_inds(5,:));

% l_foot on the left plane
lfoot_cnstr2 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(1,4);zeros(1,4);nan(1,4)],nan(3,4));
cdfkp = cdfkp.addConstraint(lfoot_cnstr2,num2cell(1:nT));
% r_foot on the right plane
rfoot_cnstr2 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,nan(3,4),[nan(1,4);zeros(1,4);nan(1,4)]);
cdfkp = cdfkp.addConstraint(rfoot_cnstr2,num2cell(1:nT));

% symmetric constraint
cdfkp = cdfkp.addConstraint(LinearConstraint(zeros(nq-1,1),zeros(nq-1,1),[eye(nq-1) -eye(nq-1)]),[cdfkp.q_inds(2:nq,1);cdfkp.q_inds(2:nq,nT)]);
cdfkp = cdfkp.addConstraint(LinearConstraint(zeros(nv,1),zeros(nv,1),[eye(nv) -eye(nv)]),[cdfkp.v_inds(:,1);cdfkp.v_inds(:,nT)]);
x_init = zeros(cdfkp.num_vars,1);
x_init(cdfkp.q_inds) = reshape(repmat(q0,1,nT),[],1);
cdfkp = cdfkp.setSolverOptions('snopt','majoriterationslimit',300);
cdfkp = cdfkp.setSolverOptions('snopt','majoroptimalitytolerance',1e-4);
cdfkp = cdfkp.setSolverOptions('snopt','print','wall_walking_init.out');


[x_sol,F,info] = cdfkp.solve(x_init);
keyboard;
end