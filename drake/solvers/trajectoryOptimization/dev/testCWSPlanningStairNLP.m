function testCWSPlanningStairNLP(mode)
warning('off','Drake:RigidBody:SimplifiedCollisionGeometry');
warning('off','Drake:RigidBody:NonPositiveInertiaMatrix');
warning('off','Drake:RigidBodyManipulator:UnsupportedContactPoints');
warning('off','Drake:RigidBodyManipulator:UnsupportedVelocityLimits');
warning('off','Drake:RigidBodyManipulator:ReplacedCylinder');
robot = RigidBodyManipulator([getDrakePath,'/examples/Atlas/urdf/atlas_minimal_contact.urdf'],struct('floating',true));
nq = robot.getNumPositions();
nv = robot.getNumVelocities();

r_foot = robot.findLinkId('r_foot');
l_foot = robot.findLinkId('l_foot');
r_toe = robot.getBody(r_foot).getTerrainContactPoints('toe');
r_midfoot_rear = robot.getBody(r_foot).getTerrainContactPoints('midfoot_rear');
l_toe = robot.getBody(r_foot).getTerrainContactPoints('toe');
l_midfoot_rear = robot.getBody(r_foot).getTerrainContactPoints('midfoot_rear');
l_foot_contact_pts = [l_toe l_midfoot_rear];
r_foot_contact_pts = [r_toe r_midfoot_rear];
l_hand = robot.findLinkId('l_hand');
r_hand = robot.findLinkId('r_hand');
l_hand_pt = [0;0.2;0];
r_hand_pt = [0;-0.2;0];
pelvis = robot.findLinkId('pelvis');
utorso = robot.findLinkId('utorso');

xstar = load([getDrakePath,'/examples/Atlas/data/atlas_fp.mat'],'xstar');
xstar = xstar.xstar;
xstar(robot.findPositionIndices('r_arm_shz')) = 1.3;
xstar(robot.findPositionIndices('r_arm_shx')) = 0;
xstar(robot.findPositionIndices('r_arm_elx')) = -0.5;
xstar(robot.findPositionIndices('r_arm_ely')) = 0;
xstar(robot.findPositionIndices('l_arm_shz')) = -1.3;
xstar(robot.findPositionIndices('l_arm_shx')) = 0;
xstar(robot.findPositionIndices('l_arm_elx')) = 0.5;
xstar(robot.findPositionIndices('l_arm_ely')) = 0;
xstar = robot.resolveConstraints(xstar);
if(robot.getBody(pelvis).floating == 1)
  qstar = xstar(1:nq);
elseif(robot.getBody(pelvis).floating == 2)
  qstar = zeros(nq,1);
  qstar(1:3) = xstar(1:3);
  qstar(4:7) = rpy2quat(xstar(4:6));
  qstar(8:nq) = xstar(7:nq-1);
end
vstar = xstar(end-nv+1:end);

box_size = [0.29 39*0.0254 0.22];
num_stairs = 5;
box_tops = (bsxfun(@times,[0.1 0 0],ones(num_stairs,1))+bsxfun(@times,[box_size(1) 0 box_size(3)],(0:num_stairs-1)'))';
for i = 1:num_stairs
  stair = RigidBodyBox(box_size,box_tops(:,i)+[0;0;-box_size(3)/2],[0;0;0]);
  robot = robot.addGeometryToBody('world',stair);
end
robot = robot.compile();

v = robot.constructVisualizer();

kinsol_star = robot.doKinematics(qstar,vstar);
lfoot_pos_star = robot.forwardKin(kinsol_star,l_foot,[0;0;0],2);
rfoot_pos_star = robot.forwardKin(kinsol_star,r_foot,[0;0;0],2);
lfoot_contact_pos_star = robot.forwardKin(kinsol_star,l_foot,l_foot_contact_pts,0);
rfoot_contact_pos_star = robot.forwardKin(kinsol_star,r_foot,r_foot_contact_pts,0);
com_star = robot.getCOM(kinsol_star);

robot_mass = robot.getMass();
gravity = 9.81;

mu_ground = 1;
num_fc_edges = 8;
theta = linspace(0,2*pi,num_fc_edges);
ground_fc_edges = robot_mass*gravity*[mu_ground*cos(theta);mu_ground*sin(theta);ones(1,num_fc_edges)];
lfoot_cw = LinearFrictionConeWrench(robot,l_foot,l_foot_contact_pts,ground_fc_edges);
rfoot_cw = LinearFrictionConeWrench(robot,r_foot,r_foot_contact_pts,ground_fc_edges);

nT = 12;
rfoot_takeoff_idx = 2;
rfoot_land_idx = 6;
lfoot_takeoff_idx = 7;
lfoot_land_idx = 11;

lfoot_contact_pos1 = lfoot_contact_pos_star+repmat([box_size(1)+0.02;0;box_size(3)],1,4);
rfoot_contact_pos1 = rfoot_contact_pos_star+repmat([box_size(1)+0.02;0;box_size(3)],1,4);

lfoot_contact_wrench0 = struct('active_knot',1:lfoot_takeoff_idx,'cw',lfoot_cw,'contact_pos',lfoot_contact_pos_star);
lfoot_contact_wrench1 = struct('active_knot',lfoot_land_idx:nT,'cw',lfoot_cw,'contact_pos',lfoot_contact_pos1);
rfoot_contact_wrench0 = struct('active_knot',1:rfoot_takeoff_idx,'cw',rfoot_cw,'contact_pos',rfoot_contact_pos_star);
rfoot_contact_wrench1 = struct('active_knot',rfoot_land_idx:nT,'cw',rfoot_cw,'contact_pos',rfoot_contact_pos1);

Q_comddot = eye(3);
Qv = 0.1*eye(nv);
Q = eye(nq);
Q(1,1) = 0;
Q(2,2) = 0;
Q(6,6) = 0;
T_lb = 0.4;
T_ub = 0.7;
tf_range = [T_lb,T_ub];
q_nom = repmat(qstar,1,nT);
contact_wrench_struct = [lfoot_contact_wrench0 lfoot_contact_wrench1 rfoot_contact_wrench0 rfoot_contact_wrench1];
cws_margin_cost = 100;
disturbance_pos = repmat(com_star,1,nT)+bsxfun(@times,(mean(lfoot_contact_pos1,2)-mean(lfoot_contact_pos_star,2))/(nT-1),0:nT-1);
Qw = eye(6);
fccdfkp = FixedContactsFixedDisturbanceComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos);

fccdfkp = fccdfkp.setSolverOptions('snopt','print','test_fccdfkp_stairs.out');
fccdfkp = fccdfkp.setSolverOptions('snopt','majoroptimalitytolerance',1e-5);

% lfoot_off_ground
lfoot_offground_cnstr1 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);0.05*ones(1,4)],repmat([box_tops(1,2)-box_size(1)/2-0.03;nan(2,1)],1,4));
fccdfkp = fccdfkp.addConstraint(lfoot_offground_cnstr1,{lfoot_takeoff_idx+1});
lfoot_offground_cnstr2 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);(box_tops(3,2)+0.02)*ones(1,4)],[repmat([box_tops(1,2)-box_size(1)/2+0.05;nan(2,1)],1,2) repmat([box_tops(1,2)-box_size(1)/2;nan(2,1)],1,2)]);
fccdfkp = fccdfkp.addConstraint(lfoot_offground_cnstr2,{lfoot_takeoff_idx+2});
lfoot_offground_cnstr3 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);(box_tops(3,2)+0.03)*ones(1,4)],repmat([box_tops(1,2)+box_size(1)/2-0.03;nan(2,1)],1,4));
fccdfkp = fccdfkp.addConstraint(lfoot_offground_cnstr3,{lfoot_takeoff_idx+3});

% rfoot_off_ground
rfoot_offground_cnstr1 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);0.05*ones(1,4)],repmat([box_tops(1,2)-box_size(1)/2-0.03;nan(2,1)],1,4));
fccdfkp = fccdfkp.addConstraint(rfoot_offground_cnstr1,{rfoot_takeoff_idx+1});
rfoot_offground_cnstr2 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);(box_tops(3,2)+0.02)*ones(1,4)],[repmat([box_tops(1,2)-box_size(1)/2+0.05;nan(2,1)],1,2) repmat([box_tops(1,2)-box_size(1)/2;nan(2,1)],1,2)]);
fccdfkp = fccdfkp.addConstraint(rfoot_offground_cnstr2,{rfoot_takeoff_idx+2});
rfoot_offground_cnstr3 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);(box_tops(3,2)+0.03)*ones(1,4)],repmat([box_tops(1,2)+box_size(1)/2-0.03;nan(2,1)],1,4));
fccdfkp = fccdfkp.addConstraint(rfoot_offground_cnstr3,{rfoot_takeoff_idx+3});

% dt bound
fccdfkp = fccdfkp.addConstraint(BoundingBoxConstraint(0.05*ones(nT-1,1),0.1*ones(nT-1,1)),fccdfkp.h_inds);

x_init = fccdfkp.getInitialVars(repmat(qstar,1,nT),zeros(nv,nT),0.1*ones(nT-1,1));
if(mode == 1)
  tic
  [x_sol,cost,info] = fccdfkp.solve(x_init);
  toc
  sol = fccdfkp.retrieveSolution(x_sol);
  if(info<10)
    save('test_fccdfkp_stair.mat','sol');
  end
elseif(mode == 2)
  load('test_fccdfkp_stair.mat');
end
keyboard;
end