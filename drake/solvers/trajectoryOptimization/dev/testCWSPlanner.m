function testCWSPlanner(mode)
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
r_heel = robot.getBody(r_foot).getTerrainContactPoints('heel');
l_toe = robot.getBody(r_foot).getTerrainContactPoints('toe');
l_heel = robot.getBody(r_foot).getTerrainContactPoints('heel');
l_foot_contact_pts = [l_toe l_heel];
r_foot_contact_pts = [r_toe r_heel];
l_hand = robot.findLinkId('l_hand');
r_hand = robot.findLinkId('r_hand');
l_hand_pt = [0;0.2;0];
r_hand_pt = [0;-0.2;0];
pelvis = robot.findLinkId('pelvis');
utorso = robot.findLinkId('utorso');

xstar = load([getDrakePath,'/examples/Atlas/data/atlas_fp.mat'],'xstar');
xstar = xstar.xstar;
if(robot.getBody(pelvis).floating == 1)
  qstar = xstar(1:nq);
elseif(robot.getBody(pelvis).floating == 2)
  qstar = zeros(nq,1);
  qstar(1:3) = xstar(1:3);
  qstar(4:7) = rpy2quat(xstar(4:6));
  qstar(8:nq) = xstar(7:nq-1);
end
vstar = zeros(nv,1);
kinsol_star = robot.doKinematics(qstar,vstar,struct('use_mex',false));
rhand_pos_star = robot.forwardKin(kinsol_star,r_hand,r_hand_pt,2);
rhand_rail_pos = rhand_pos_star(1:3)+[0;-0.1;0.4];
rhand_rail = RigidBodyCylinder(0.01,1,rhand_rail_pos,[0;pi/2;0]);
robot = robot.addGeometryToBody('world',rhand_rail);
robot = robot.compile();
v = robot.constructVisualizer();

lfoot_pos_star = robot.forwardKin(kinsol_star,l_foot,[0;0;0],2);
rfoot_pos_star = robot.forwardKin(kinsol_star,r_foot,[0;0;0],2);
lfoot_cnstr_star = {WorldPositionConstraint(robot,l_foot,[0;0;0],lfoot_pos_star(1:3),lfoot_pos_star(1:3)),...
  WorldQuatConstraint(robot,l_foot,lfoot_pos_star(4:7),0)};
rfoot_cnstr_star = {WorldPositionConstraint(robot,r_foot,[0;0;0],rfoot_pos_star(1:3),rfoot_pos_star(1:3)),...
  WorldQuatConstraint(robot,r_foot,rfoot_pos_star(4:7),0)};
pelvis_cnstr = WorldGazeDirConstraint(robot,pelvis,[0;0;1],[0;0;1],0.05*pi);
rhand_cnstr_star = WorldPositionConstraint(robot,r_hand,r_hand_pt,rhand_rail_pos(1:3)-[0.1;0;0],rhand_rail_pos(1:3)+[0.2;0;0]);
utorso_cnstr = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],0.05*pi);
ik0 = InverseKinematics(robot,qstar,lfoot_cnstr_star{:},rfoot_cnstr_star{:},rhand_cnstr_star,pelvis_cnstr,utorso_cnstr);
q0 = ik0.solve(qstar);
kinsol0 = robot.doKinematics(q0,vstar,struct('use_mex',false));
com0 = robot.getCOM(kinsol0);
rhand_pos0 = robot.forwardKin(kinsol0,r_hand,r_hand_pt,0);

lfoot_contact_pos_star = robot.forwardKin(kinsol_star,l_foot,l_foot_contact_pts,0);
rfoot_contact_pos_star = robot.forwardKin(kinsol_star,r_foot,r_foot_contact_pts,0);
rfoot_contact_pos_land = rfoot_contact_pos_star;
rfoot_contact_pos_land(1,:) = rfoot_contact_pos_land(1,:)+0.1;

% Constraint for a step
nT = 6;
mu_ground = 1;
gravity = 9.81;
robot_mass = robot.getMass();
rfoot_takeoff_idx = 2;
rfoot_land_idx = nT-1;
num_fc_edges = 4;
theta = linspace(0,2*pi,num_fc_edges+1);
theta = theta(1:end-1);
ground_fc_edge = [mu_ground*cos(theta);mu_ground*sin(theta);ones(1,num_fc_edges)]*robot_mass*gravity;
lfoot_cw = LinearFrictionConeWrench(robot,l_foot,l_foot_contact_pts,ground_fc_edge);
rfoot_cw = LinearFrictionConeWrench(robot,r_foot,r_foot_contact_pts,ground_fc_edge);
rhand_cw = GraspWrenchPolytope(robot,r_hand,r_hand_pt,20*[eye(3) zeros(3);-eye(3) zeros(3)]');
lfoot_contact_wrench = struct('active_knot',1:nT,'cw',lfoot_cw,'contact_pos',lfoot_contact_pos_star);
rfoot_contact_wrench1 = struct('active_knot',1:rfoot_takeoff_idx,'cw',rfoot_cw,'contact_pos',rfoot_contact_pos_star);
rfoot_contact_wrench2 = struct('active_knot',rfoot_land_idx:nT,'cw',rfoot_cw,'contact_pos',rfoot_contact_pos_land);
rhand_contact_wrench = struct('active_knot',1:nT,'cw',rhand_cw,'contact_pos',rhand_pos0);


Q_comddot = eye(3);
Qv = 0.1*eye(nv);
Q = eye(nq);
Q(1,1) = 0;
Q(2,2) = 0;
Q(6,6) = 0;
T_lb = 0.4;
T_ub = 0.7;
tf_range = [T_lb,T_ub];
q_nom = repmat(q0,1,nT);
contact_wrench_struct = [lfoot_contact_wrench rfoot_contact_wrench1 rfoot_contact_wrench2 rhand_contact_wrench];
cws_margin_cost = 100;
disturbance_pos = repmat(com0,1,nT)+[linspace(0,0.1,nT);zeros(2,nT)];
Qw = eye(6);
fccdfkp = FixedContactsFixedDisturbanceComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos);

fccdfkp = fccdfkp.setSolverOptions('snopt','print','test_fccdfkp.out');

% dt bound
fccdfkp = fccdfkp.addConstraint(BoundingBoxConstraint(0.05*ones(nT-1,1),0.15*ones(nT-1,1)),fccdfkp.h_inds);

% right foot off ground
cnstr = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);0.03*ones(1,4)],nan(3,4));
fccdfkp = fccdfkp.addConstraint(cnstr,num2cell(rfoot_takeoff_idx+1:rfoot_land_idx-1));

if(mode == 1)
  x_init = fccdfkp.setInitialVar(repmat(q0,1,nT),zeros(nv,nT),0.1*ones(nT-1,1));
  tic
  [x_sol,cost,info] = fccdfkp.solve(x_init);
  toc
  if(info < 10)
    sol = fccdfkp.retrieveSolution(x_sol);
    fccdfkp.checkSolution(sol);
  end
elseif(mode == 2)
  load('test_fccdfkp.mat');
else
  error('mode is not given');
end
keyboard;
num_fc_pts = zeros(nT,1);
num_grasp_pts = zeros(nT,1);
num_grasp_wrench_vert = cell(nT,1);
fc_contact_pos = cell(nT,1);
fc_axis = cell(nT,1);
fc_mu = cell(nT,1);
grasp_pos = cell(nT,1);
grasp_wrench_vert = cell(nT,1);
num_fc_pts(1:rfoot_takeoff_idx) = 8;
num_fc_pts(rfoot_takeoff_idx+1:rfoot_land_idx-1) = 4;
num_fc_pts(rfoot_land_idx:nT) = 8;
num_grasp_pts = ones(nT,1);
for i = 1:nT
  num_grasp_wrench_vert{i} = rhand_cw.num_wrench_vert;
  fc_contact_pos{i} = lfoot_contact_pos_star;
  fc_axis{i} = repmat([0;0;1],1,4);
  fc_mu{i} = mu_ground*ones(1,4);
  grasp_pos{i} = rhand_pos0;
  grasp_wrench_vert{i} = {rhand_cw.wrench_vert};
end
for i = 1:rfoot_takeoff_idx
  fc_contact_pos{i} = [fc_contact_pos{i} rfoot_contact_pos_star];
  fc_axis{i} = [fc_axis{i} repmat([0;0;1],1,4)];
  fc_mu{i} = [fc_mu{i} mu_ground*ones(1,4)];
end
for i = rfoot_land_idx:nT
  fc_contact_pos{i} = [fc_contact_pos{i} rfoot_contact_pos_land];
  fc_axis{i} = [fc_axis{i} repmat([0;0;1],1,4)];
  fc_mu{i} = [fc_mu{i} mu_ground*ones(1,4)];
end
prog_lagrangian = FixedMotionSearchCWSmarginLinFC(4,robot_mass,nT,Qw,num_fc_pts,num_grasp_pts,num_grasp_wrench_vert);
[cws_margin_sol,l0,l1,l2,l3,l4,solver_sol,info] = prog_lagrangian.findCWSmargin(0,fc_contact_pos,fc_axis,fc_mu,grasp_pos,grasp_wrench_vert,disturbance_pos,sol.momentum_dot,sol.com);
keyboard;
options = struct('use_lin_fc',true);
fccdfkp_sos_planner = SearchContactFixedDisturbanceFullKinematicsSOSPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos,options);

end