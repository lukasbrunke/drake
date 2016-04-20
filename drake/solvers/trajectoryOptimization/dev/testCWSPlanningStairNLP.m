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
r_heel = robot.getBody(r_foot).getTerrainContactPoints('heel');
l_toe = robot.getBody(r_foot).getTerrainContactPoints('toe');
l_midfoot_rear = robot.getBody(r_foot).getTerrainContactPoints('midfoot_rear');
l_heel = robot.getBody(l_foot).getTerrainContactPoints('heel');
l_foot_contact_pts0 = [l_toe l_heel];
r_foot_contact_pts0 = [r_toe r_heel];
l_foot_contact_pts1 = [l_toe l_midfoot_rear];
r_foot_contact_pts1 = [r_toe r_midfoot_rear];
l_hand = robot.findLinkId('l_hand');
r_hand = robot.findLinkId('r_hand');
l_hand_pt = [0;0.2;0];
r_hand_pt = [0;-0.2;0];
pelvis = robot.findLinkId('pelvis');
utorso = robot.findLinkId('utorso');

l_leg_kny = robot.findPositionIndices('l_leg_kny');
r_leg_kny = robot.findPositionIndices('r_leg_kny');
back = robot.findPositionIndices('back');
l_arm = robot.findPositionIndices('l_arm');
r_arm = robot.findPositionIndices('r_arm');

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
  floating_base_idx = (1:6)';
elseif(robot.getBody(pelvis).floating == 2)
  qstar = zeros(nq,1);
  qstar(1:3) = xstar(1:3);
  qstar(4:7) = rpy2quat(xstar(4:6));
  qstar(8:nq) = xstar(7:nq-1);
  floating_base_idx = (1:7)';
end
vstar = xstar(end-nv+1:end);

kinsol_star = robot.doKinematics(qstar,vstar);
lfoot_pos_star = robot.forwardKin(kinsol_star,l_foot,[0;0;0],2);
rfoot_pos_star = robot.forwardKin(kinsol_star,r_foot,[0;0;0],2);
lfoot_pos0 = lfoot_pos_star;
lfoot_pos0(1) = lfoot_pos0(1);
lfoot_pos0(2) = lfoot_pos0(2);
rfoot_pos0 = rfoot_pos_star;
rfoot_pos0(1) = rfoot_pos0(1);
rfoot_pos0(2) = rfoot_pos0(2);
cnstr = {WorldPositionConstraint(robot,l_foot,[0;0;0],[lfoot_pos0(1:2);nan],[lfoot_pos0(1:2);nan]);...
  WorldPositionConstraint(robot,l_foot,l_foot_contact_pts0,[nan(2,4);zeros(1,4)],[nan(2,4);zeros(1,4)]);...
  WorldPositionConstraint(robot,r_foot,[0;0;0],[rfoot_pos0(1:2);nan],[rfoot_pos0(1:2);nan]);...
  WorldPositionConstraint(robot,r_foot,r_foot_contact_pts0,[nan(2,4);zeros(1,4)],[nan(2,4);zeros(1,4)])};
[q0,info] = robot.inverseKin(qstar,qstar,cnstr{:});
if(info~=1)
  error('ik fails');
end
kinsol0 = robot.doKinematics(q0,vstar,struct('use_mex',false));
lfoot_contact_pos0 = robot.forwardKin(kinsol0,l_foot,l_foot_contact_pts0,0);
rfoot_contact_pos0 = robot.forwardKin(kinsol0,r_foot,r_foot_contact_pts0,0);

lfoot_pos0 = robot.forwardKin(kinsol0,l_foot,[0;0;0],2);
rfoot_pos0 = robot.forwardKin(kinsol0,r_foot,[0;0;0],2);
com0 = robot.getCOM(kinsol0);


box_size = [0.29 39*0.0254 0.22];
num_stairs = 5;
box_tops = (bsxfun(@times,[0.1 0 0],ones(num_stairs,1))+bsxfun(@times,[box_size(1) 0 box_size(3)],(0:num_stairs-1)'))';
for i = 1:num_stairs
  stair = RigidBodyBox(box_size,box_tops(:,i)+[0;0;-box_size(3)/2],[0;0;0]);
  robot = robot.addGeometryToBody('world',stair);
end
robot = robot.compile();

v = robot.constructVisualizer();

robot_mass = robot.getMass();
gravity = 9.81;

mu_ground = 1;
num_fc_edges = 4;
theta = linspace(0,2*pi,num_fc_edges);
ground_fc_edges = robot_mass*gravity*[mu_ground*cos(theta);mu_ground*sin(theta);ones(1,num_fc_edges)];
lfoot_cw0 = LinearFrictionConeWrench(robot,l_foot,l_foot_contact_pts0,ground_fc_edges);
rfoot_cw0 = LinearFrictionConeWrench(robot,r_foot,r_foot_contact_pts0,ground_fc_edges);
lfoot_cw1 = LinearFrictionConeWrench(robot,l_foot,l_foot_contact_pts1,ground_fc_edges);
rfoot_cw1 = LinearFrictionConeWrench(robot,r_foot,r_foot_contact_pts1,ground_fc_edges);

nT = 12;
rfoot_takeoff_idx = 2;
rfoot_land_idx = 6;
lfoot_takeoff_idx = 7;
lfoot_land_idx = 11;

lfoot_contact_pos1 = robot.forwardKin(kinsol0,l_foot,l_foot_contact_pts1,0);
rfoot_contact_pos1 = robot.forwardKin(kinsol0,r_foot,r_foot_contact_pts1,0);
lfoot_contact_pos1 = lfoot_contact_pos1+repmat([box_size(1)+0.025;0;box_size(3)],1,4);
rfoot_contact_pos1 = rfoot_contact_pos1+repmat([box_size(1)+0.025;0;box_size(3)],1,4);

lfoot_contact_wrench0 = struct('active_knot',1:lfoot_takeoff_idx,'cw',lfoot_cw0,'contact_pos',lfoot_contact_pos0);
lfoot_contact_wrench1 = struct('active_knot',lfoot_land_idx:nT,'cw',lfoot_cw1,'contact_pos',lfoot_contact_pos1);
rfoot_contact_wrench0 = struct('active_knot',1:rfoot_takeoff_idx,'cw',rfoot_cw0,'contact_pos',rfoot_contact_pos0);
rfoot_contact_wrench1 = struct('active_knot',rfoot_land_idx:nT,'cw',rfoot_cw1,'contact_pos',rfoot_contact_pos1);

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
disturbance_pos = repmat(com0,1,nT)+bsxfun(@times,(mean(lfoot_contact_pos1,2)-mean(lfoot_contact_pos0,2))/(nT-1),0:nT-1);
Qw = eye(6);
fccdfkp = FixedContactsFixedDisturbanceComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos);
options = struct();
options.use_lin_fc = true;
options.num_fc_edges = num_fc_edges;
options.sos_cnstr_normalizer = robot_mass*gravity*2;
sccdfkp_sos = SearchContactsFixedDisturbanceFullKinematicsSOSPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos,options);
% add kinematic constraints on contact locations
lfoot_cnstr0 = {WorldPositionConstraint(robot,l_foot,[0;0;0],lfoot_pos0(1:3),lfoot_pos0(1:3));...
  WorldQuatConstraint(robot,l_foot,lfoot_pos0(4:7),0)};
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_cnstr0{1},{1});
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_cnstr0{2},{1});
rfoot_cnstr0 = {WorldPositionConstraint(robot,r_foot,[0;0;0],rfoot_pos0(1:3),rfoot_pos0(1:3));...
  WorldQuatConstraint(robot,r_foot,rfoot_pos0(4:7),0)};
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_cnstr0{1},{1});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_cnstr0{2},{1});

lfoot_cnstr1 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts1,repmat([box_tops(1,2)-box_size(1)/2+0.02;0;box_tops(3,2)],1,4),repmat([box_tops(1,2)+box_size(1)/2-0.05;box_size(2)/2;box_tops(3,2)],1,4));
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_cnstr1,{lfoot_land_idx});
rfoot_cnstr1 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts1,repmat([box_tops(1,2)-box_size(1)/2+0.02;-box_size(2)/2;box_tops(3,2)],1,4),repmat([box_tops(1,2)+box_size(1)/2-0.05;0;box_tops(3,2)],1,4));
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_cnstr1,{rfoot_land_idx});

% add centroidal momentum cost
% Q_centroidal_momentum = eye(6);
% sccdfkp_sos = sccdfkp_sos.setCentroidalMomentumCost(Q_centroidal_momentum);

fccdfkp = fccdfkp.setSolverOptions('snopt','print','test_fccdfkp_stairs.out');
fccdfkp = fccdfkp.setSolverOptions('snopt','majoroptimalitytolerance',2e-5);

sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','print','test_sccdfkp_sos_stairs.out');
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','majoroptimalitytolerance',3e-5);
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','iterationslimit',1e5);

% lfoot_off_ground
lfoot_offground_cnstr1 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts1,[nan(2,4);0.1*ones(1,4)],repmat([box_tops(1,2)-box_size(1)/2-0.03;box_size(2)/2;nan(1,1)],1,4));
fccdfkp = fccdfkp.addConstraint(lfoot_offground_cnstr1,{lfoot_takeoff_idx+1});
lfoot_offground_cnstr2 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts1,[nan(2,4);(box_tops(3,2)+0.02)*ones(1,4)],[repmat([box_tops(1,2)-box_size(1)/2;nan(2,1)],1,2) repmat([box_tops(1,2)-box_size(1)/2;box_size(2)/2;nan(1,1)],1,2)]);
fccdfkp = fccdfkp.addConstraint(lfoot_offground_cnstr2,{lfoot_takeoff_idx+2});
lfoot_offground_cnstr3 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts1,[nan(2,4);(box_tops(3,2)+0.03)*ones(1,4)],repmat([box_tops(1,2)+box_size(1)/2-0.03;box_size(2)/2;nan(1,1)],1,4));
fccdfkp = fccdfkp.addConstraint(lfoot_offground_cnstr3,{lfoot_takeoff_idx+3});

sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_offground_cnstr1,{lfoot_takeoff_idx+1});
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_offground_cnstr2,{lfoot_takeoff_idx+2});
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_offground_cnstr3,{lfoot_takeoff_idx+3});

% rfoot_off_ground
rfoot_offground_cnstr1 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts1,[nan(1,4);-box_size(2)/2*ones(1,4);0.1*ones(1,4)],repmat([box_tops(1,2)-box_size(1)/2-0.03;nan(2,1)],1,4));
fccdfkp = fccdfkp.addConstraint(rfoot_offground_cnstr1,{rfoot_takeoff_idx+1});
rfoot_offground_cnstr2 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts1,[nan(1,4);-box_size(2)/2*ones(1,4);(box_tops(3,2)+0.02)*ones(1,4)],[repmat([box_tops(1,2)-box_size(1)/2;nan(2,1)],1,2) repmat([box_tops(1,2)-box_size(1)/2;nan(2,1)],1,2)]);
fccdfkp = fccdfkp.addConstraint(rfoot_offground_cnstr2,{rfoot_takeoff_idx+2});
rfoot_offground_cnstr3 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts1,[nan(1,4);-box_size(2)/2*ones(1,4);(box_tops(3,2)+0.03)*ones(1,4)],repmat([box_tops(1,2)+box_size(1)/2-0.03;nan(2,1)],1,4));
fccdfkp = fccdfkp.addConstraint(rfoot_offground_cnstr3,{rfoot_takeoff_idx+3});

sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_offground_cnstr1,{rfoot_takeoff_idx+1});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_offground_cnstr2,{rfoot_takeoff_idx+2});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_offground_cnstr3,{rfoot_takeoff_idx+3});

% dt bound
dt_cnstr = BoundingBoxConstraint(0.05*ones(nT-1,1),0.1*ones(nT-1,1));
fccdfkp = fccdfkp.addConstraint(dt_cnstr,fccdfkp.h_inds);

sccdfkp_sos = sccdfkp_sos.addConstraint(dt_cnstr,sccdfkp_sos.h_inds);

% fix initial posture
q0_upperbody_init_cnstr = ConstantConstraint(q0([floating_base_idx;back;l_arm;r_arm]));
fccdfkp = fccdfkp.addConstraint(q0_upperbody_init_cnstr,fccdfkp.q_inds([floating_base_idx;back;l_arm;r_arm],1));

sccdfkp_sos = sccdfkp_sos.addConstraint(q0_upperbody_init_cnstr,sccdfkp_sos.q_inds([floating_base_idx;back;l_arm;r_arm],1));

% Do not bend the knee too much
kny_cnstr = BoundingBoxConstraint(-inf(nT,1),1.4*ones(nT,1));
fccdfkp = fccdfkp.addConstraint(kny_cnstr,fccdfkp.q_inds(l_leg_kny,:));
fccdfkp = fccdfkp.addConstraint(kny_cnstr,fccdfkp.q_inds(r_leg_kny,:));

sccdfkp_sos = sccdfkp_sos.addConstraint(kny_cnstr,sccdfkp_sos.q_inds(l_leg_kny,:));
sccdfkp_sos = sccdfkp_sos.addConstraint(kny_cnstr,sccdfkp_sos.q_inds(r_leg_kny,:));

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

prog_lagrangian = FixedMotionSearchCWSmarginLinFC(num_fc_edges,robot_mass,nT,Qw,sol.num_fc_pts,sol.num_grasp_pts,sol.num_grasp_wrench_vert);
[cws_margin_sol,l0,l1,l2,l3,l4,V,solver_sol,info] = prog_lagrangian.findCWSmargin(0,sol.friction_cones,sol.grasp_pos,sol.grasp_wrench_vert,disturbance_pos,sol.momentum_dot,sol.com);
[cws_margin_sol,l0,l1,l2,l3,l4,V,solver_sol,info] = prog_lagrangian.findCWSmargin(cws_margin_sol,sol.friction_cones,sol.grasp_pos,sol.grasp_wrench_vert,disturbance_pos,sol.momentum_dot,sol.com);
keyboard

x_init = sccdfkp_sos.getInitialVars(sol.q,sol.v,sol.dt);
x_init(sccdfkp_sos.world_momentum_dot_inds) = sol.momentum_dot(:);
x_init(sccdfkp_sos.cws_margin_ind) = cws_margin_sol;
x_init = sccdfkp_sos.setL0GramVarVal(x_init,l0);
x_init = sccdfkp_sos.setL1GramVarVal(x_init,l1);
x_init = sccdfkp_sos.setL2GramVarVal(x_init,l2);
x_init = sccdfkp_sos.setL3GramVarVal(x_init,l3);
x_init = sccdfkp_sos.setL4GramVarVal(x_init,l4);
x_init = sccdfkp_sos.setVGramVarVal(x_init,clean(V));

sccdfkp_sos = sccdfkp_sos.fixL2(l2);
tic
[x_sol,cost,info] = sccdfkp_sos.solve(x_init);
toc
end