function testAtlasWallWalking(options)
if(nargin<1)
  options = struct();
end
if(~isfield(options,'mode'))
  options.mode = 1;
end
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

mirror_q_mat = zeros(nq-1,2*nq);
mirror_q_mat(1,[2,nq+2]) = [1 1];
mirror_q_mat(2,[3,nq+3]) = [1 -1];
mirror_q_mat(3,[4,nq+4]) = [1 1];
mirror_q_mat(4,[5,nq+5]) = [1 -1];
mirror_q_mat(5,[6,nq+6]) = [1 1];
back_bkz = robot.getBody(robot.findJointId('back_bkz')).position_num;
back_bky = robot.getBody(robot.findJointId('back_bky')).position_num;
back_bkx = robot.getBody(robot.findJointId('back_bkx')).position_num;
l_arm_shz = robot.getBody(robot.findJointId('l_arm_shz')).position_num;
l_leg_hpz = robot.getBody(robot.findJointId('l_leg_hpz')).position_num;
l_leg_hpx = robot.getBody(robot.findJointId('l_leg_hpx')).position_num;
l_leg_hpy = robot.getBody(robot.findJointId('l_leg_hpy')).position_num;
l_leg_kny = robot.getBody(robot.findJointId('l_leg_kny')).position_num;
l_leg_aky = robot.getBody(robot.findJointId('l_leg_aky')).position_num;
l_leg_akx = robot.getBody(robot.findJointId('l_leg_akx')).position_num;
l_arm_shx = robot.getBody(robot.findJointId('l_arm_shx')).position_num;
l_arm_ely = robot.getBody(robot.findJointId('l_arm_ely')).position_num;
l_arm_elx = robot.getBody(robot.findJointId('l_arm_elx')).position_num;
l_arm_uwy = robot.getBody(robot.findJointId('l_arm_uwy')).position_num;
l_arm_mwx = robot.getBody(robot.findJointId('l_arm_mwx')).position_num;
l_arm_lwy = robot.getBody(robot.findJointId('l_arm_lwy')).position_num;
r_arm_shz = robot.getBody(robot.findJointId('r_arm_shz')).position_num;
r_leg_hpz = robot.getBody(robot.findJointId('r_leg_hpz')).position_num;
r_leg_hpx = robot.getBody(robot.findJointId('r_leg_hpx')).position_num;
r_leg_hpy = robot.getBody(robot.findJointId('r_leg_hpy')).position_num;
r_leg_kny = robot.getBody(robot.findJointId('r_leg_kny')).position_num;
r_leg_aky = robot.getBody(robot.findJointId('r_leg_aky')).position_num;
r_leg_akx = robot.getBody(robot.findJointId('r_leg_akx')).position_num;
r_arm_shx = robot.getBody(robot.findJointId('r_arm_shx')).position_num;
r_arm_ely = robot.getBody(robot.findJointId('r_arm_ely')).position_num;
r_arm_elx = robot.getBody(robot.findJointId('r_arm_elx')).position_num;
r_arm_uwy = robot.getBody(robot.findJointId('r_arm_uwy')).position_num;
r_arm_mwx = robot.getBody(robot.findJointId('r_arm_mwx')).position_num;
r_arm_lwy = robot.getBody(robot.findJointId('r_arm_lwy')).position_num;
neck_ay = robot.getBody(robot.findJointId('neck_ay')).position_num;
mirror_q_mat(6,[back_bkz nq+back_bkz]) = [1 1];
mirror_q_mat(7,[back_bky nq+back_bky]) = [1 -1];
mirror_q_mat(8,[back_bkx nq+back_bkx]) = [1 1];
mirror_q_mat(9,[l_arm_shz nq+r_arm_shz]) = [1 1];
mirror_q_mat(10,[r_arm_shz nq+l_arm_shz]) = [1 1];
mirror_q_mat(11,[l_leg_hpz nq+r_leg_hpz]) = [1 1];
mirror_q_mat(12,[r_leg_hpz nq+l_leg_hpz]) = [1 1];
mirror_q_mat(13,[l_leg_hpx nq+r_leg_hpx]) = [1 1];
mirror_q_mat(14,[r_leg_hpx nq+l_leg_hpx]) = [1 1];
mirror_q_mat(15,[l_leg_hpy nq+r_leg_hpy]) = [1 -1];
mirror_q_mat(16,[r_leg_hpy nq+l_leg_hpy]) = [1 -1];
mirror_q_mat(17,[l_leg_kny nq+r_leg_kny]) = [1 -1];
mirror_q_mat(18,[r_leg_kny nq+l_leg_kny]) = [1 -1];
mirror_q_mat(19,[l_leg_aky nq+r_leg_aky]) = [1 -1];
mirror_q_mat(20,[r_leg_aky nq+l_leg_aky]) = [1 -1];
mirror_q_mat(21,[l_leg_akx nq+r_leg_akx]) = [1 1];
mirror_q_mat(22,[r_leg_akx nq+l_leg_akx]) = [1 1];
mirror_q_mat(23,[l_arm_shz nq+r_arm_shx]) = [1 1];
mirror_q_mat(24,[r_arm_shz nq+l_arm_shx]) = [1 1];
mirror_q_mat(25,[l_arm_ely nq+r_arm_ely]) = [1 -1];
mirror_q_mat(26,[r_arm_ely nq+l_arm_ely]) = [1 -1];
mirror_q_mat(27,[l_arm_elx nq+r_arm_elx]) = [1 1];
mirror_q_mat(28,[r_arm_elx nq+l_arm_elx]) = [1 1];
mirror_q_mat(29,[l_arm_uwy nq+r_arm_uwy]) = [1 -1];
mirror_q_mat(30,[r_arm_uwy nq+l_arm_uwy]) = [1 -1];
mirror_q_mat(31,[l_arm_mwx nq+r_arm_mwx]) = [1 1];
mirror_q_mat(32,[r_arm_mwx nq+l_arm_mwx]) = [1 1];
mirror_q_mat(33,[l_arm_lwy nq+r_arm_lwy]) = [1 -1];
mirror_q_mat(34,[r_arm_lwy nq+l_arm_lwy]) = [1 -1];
mirror_q_mat(35,[neck_ay nq+neck_ay]) = [1 -1];
mirror_v_mat = [zeros(1,2*nv);mirror_q_mat];
mirror_v_mat(1,[1,nv+1]) = [1 -1];

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
nT = 6;

lfoot_cw = LinearFrictionConeWrench(robot,l_foot,l_foot_contact_pts,ground_fc_edges);
rfoot_cw = LinearFrictionConeWrench(robot,r_foot,r_foot_contact_pts,ground_fc_edges);
rheel_cw = LinearFrictionConeWrench(robot,r_foot,r_heel,ground_fc_edges);
rtoe_cw = LinearFrictionConeWrench(robot,r_foot,r_toe,ground_fc_edges);
lhand_cw = LinearFrictionConeWrench(robot,l_hand,lhand_pt,lwall_fc_edges);


lfoot_contact_wrench0 = struct('active_knot',1:nT,'cw',lfoot_cw);
rheel_contact_wrench0 = struct('active_knot',1:r_heel_takeoff-1,'cw',rheel_cw);
rtoe_contact_wrench0 = struct('active_knot',1:r_toe_takeoff-1,'cw',rtoe_cw);
rfoot_contact_wrench1 = struct('active_knot',r_foot_land+1:nT,'cw',rfoot_cw);
lhand_contact_wrench0 = struct('active_knot',l_hand_active+1:l_hand_inactive-1,'cw',lhand_cw);


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
T_lb = 0.3;
T_ub = 0.5;

cws_margin_cost = 1;
q_nom = repmat(qstar,1,nT);
contact_wrench_struct = [lfoot_contact_wrench0 rheel_contact_wrench0 rtoe_contact_wrench0 rfoot_contact_wrench1,lhand_contact_wrench0];
disturbance_pos = repmat(com0,1,nT)+bsxfun(@times,[0.02;0;0],0:nT-1);
Qw = eye(6);
Q_contact_force = 1e-6*eye(3);
tf_range = [T_lb T_ub];
cdfkp = ComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,q_nom,Q_contact_force,contact_wrench_struct);

% dt bound
dt_bnd = BoundingBoxConstraint([0.03 0.03 0.04 0.04 0.03],[0.05 0.05 0.1 0.1 0.1]);
cdfkp = cdfkp.addConstraint(dt_bnd,cdfkp.h_inds);
% lfoot on ground
lfoot_cnstr0 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);zeros(1,4)],[nan(2,4);zeros(1,4)]);
cdfkp = cdfkp.addConstraint(lfoot_cnstr0,num2cell(1:nT));
lfoot_fixed = WorldFixedBodyPoseConstraint(robot,l_foot);
cdfkp = cdfkp.addConstraint(lfoot_fixed,{1:nT});
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


% lhand off wall
lhand_offwall_cnstr = WorldPositionConstraint(robot,l_hand,lhand_pt,nan(3,1),[nan;l_wall_pos(2)-wall_size(2)/2-0.05;nan]);
cdfkp = cdfkp.addConstraint(lhand_offwall_cnstr,num2cell(1:l_hand_active-1));
% rhand_off_wall
rhand_offwall_cnstr = WorldPositionConstraint(robot,r_hand,rhand_pt,[nan;r_wall_pos(2)+wall_size(2)/2+0.05;nan],nan(3,1));
cdfkp = cdfkp.addConstraint(rhand_offwall_cnstr,num2cell(1:nT));

% torso upright
torso_upright = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],pi/9);
cdfkp = cdfkp.addConstraint(torso_upright,num2cell(1:nT));

% pelvis height bound
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
cdfkp = cdfkp.addConstraint(LinearConstraint(zeros(nq-1,1),zeros(nq-1,1),mirror_q_mat),[cdfkp.q_inds(:,1);cdfkp.q_inds(:,nT)]);
cdfkp = cdfkp.addConstraint(LinearConstraint(zeros(nv,1),zeros(nv,1),mirror_v_mat),[cdfkp.v_inds(:,1);cdfkp.v_inds(:,nT)]);
x_init = zeros(cdfkp.num_vars,1);
x_init(cdfkp.q_inds) = reshape(repmat(q0,1,nT),[],1);
cdfkp = cdfkp.setSolverOptions('snopt','majoriterationslimit',300);
cdfkp = cdfkp.setSolverOptions('snopt','majoroptimalitytolerance',1e-4);
cdfkp = cdfkp.setSolverOptions('snopt','print','wall_walking_cdfkp.out');

if(options.mode == 1)
  [x_sol,F,info] = cdfkp.solve(x_init);
  if(info<10)
    save('test_wall_cdfkp.mat','x_sol');
  end
else
  x_sol =load('test_wall_cdfkp.mat');
  x_sol = x_sol.x_sol;
end
[q_sol,v_sol,h,t,com,comdot,comddot,H,Hdot,lambda,wrench] = cdfkp.parseSolution(x_sol);
keyboard;
num_fc_pts = zeros(1,nT);
for i = 1:length(cdfkp.contact_wrench)
  num_fc_pts(cdfkp.contact_wrench_active_knot{i}) = num_fc_pts(cdfkp.contact_wrench_active_knot{i})+cdfkp.contact_wrench{i}.num_pts;
end
friction_cones = cell(nT,1);
kinsol = cell(nT,1);
for i = 1:nT
  friction_cones{i} = LinearizedFrictionCone.empty(num_fc_pts(i),0);
  kinsol{i} = robot.doKinematics(q_sol(:,i),v_sol(:,i),struct('use_mex',false));
end
fc_cones_count = zeros(nT,1);
for i = 1:length(cdfkp.contact_wrench)
  for j = reshape(cdfkp.contact_wrench_active_knot{i},1,[])
    pos = robot.forwardKin(kinsol{j},cdfkp.contact_wrench{i}.body,cdfkp.contact_wrench{i}.body_pts,0);
    for k = 1:size(pos,2)
      friction_cones{j}(fc_cones_count(j)+1) = LinearizedFrictionCone(pos(:,k),cdfkp.contact_wrench{i}.normal_dir(:,k),cdfkp.contact_wrench{i}.mu_face(k),cdfkp.contact_wrench{i}.FC_edge);
      fc_cones_count(j) = fc_cones_count(j)+1;
    end
  end
end
num_grasp_pts = zeros(1,nT);
num_grasp_wrench_vert = cell(nT,1);
momentum_dot = [robot.getMass()*comddot;Hdot+cross(robot.getMass()*com,comddot)];

lfoot_contact_wrench0 = struct('active_knot',1:nT,'cw',lfoot_cw,'contact_pos',robot.forwardKin(kinsol{1},l_foot,l_foot_contact_pts,0));
rheel_contact_wrench0 = struct('active_knot',1:r_heel_takeoff-1,'cw',rheel_cw,'contact_pos',robot.forwardKin(kinsol{1},r_foot,r_heel,0));
rtoe_contact_wrench0 = struct('active_knot',1:r_toe_takeoff-1,'cw',rtoe_cw,'contact_pos',robot.forwardKin(kinsol{1},r_foot,r_toe,0));
rfoot_contact_wrench1 = struct('active_knot',r_foot_land+1:nT,'cw',rfoot_cw,'contact_pos',robot.forwardKin(kinsol{r_foot_land+1},r_foot,r_foot_contact_pts,0));
lhand_contact_wrench0 = struct('active_knot',l_hand_active+1:l_hand_inactive-1,'cw',lhand_cw,'contact_pos',robot.forwardKin(kinsol{l_hand_active+1},l_hand,lhand_pt,0));

contact_wrench_struct = [lfoot_contact_wrench0 rheel_contact_wrench0 rtoe_contact_wrench0 rfoot_contact_wrench1 lhand_contact_wrench0];
fccdfkp_options = struct();
sccdfkp_sos_options = struct();
fccdfkp = FixedContactsFixedDisturbanceComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos,fccdfkp_options);
sccdfkp_sos = SearchContactsFixedDisturbanceFullKinematicsSOSPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos,sccdfkp_sos_options);
% dt bound
fccdfkp = fccdfkp.addConstraint(dt_bnd,fccdfkp.h_inds);
sccdfkp_sos = sccdfkp_sos.addConstraint(dt_bnd,sccdfkp_sos.h_inds);

% lfoot on ground
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_cnstr0,num2cell(1:nT));

% rfoot on ground
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_cnstr0,num2cell(1:r_heel_takeoff-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(rtoe_cnstr0,num2cell(r_heel_takeoff:r_toe_takeoff-1));

% rfoot above ground
fccdfkp = fccdfkp.addConstraint(rfoot_air_cnstr,num2cell(r_toe_takeoff:r_foot_land-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_air_cnstr,num2cell(r_toe_takeoff:r_foot_land-1));

% rfoot land on ground
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_cnstr1,{r_foot_land});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_cnstr2,{r_foot_land});

% lhand on wall
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_wall_cnstr{1},{l_hand_active});
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_wall_cnstr{2},{l_hand_active});

% lhand off wall
fccdfkp = fccdfkp.addConstraint(lhand_offwall_cnstr,num2cell(1:l_hand_active-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_offwall_cnstr,num2cell(1:l_hand_active-1));

% rhand off wall
fccdfkp = fccdfkp.addConstraint(rhand_offwall_cnstr,num2cell(1:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(rhand_offwall_cnstr,num2cell(1:nT));

% torso upright
fccdfkp = fccdfkp.addConstraint(torso_upright,num2cell(1:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(torso_upright,num2cell(1:nT));

% pelvis height bound
fccdfkp = fccdfkp.addConstraint(BoundingBoxConstraint(0.65*ones(nT,1),0.9*ones(nT,1)),fccdfkp.q_inds(3,:));
sccdfkp_sos = sccdfkp_sos.addConstraint(BoundingBoxConstraint(0.65*ones(nT,1),0.9*ones(nT,1)),sccdfkp_sos.q_inds(3,:));

% not too much pitch on the pelvis
fccdfkp = fccdfkp.addConstraint(BoundingBoxConstraint(-pi/9*ones(nT,1),pi/9*ones(nT,1)),fccdfkp.q_inds(5,:));
sccdfkp_sos = sccdfkp_sos.addConstraint(BoundingBoxConstraint(-pi/9*ones(nT,1),pi/9*ones(nT,1)),sccdfkp_sos.q_inds(5,:));
% l_foot on the left plane
fccdfkp = fccdfkp.addConstraint(lfoot_cnstr2,num2cell(1:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_cnstr2,num2cell(1:nT));
% r_foot on the right plane
fccdfkp = fccdfkp.addConstraint(rfoot_cnstr2,num2cell(1:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_cnstr2,num2cell(1:nT));

% symmetric constraint
fccdfkp = fccdfkp.addConstraint(LinearConstraint(zeros(nq-1,1),zeros(nq-1,1),mirror_q_mat),[fccdfkp.q_inds(:,1);fccdfkp.q_inds(:,nT)]);
fccdfkp = fccdfkp.addConstraint(LinearConstraint(zeros(nv,1),zeros(nv,1),mirror_v_mat),[fccdfkp.v_inds(:,1);fccdfkp.v_inds(:,nT)]);
sccdfkp_sos = sccdfkp_sos.addConstraint(LinearConstraint(zeros(nq-1,1),zeros(nq-1,1),mirror_q_mat),[sccdfkp_sos.q_inds(:,1);sccdfkp_sos.q_inds(:,nT)]);
sccdfkp_sos = sccdfkp_sos.addConstraint(LinearConstraint(zeros(nv,1),zeros(nv,1),mirror_v_mat),[sccdfkp_sos.v_inds(:,1);sccdfkp_sos.v_inds(:,nT)]);

x_init = zeros(fccdfkp.num_vars,1);
x_init(fccdfkp.q_inds) = q_sol(:);
x_init(fccdfkp.v_inds) = v_sol(:);

fccdfkp = fccdfkp.setSolverOptions('snopt','majoriterationslimit',300);
fccdfkp = fccdfkp.setSolverOptions('snopt','majoroptimalitytolerance',1e-4);
fccdfkp = fccdfkp.setSolverOptions('snopt','print','wall_walking_fccdfkp.out');

if(options.mode == 2)
  [x_sol,F,info] = fccdfkp.solve(x_init);
  sol = fccdfkp.retrieveSolution(x_sol);
  if(info<10)
    save('test_wall_fccdfkp.mat','x_sol');
  end
else
  sol = load('test_wall_fccdfkp.mat');
  sol = sol.sol;
end
keyboard;
robot_mass = robot.getMass();
prog_lagrangian = FixedMotionSearchCWSmarginLinFC(num_fc_edges,robot_mass,nT,Qw,sol.num_fc_pts,sol.num_grasp_pts,sol.num_grasp_wrench_vert);
[cws_margin_sol,l0,l1,l2,l3,l4,V,solver_sol,info] = prog_lagrangian.findCWSmargin(0,sol.friction_cones,sol.grasp_pos,sol.grasp_wrench_vert,disturbance_pos,sol.momentum_dot,sol.com);

x_init = sccdfkp_sos.getInitialVars(sol.q,sol.v,sol.dt);
x_init = sccdfkp_sos.setMomentumDot(x_init,sol.momentum_dot);
x_init(sccdfkp_sos.cws_margin_ind) = cws_margin_sol;
x_init = sccdfkp_sos.setL0GramVarVal(x_init,l0);
x_init = sccdfkp_sos.setL1GramVarVal(x_init,l1);
x_init = sccdfkp_sos.setL2GramVarVal(x_init,l2);
x_init = sccdfkp_sos.setL3GramVarVal(x_init,l3);
x_init = sccdfkp_sos.setL4GramVarVal(x_init,l4);
x_init = sccdfkp_sos.setVGramVarVal(x_init,clean(V));

sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','print','wall_walking_sccdfkp_sos.out');
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','majoriterationslimit',300);
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','superbasicslimit',5000);
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','majoroptimalitytolerance',3e-4);

sccdfkp_sos = sccdfkp_sos.addConstraint(BoundingBoxConstraint(cws_margin_sol*1.1,inf),sccdfkp_sos.cws_margin_ind);

[x_sol,F,info] = sccdfkp_sos.solve(x_init);
sos_sol = sccdfkp_sos.retrieveSolution(x_sol);
keyboard;
end

function [t,q,v] = halfstrideToFullStride(t1,q1,v1,mirror_q_mat,mirror_v_mat)
q2 = -mirror_q_mat(:,nq+2:end)\(mirror_q_mat(:,2:nq)*q1(2:end,:));
q2 = [q1(1,:)+q1(1,end)-q1(1,1);q2];
q = [q1 q2(:,2:end)];
v2 = -mirror_v_mat(:,nq+1:end)\(mirror_v_mat(:,1:nq)*v1);
v = [v1 v2(:,2:end)];
t = [t1(:)' reshape(t1(2:end),1,[])+t1(end)];
end