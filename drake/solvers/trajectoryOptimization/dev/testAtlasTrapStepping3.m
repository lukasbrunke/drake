function testAtlasTrapStepping3(options)
% left hand off the wall, and then come back again.
if(nargin<1)
  options = struct();
end
if(~isfield(options,'mode'))
  options.mode = 1;
end
robot = RigidBodyManipulator([getDrakePath,'/examples/Atlas/urdf/atlas_minimal_contact.urdf'],struct('floating',true));
% add a wall
l_wall_pos = [0.5;0.9;1];
r_wall_pos = [0.5;-0.9;1];
wall_size = [2;0.1;2];
l_wall = RigidBodyBox(wall_size,l_wall_pos,[0;0;0]);
r_wall = RigidBodyBox(wall_size,r_wall_pos,[0;0;0]);
robot = robot.addGeometryToBody(1,l_wall);
robot = robot.addGeometryToBody(1,r_wall);
trap_size = [0.5;1;0.05];
trap_pos = [0.5;0;0];
ground_pos1 = [(trap_pos(1)-trap_size(1)/2+(l_wall_pos(1)-wall_size(1)/2))/2;0;-0.05];
ground_size1 = [(trap_pos(1)-trap_size(1)/2-(l_wall_pos(1)-wall_size(1)/2));l_wall_pos(2)-r_wall_pos(2)+wall_size(2);0.1];
ground1 = RigidBodyBox(ground_size1,ground_pos1,zeros(3,1));
robot = robot.addGeometryToBody(1,ground1);
ground_pos2 = [trap_pos(1);(trap_pos(2)+trap_size(2)/2+l_wall_pos(2)+wall_size(2)/2)/2;-0.05];
ground_size2 = [trap_size(1);(l_wall_pos(2)+wall_size(2)/2-(trap_pos(2)+trap_size(2)/2));0.1];
ground2 = RigidBodyBox(ground_size2,ground_pos2,zeros(3,1));
robot = robot.addGeometryToBody(1,ground2);
ground_pos3 = [trap_pos(1);(trap_pos(2)-trap_size(2)/2+r_wall_pos(2)-wall_size(2)/2)/2;-0.05];
ground_size3 = [trap_size(1);trap_pos(2)-trap_size(2)/2-(r_wall_pos(2)-wall_size(2)/2);0.1];
ground3 = RigidBodyBox(ground_size3,ground_pos3,zeros(3,1));
robot = robot.addGeometryToBody(1,ground3);
ground_pos4 = [(trap_pos(1)+trap_size(1)/2+l_wall_pos(1)+wall_size(1)/2)/2;0;-0.05];
ground_size4 = [l_wall_pos(1)+wall_size(1)/2-((trap_pos(1)+trap_size(1)/2));l_wall_pos(2)-r_wall_pos(2)+wall_size(2);0.1];
ground4 = RigidBodyBox(ground_size4,ground_pos4,zeros(3,1));
robot = robot.addGeometryToBody(1,ground4);
robot = robot.compile();
v = robot.constructVisualizer();
nq = robot.getNumPositions();
nv = robot.getNumVelocities();
l_foot = robot.findLinkId('l_foot');
r_foot = robot.findLinkId('r_foot');
l_toe = robot.getBody(l_foot).getTerrainContactPoints('toe');
l_heel = robot.getBody(l_foot).getTerrainContactPoints('heel');
l_foot_contact_pts = [l_toe l_heel];
l_foot_outer_edge = [l_toe(:,1) l_heel(:,1)];
l_foot_center = mean(l_foot_contact_pts,2);
l_toe_middle = mean(l_toe,2);
r_toe = robot.getBody(r_foot).getTerrainContactPoints('toe');
r_heel = robot.getBody(r_foot).getTerrainContactPoints('heel');
r_foot_contact_pts = [r_toe r_heel];
r_foot_outer_edge = [r_toe(:,2) r_heel(:,2)];
r_foot_center = mean(r_foot_contact_pts,2);
r_toe_middle = mean(r_toe,2);
r_hand = robot.findLinkId('r_hand');
l_hand = robot.findLinkId('l_hand');
rhand_pt = [0;-0.15;0];
lhand_pt = [0;-0.15;0];
rhand_dir = [0;-1;0];
lhand_dir = [0;-1;0];
utorso = robot.findLinkId('utorso');
l_ufarm = robot.findLinkId('l_ufarm');
r_ufarm = robot.findLinkId('r_ufarm');

l_arm = robot.findPositionIndices('l_arm');
r_arm = robot.findPositionIndices('r_arm');
l_leg = robot.findPositionIndices('l_leg');
r_leg = robot.findPositionIndices('r_leg');
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
% find an initial posture
xstar = load([getDrakePath,'/examples/Atlas/data/atlas_fp.mat']);
xstar = xstar.xstar;
qstar = xstar(1:nq);
qstar(l_arm_shx) = -1;
qstar(l_arm_elx) = 1;
qstar(r_arm_shx) = 1;
qstar(r_arm_elx) = -1;
lfoot_on_ground0 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(1,4);zeros(1,4)+0.05;zeros(1,4)],[(trap_pos(1)-trap_size(1)/2-0.05)*ones(1,4);nan(1,4);zeros(1,4)]);
rfoot_on_ground0 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);zeros(1,4)],[(trap_pos(1)-trap_size(1)/2-0.05)*ones(1,4);-0.05*ones(1,4);zeros(1,4)]);
lhand_on_wall = WorldPositionConstraint(robot,l_hand,lhand_pt,[trap_pos(1)-trap_size(1)/2+0.3;l_wall_pos(2)-wall_size(2)/2;1.4],[l_wall_pos(1)+wall_size(1)/2;l_wall_pos(2)-wall_size(2)/2;nan]);
lhand_dir0 = WorldGazeDirConstraint(robot,l_hand,lhand_dir,[0;1;0],pi/2.3);
rhand_on_wall = WorldPositionConstraint(robot,r_hand,rhand_pt,[trap_pos(1)-trap_size(1)/2+0.3;r_wall_pos(2)+wall_size(2)/2;1.4],[r_wall_pos(1)+wall_size(1)/2;r_wall_pos(2)+wall_size(2)/2;nan]);
rhand_dir0 = WorldGazeDirConstraint(robot,l_hand,rhand_dir,[0;-1;0],pi/2.3);
utorso_upright0 = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],pi/6);
ik = InverseKinematics(robot,qstar,lfoot_on_ground0,rfoot_on_ground0,lhand_on_wall,utorso_upright0,rhand_on_wall);
[q0,F,info] = ik.solve(qstar);
kinsol0 = robot.doKinematics(q0);
rhand_pos0 = robot.forwardKin(kinsol0,r_hand,rhand_pt,2);
lhand_pos0 = robot.forwardKin(kinsol0,l_hand,lhand_pt,2);
rfoot_pos0 = robot.forwardKin(kinsol0,r_foot,r_foot_center,2);
lfoot_pos0 = robot.forwardKin(kinsol0,l_foot,l_foot_center,2);
ltoe_middle_pos0 = robot.forwardKin(kinsol0,l_foot,l_toe_middle,0);
rtoe_middle_pos0 = robot.forwardKin(kinsol0,r_foot,r_toe_middle,0);
ltoe_pos0 = robot.forwardKin(kinsol0,l_foot,l_toe,0);
rtoe_pos0 = robot.forwardKin(kinsol0,r_foot,r_toe,0);
com0 = robot.getCOM(kinsol0);

% find the middle posture
rfoot_on_ground1 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,repmat([(trap_pos(1)-trap_size(1)/2);(r_wall_pos(2)+wall_size(2)/2);0],1,4),repmat([(trap_pos(1)+trap_size(1)/2);(trap_pos(2)-trap_size(2)/2-0.03);0],1,4));
lfoot_on_ground1 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[(trap_pos(1)+trap_size(1)/2+0.03)*ones(1,4);zeros(1,4)+0.05;zeros(1,4)],[nan(2,4);zeros(1,4)]);
lhand_pos_cnstr = WorldPositionConstraint(robot,l_hand,lhand_pt,[nan;l_wall_pos(2)-wall_size(2)/2;1],[nan;l_wall_pos(2)-wall_size(2)/2;1]);
rhand_pos_cnstr = WorldPositionConstraint(robot,r_hand,rhand_pt,rhand_pos0(1:3),rhand_pos0(1:3));
utorso_upright1 = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],pi/10);
ik = InverseKinematics(robot,q0,rfoot_on_ground1,lfoot_on_ground1,lhand_pos_cnstr,rhand_pos_cnstr,utorso_upright1);
[q1,F,info] = ik.solve(q0);
kinsol1 = robot.doKinematics(q1);
lfoot_pos1 = robot.forwardKin(kinsol1,l_foot,l_foot_center,2);
lfoot_outer_pos1 = robot.forwardKin(kinsol1,l_foot,l_foot_outer_edge,0);
rfoot_pos1 = robot.forwardKin(kinsol1,r_foot,r_foot_center,2);
lhand_pos1 = robot.forwardKin(kinsol1,l_hand,lhand_pt,0);

% find the end posture
rfoot_on_ground2 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[(trap_pos(1)+trap_size(1)/2+0.03)*ones(1,4);nan(1,4);zeros(1,4)],[nan(1,4);zeros(1,4)-0.05;zeros(1,4)]);
lhand_pos_cnstr1 = WorldPositionConstraint(robot,l_hand,lhand_pt,lhand_pos1,lhand_pos1);
lfoot_pos_cnstr1 = {WorldPositionConstraint(robot,l_foot,l_foot_center,lfoot_pos1(1:3),lfoot_pos1(1:3));...
  WorldQuatConstraint(robot,l_foot,lfoot_pos1(4:7),0)};
qsc2 = QuasiStaticConstraint(robot);
qsc2 = qsc2.addContact(l_foot,l_foot_contact_pts,r_foot,r_foot_contact_pts);
qsc2 = qsc2.setActive(true);
ik = InverseKinematics(robot,q1,lfoot_pos_cnstr1{:},rfoot_on_ground2,lhand_pos_cnstr1,rhand_pos_cnstr,utorso_upright1,qsc2);
[q2,F,info] = ik.solve(q1);
kinsol2 = robot.doKinematics(q2);
rfoot_pos2 = robot.forwardKin(kinsol2,r_foot,r_foot_center,2);
rfoot_outer_pos2 = robot.forwardKin(kinsol2,r_foot,r_foot_outer_edge,0);
com2 = robot.getCOM(kinsol2);

num_ground_fc_edges = 4;
num_wall_fc_edges = 3;
ground_mu = 2;
wall_mu = 1;
ground_fc_theta = linspace(0,2*pi,num_ground_fc_edges+1);
ground_fc_theta = ground_fc_theta(1:end-1);
wall_fc_theta = linspace(0,2*pi,num_wall_fc_edges+1);
wall_fc_theta = wall_fc_theta(1:end-1);
ground_fc = [cos(ground_fc_theta)*ground_mu;sin(ground_fc_theta)*ground_mu;ones(1,num_ground_fc_edges)];
lwall_fc = rotateVectorToAlign([0;0;1],[0;-1;0])*[cos(wall_fc_theta)*wall_mu;sin(wall_fc_theta)*wall_mu;ones(1,num_wall_fc_edges)];
rwall_fc = rotateVectorToAlign([0;0;1],[0;1;0])*[cos(wall_fc_theta)*wall_mu;sin(wall_fc_theta)*wall_mu;ones(1,num_wall_fc_edges)];
lhand_force_max = 400;
rhand_force_max = 400;

rfoot_takeoff0 = 3;
rfoot_land0 = 4;
lhand_takeoff0 = 5;
lhand_land0 = 6;
lfoot_takeoff0 = 7;
lfoot_land0 = 8;
rfoot_takeoff1 = 9;
rfoot_land1 = 10;
hand_off = 11;
nT = 12;

lfoot_cw = LinearFrictionConeWrench(robot,l_foot,l_foot_center,ground_fc);
rfoot_cw = LinearFrictionConeWrench(robot,r_foot,r_foot_center,ground_fc);
lhand_cw = GraspWrenchPolytope(robot,l_hand,lhand_pt,lhand_force_max*[zeros(6,1) [lwall_fc;zeros(3,num_wall_fc_edges)]]);
rhand_cw = GraspWrenchPolytope(robot,r_hand,rhand_pt,rhand_force_max*[zeros(6,1) [rwall_fc;zeros(3,num_wall_fc_edges)]]);
lfoot_cw2 = LinearFrictionConeWrench(robot,l_foot,l_foot_outer_edge,ground_fc);
rfoot_cw2 = LinearFrictionConeWrench(robot,r_foot,r_foot_outer_edge,ground_fc);

lfoot_contact_wrench0 = struct('active_knot',1:lfoot_takeoff0-1,'cw',lfoot_cw,'contact_pos',lfoot_pos0(1:3));
rfoot_contact_wrench0 = struct('active_knot',1:rfoot_takeoff0-1,'cw',rfoot_cw,'contact_pos',rfoot_pos0(1:3));
lfoot_contact_wrench1 = struct('active_knot',lfoot_land0:hand_off-1,'cw',lfoot_cw,'contact_pos',lfoot_pos1(1:3));
rfoot_contact_wrench1 = struct('active_knot',rfoot_land0:rfoot_takeoff1-1,'cw',rfoot_cw,'contact_pos',rfoot_pos1(1:3));
rfoot_contact_wrench2 = struct('active_knot',rfoot_land1:hand_off-1,'cw',rfoot_cw,'contact_pos',rfoot_pos2(1:3));
lfoot_contact_wrench3 = struct('active_knot',hand_off:nT,'cw',lfoot_cw2,'contact_pos',lfoot_outer_pos1);
rfoot_contact_wrench3 = struct('active_knot',hand_off:nT,'cw',rfoot_cw2,'contact_pos',rfoot_outer_pos2);
lhand_contact_wrench0 = struct('active_knot',1:lhand_takeoff0-1,'cw',lhand_cw,'contact_pos',lhand_pos0(1:3));
lhand_contact_wrench1 = struct('active_knot',lhand_land0:hand_off-1,'cw',lhand_cw,'contact_pos',lhand_pos1(1:3));
rhand_contact_wrench = struct('active_knot',1:hand_off-1,'cw',rhand_cw,'contact_pos',rhand_pos0(1:3));


Q_comddot = eye(3);
Qv = 0.1*ones(nv,1);
Qv(4:6) = 10;
Qv(l_arm) = 20;
Qv(r_arm) = 20;
Qv(back_bkx) = 10;
Qv(back_bky) = 10;
Qv = diag(Qv);
Q = ones(nq,1);
Q(1) = 0;
Q(2) = 0;
Q(5) = 10;
Q(6) = 10;
Q = diag(Q);
T_lb = 2.5;
T_ub = 4;
tf_range = [T_lb T_ub];

cws_margin_cost1 = 0;
cws_margin_cost2 = 0.1;
q_nom = repmat(q0,1,nT);
contact_wrench_struct = [lfoot_contact_wrench0 rfoot_contact_wrench0 lfoot_contact_wrench1 rfoot_contact_wrench1 ...
  rfoot_contact_wrench2 lfoot_contact_wrench3 rfoot_contact_wrench3 lhand_contact_wrench0 lhand_contact_wrench1 rhand_contact_wrench];

disturbance_pos = repmat(com0,1,nT)+bsxfun(@times,(0:nT-1),(com2-com0)/(nT-1));
Qw = eye(6);
fccdfkp_options = struct();
fccdfkp = FixedContactsFixedDisturbanceComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost1,q_nom,contact_wrench_struct,Qw,disturbance_pos,fccdfkp_options);
sccdfkp_sos_options = struct('num_fc_edges',num_ground_fc_edges,'l1_normalizer',5e3,'l2_normalizer',10);
sccdfkp_sos = SearchContactsFixedDisturbanceFullKinematicsSOSPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost2,q_nom,contact_wrench_struct,Qw,disturbance_pos,sccdfkp_sos_options);

% add feet contact constraint 
lfoot_above_ground = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);0.05*ones(1,4)],nan(3,4));
rfoot_above_ground = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);0.05*ones(1,4)],nan(3,4));
fccdfkp = fccdfkp.addConstraint(lfoot_above_ground,num2cell(lfoot_takeoff0:lfoot_land0-1));
fccdfkp = fccdfkp.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff0:rfoot_land0-1));
fccdfkp = fccdfkp.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff1:rfoot_land1-1));

sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_on_ground0,{1});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_on_ground0,{1});
lfoot_fixed = WorldFixedBodyPoseConstraint(robot,l_foot);
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_fixed,{1:lfoot_takeoff0-1});
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_fixed,{lfoot_land0:nT});
rfoot_fixed = WorldFixedBodyPoseConstraint(robot,r_foot);
if(rfoot_takeoff0>2)
  sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_fixed,{1:rfoot_takeoff0-1});
end
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_fixed,{rfoot_land0:rfoot_takeoff1-1});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_fixed,{rfoot_land1:nT});
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_on_ground1,{lfoot_land0});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_on_ground1,{rfoot_land0});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_on_ground2,{rfoot_land1});

% add hand contact position constraint
lhand_on_wall1 = WorldPositionConstraint(robot,l_hand,lhand_pt,[nan;l_wall_pos(2)-wall_size(2)/2;0.8],[nan;l_wall_pos(2)-wall_size(2)/2;nan]);
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_on_wall1,{1});
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_on_wall1,{lhand_land0});
rhand_on_wall1 = WorldPositionConstraint(robot,r_hand,rhand_pt,[nan;r_wall_pos(2)+wall_size(2)/2;0.8],[nan;r_wall_pos(2)+wall_size(2)/2;nan]);
sccdfkp_sos = sccdfkp_sos.addConstraint(rhand_on_wall1,{1});

% fix the foot orientation
lfoot_orient_cnstr0 = WorldQuatConstraint(robot,l_foot,lfoot_pos0(4:7),0);
fccdfkp = fccdfkp.addConstraint(lfoot_orient_cnstr0,num2cell(1:lfoot_takeoff0-1));
rfoot_orient_cnstr0 = WorldQuatConstraint(robot,r_foot,rfoot_pos0(4:7),0);
fccdfkp = fccdfkp.addConstraint(rfoot_orient_cnstr0,num2cell(1:rfoot_takeoff0-1));

rfoot_orient_cnstr1 = WorldQuatConstraint(robot,r_foot,rfoot_pos1(4:7),0);
fccdfkp = fccdfkp.addConstraint(rfoot_orient_cnstr1,num2cell(rfoot_land0:rfoot_takeoff1-1));

lfoot_orient_cnstr1 = WorldQuatConstraint(robot,l_foot,lfoot_pos1(4:7),0);
fccdfkp = fccdfkp.addConstraint(lfoot_orient_cnstr1,num2cell(lfoot_land0:nT));
rfoot_orient_cnstr2 = WorldQuatConstraint(robot,r_foot,rfoot_pos2(4:7),0);
fccdfkp = fccdfkp.addConstraint(rfoot_orient_cnstr2,num2cell(rfoot_land1:nT));

% dt bound
dt_lb = 0.2*ones(nT-1,1);
dt_ub = 0.3*ones(nT-1,1);
dt_bnd = BoundingBoxConstraint(dt_lb,dt_ub);
fccdfkp = fccdfkp.addConstraint(dt_bnd,fccdfkp.h_inds);
sccdfkp_sos = sccdfkp_sos.addConstraint(dt_bnd,sccdfkp_sos.h_inds);

% foot above ground
lfoot_above_ground = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);0.05*ones(1,4)],[nan(1,4);(l_wall_pos(2)-wall_size(2)/2-0.01)*ones(1,4);nan(1,4)]);
rfoot_above_ground = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(1,4);(r_wall_pos(2)+wall_size(2)/2+0.01)*ones(1,4);0.05*ones(1,4)],nan(3,4));
fccdfkp = fccdfkp.addConstraint(lfoot_above_ground,num2cell(lfoot_takeoff0:lfoot_land0-1));
fccdfkp = fccdfkp.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff0:rfoot_land0-1));
fccdfkp = fccdfkp.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff1:rfoot_land1-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_above_ground,num2cell(lfoot_takeoff0:lfoot_land0-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff0:rfoot_land0-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff1:rfoot_land1-1));
% hand off wall
lhand_off_wall = WorldPositionConstraint(robot,l_hand,lhand_pt,nan(3,1),[nan;l_wall_pos(2)-wall_size(2)/2-0.05;nan]);
rhand_off_wall = WorldPositionConstraint(robot,r_hand,rhand_pt,[nan;r_wall_pos(2)+wall_size(2)/2+0.05;nan],nan(3,1));
fccdfkp = fccdfkp.addConstraint(lhand_off_wall,num2cell(hand_off:nT));
fccdfkp = fccdfkp.addConstraint(rhand_off_wall,num2cell(hand_off:nT));
fccdfkp = fccdfkp.addConstraint(lhand_off_wall,num2cell(lhand_takeoff0:lhand_land0-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_off_wall,num2cell(hand_off:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(rhand_off_wall,num2cell(hand_off:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_off_wall,num2cell(lhand_takeoff0:lhand_land0-1));

% torso upright
utorso_upright_init = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],pi/9);
fccdfkp = fccdfkp.addConstraint(utorso_upright_init,{1});
sccdfkp_sos = sccdfkp_sos.addConstraint(utorso_upright_init,{1});
utorso_upright_end = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],pi/9);
fccdfkp = fccdfkp.addConstraint(utorso_upright_end,{nT});
sccdfkp_sos = sccdfkp_sos.addConstraint(utorso_upright_end,{nT});

% hand not penetrating into the wall
lhand_dir_cnstr = WorldGazeDirConstraint(robot,l_hand,lhand_dir,[0;1;0],pi/2.5);
rhand_dir_cnstr = WorldGazeDirConstraint(robot,r_hand,rhand_dir,[0;-1;0],pi/2.5);
fccdfkp = fccdfkp.addConstraint(lhand_dir_cnstr,num2cell(1:lhand_takeoff0-1));
fccdfkp = fccdfkp.addConstraint(lhand_dir_cnstr,num2cell(lhand_land0:hand_off-1));
fccdfkp = fccdfkp.addConstraint(rhand_dir_cnstr,num2cell(1:hand_off-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_dir_cnstr,num2cell(1:lhand_takeoff0-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_dir_cnstr,num2cell(lhand_land0:hand_off-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(rhand_dir_cnstr,num2cell(1:hand_off-1));

% ufarm off the wall
lufarm_pts = repmat([0.05;0;0.04],1,4).*[1 1 -1 -1;0 0 0 0;1 -1 1 -1];
rufarm_pts = repmat([0.05;0;0.04],1,4).*[1 1 -1 -1;0 0 0 0;1 -1 1 -1];
lufarm_cnstr = WorldPositionConstraint(robot,l_ufarm,lufarm_pts,nan(3,4),[nan(1,4);(l_wall_pos(2)-wall_size(2)/2-0.01)*ones(1,4);nan(1,4)]);
rufarm_cnstr = WorldPositionConstraint(robot,r_ufarm,rufarm_pts,[nan(1,4);(r_wall_pos(2)+wall_size(2)/2+0.01)*ones(1,4);nan(1,4)],nan(3,4));

fccdfkp = fccdfkp.addConstraint(lufarm_cnstr,num2cell(lhand_takeoff0:lhand_land0-1));
fccdfkp = fccdfkp.addConstraint(lufarm_cnstr,num2cell(hand_off-1:nT));
fccdfkp = fccdfkp.addConstraint(rufarm_cnstr,num2cell(hand_off-1:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(lufarm_cnstr,num2cell(lhand_takeoff0:lhand_land0-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(lufarm_cnstr,num2cell(hand_off-1:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(rufarm_cnstr,num2cell(hand_off-1:nT));

% arm_lwy should have zero velocity
fccdfkp = fccdfkp.addConstraint(ConstantConstraint(zeros(2*nT,1)),fccdfkp.v_inds([l_arm_lwy;r_arm_lwy],:));
sccdfkp_sos = sccdfkp_sos.addConstraint(ConstantConstraint(zeros(2*nT,1)),sccdfkp_sos.v_inds([l_arm_lwy;r_arm_lwy],:));

% knee not straight
kny_lb = BoundingBoxConstraint(0.5*ones(2*nT,1),inf(2*nT,1));
fccdfkp = fccdfkp.addConstraint(kny_lb,fccdfkp.q_inds([l_leg_kny;r_leg_kny],:));
sccdfkp_sos = sccdfkp_sos.addConstraint(kny_lb,sccdfkp_sos.q_inds([l_leg_kny;r_leg_kny],:));

% bend the knee for the initial posture
kny_lb_init = BoundingBoxConstraint(0.8*ones(2,1),inf(2,1));
fccdfkp = fccdfkp.addConstraint(kny_lb_init,fccdfkp.q_inds([l_leg_kny;r_leg_kny],1));
sccdfkp_sos = sccdfkp_sos.addConstraint(kny_lb_init,sccdfkp_sos.q_inds([l_leg_kny;r_leg_kny],1));

% % pelvis pass the trap
% pelvis_pos_cnstr = BoundingBoxConstraint(trap_pos(1)+trap_size(1)/2,inf);
% fccdfkp = fccdfkp.addConstraint(pelvis_pos_cnstr,fccdfkp.q_inds(1,nT));
% sccdfkp_sos = sccdfkp_sos.addConstraint(pelvis_pos_cnstr,sccdfkp_sos.q_inds(1,nT));

% final posture should be quasi static
qsc = QuasiStaticConstraint(robot);
qsc = qsc.addContact(l_foot,l_foot_contact_pts,r_foot,r_foot_contact_pts);
qsc = qsc.setActive(true);
fccdfkp = fccdfkp.addConstraint(qsc,{nT});
sccdfkp_sos = sccdfkp_sos.addConstraint(qsc,{nT});

% final velocity should be small
final_velocity_bnd = BoundingBoxConstraint([-0.1*ones(6,1);-0.2*ones(nv-6,1)],[0.1*ones(6,1);0.2*ones(nv-6,1)]);
fccdfkp = fccdfkp.addConstraint(final_velocity_bnd,fccdfkp.v_inds(:,nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(final_velocity_bnd,sccdfkp_sos.v_inds(:,nT));

% add cost on angular momentum
angular_momentum_cost = QuadraticSumConstraint(-inf,inf,eye(3),zeros(3,nT));
fccdfkp = fccdfkp.addCost(angular_momentum_cost,reshape(fccdfkp.centroidal_momentum_inds(1:3,:),[],1));
sccdfkp_sos = sccdfkp_sos.addCost(angular_momentum_cost,reshape(sccdfkp_sos.centroidal_momentum_inds(1:3,:),[],1));

% add lower bound on cws margin
sccdfkp_sos = sccdfkp_sos.addConstraint(BoundingBoxConstraint(70*ones(nT,1),inf(nT,1)),sccdfkp_sos.cws_margin_ind);

% initial velocity bounds
init_vel_bnd = BoundingBoxConstraint(-0.5*ones(nv,1),0.5*ones(nv,1));
fccdfkp = fccdfkp.addConstraint(init_vel_bnd,fccdfkp.v_inds(:,1));
sccdfkp_sos = sccdfkp_sos.addConstraint(init_vel_bnd,sccdfkp_sos.v_inds(:,1));

% not too much pelvis rotation in the final posture
pelvis_final_cost = QuadraticConstraint(-inf,inf,100*eye(3),zeros(3,1));
fccdfkp = fccdfkp.addCost(pelvis_final_cost,fccdfkp.q_inds(4:6,end));
sccdfkp_sos = sccdfkp_sos.addCost(pelvis_final_cost,sccdfkp_sos.q_inds(4:6,end));

% back z bounded in the final posture
back_z_bnd_final = BoundingBoxConstraint(-0.3,0.3);
fccdfkp = fccdfkp.addConstraint(back_z_bnd_final,fccdfkp.q_inds(back_bky,nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(back_z_bnd_final,sccdfkp_sos.q_inds(back_bkz,nT));

% back pitch cannot be execessive
back_bky_bnd = BoundingBoxConstraint(-0.4*ones(nT,1),0.4*ones(nT,1));
fccdfkp = fccdfkp.addConstraint(back_bky_bnd,fccdfkp.q_inds(back_bky,:));
sccdfkp_sos = sccdfkp_sos.addConstraint(back_bky_bnd,sccdfkp_sos.q_inds(back_bky,:));

% initial posture in quasi-static
qsc_init = QuasiStaticConstraint(robot);
qsc_init = qsc_init.addContact(l_foot,l_foot_contact_pts,r_foot,r_foot_contact_pts);
qsc_init = qsc_init.setActive(true);
fccdfkp = fccdfkp.addConstraint(qsc_init,{1});
sccdfkp_sos = sccdfkp_sos.addConstraint(qsc_init,{1});

% % Do not bend left shoulder backward too much
% l_arm_shz_bnd = BoundingBoxConstraint(-inf(nT-lhand_land0+1,1),0.65*ones(nT-lhand_land0+1,1));
% sccdfkp_sos = sccdfkp_sos.addConstraint(l_arm_shz_bnd,sccdfkp_sos.q_inds(l_arm_shz,lhand_land0:nT));

x_init = zeros(fccdfkp.num_vars,1);
x_init(fccdfkp.q_inds) = reshape(repmat(q0,1,nT),1,[]);

fccdfkp = fccdfkp.setSolverOptions('snopt','majoriterationslimit',300);
fccdfkp = fccdfkp.setSolverOptions('snopt','majoroptimalitytolerance',1e-4);
fccdfkp = fccdfkp.setSolverOptions('snopt','print','test_trap_fccdfkp3.out');
if(options.mode == 1)
  [x_sol,F,info] = fccdfkp.solve(x_init);
  sol = fccdfkp.retrieveSolution(x_sol);
  if(info<10)
    save('test_trap_fccdfkp3.mat','sol','x_sol');
    options.mode = 2;
  end
else
  load('test_trap_fccdfkp3.mat');
end
keyboard;

robot_mass = robot.getMass();
if(options.mode == 2)
  cws_margin_sol = zeros(nT,1);
  l0 = msspoly.zeros(nT,1);
  l1 = msspoly.zeros(nT,1);
  l2 = cell(nT,1);
  l3 = cell(nT,1);
  l4 = cell(nT,1);
  V = msspoly.zeros(nT,1);
  for i = 1:nT
    prog_lagrangian = FixedMotionSearchCWSmarginLinFC(num_ground_fc_edges,robot_mass,1,Qw,sol.num_fc_pts(i),sol.num_grasp_pts(i),sol.num_grasp_wrench_vert(i));
    prog_lagrangian.backoff_flag = 0.9;
    [cws_margin_sol(i),l0(i),l1(i),l2(i),l3{i},l4(i),V(i),solver_sol,info] = prog_lagrangian.findCWSmargin(0,sol.friction_cones(i),sol.grasp_pos(i),sol.grasp_wrench_vert(i),disturbance_pos(:,i),sol.momentum_dot(:,i),sol.com(:,i));
  end
  save('test_trap_lagrangian3.mat','cws_margin_sol','l0','l1','l2','l3','l4','V');
else
  load('test_trap_lagrangian3.mat');
end
keyboard

x_init = sccdfkp_sos.getInitialVars(sol.q,sol.v,sol.dt);
x_init = sccdfkp_sos.setMomentumDot(x_init,sol.momentum_dot);
x_init(sccdfkp_sos.cws_margin_ind) = cws_margin_sol;
for i = 1:nT
  if(~isempty(sccdfkp_sos.qsc_weight_inds{i}))
    x_init(sccdfkp_sos.qsc_weight_inds{i}) = sol.qsc_weights{i};
  end
end
x_init = sccdfkp_sos.setL0GramVarVal(x_init,l0);
x_init = sccdfkp_sos.setL1GramVarVal(x_init,l1);
x_init = sccdfkp_sos.setL2GramVarVal(x_init,l2);
x_init = sccdfkp_sos.setL3GramVarVal(x_init,l3);
x_init = sccdfkp_sos.setL4GramVarVal(x_init,l4);
V_clean = clean(V);
for i = 1:nT
  V_clean_tol = 1e-6;
  while(deg(V_clean(i))>2)
    V_clean_tol = V_clean_tol*10;
    V_clean(i) = clean(V_clean(i),V_clean_tol);
  end
end
x_init = sccdfkp_sos.setVGramVarVal(x_init,V_clean);

sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','print','test_trap_sccdfkp_sos3.out');
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','majoriterationslimit',200);
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','superbasicslimit',14000);
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','majoroptimalitytolerance',3e-4);

[x_sol,F,info] = sccdfkp_sos.solve(x_init);
sos_sol = sccdfkp_sos.retrieveSolution(x_sol);
keyboard;

% interpolate the trajectory
kinsol = cell(nT,1);
for i = 1:nT
  kinsol{i} = robot.doKinematics(sos_sol.q(:,i),sos_sol.v(:,i),struct('use_mex',false));
end
lfoot_pos0 = robot.forwardKin(kinsol{1},l_foot,[0;0;0],2);
rfoot_pos0 = robot.forwardKin(kinsol{1},r_foot,[0;0;0],2);
ltoe_pos0 = robot.forwardKin(kinsol{1},l_foot,l_toe,0);
rtoe_pos0 = robot.forwardKin(kinsol{1},r_foot,r_toe,0);
lfoot_pos1 = robot.forwardKin(kinsol{lfoot_land0},l_foot,[0;0;0],2);
rfoot_pos1 = robot.forwardKin(kinsol{rfoot_land0},r_foot,[0;0;0],2);
rtoe_pos1 = robot.forwardKin(kinsol{rfoot_land0},r_foot,r_toe,0);
rfoot_pos2 = robot.forwardKin(kinsol{rfoot_land1},r_foot,[0;0;0],2);
lhand_pos = robot.forwardKin(kinsol{1},l_hand,lhand_pt,0);
rhand_pos = robot.forwardKin(kinsol{1},r_hand,rhand_pt,0);
lfoot_cnstr0 = {WorldPositionConstraint(robot,l_foot,[0;0;0],lfoot_pos0(1:3),lfoot_pos0(1:3)),...
  WorldQuatConstraint(robot,l_foot,lfoot_pos0(4:7),0)};
rfoot_cnstr0 = {WorldPositionConstraint(robot,r_foot,[0;0;0],rfoot_pos0(1:3),rfoot_pos0(1:3)),...
  WorldQuatConstraint(robot,r_foot,rfoot_pos0(4:7),0)};
ltoe_cnstr0 = WorldPositionConstraint(robot,l_foot,l_toe,ltoe_pos0,ltoe_pos0);
rtoe_cnstr0 = WorldPositionConstraint(robot,r_foot,r_toe,rtoe_pos0,rtoe_pos0);
lfoot_cnstr1 = {WorldPositionConstraint(robot,l_foot,[0;0;0],lfoot_pos1(1:3),lfoot_pos1(1:3)),...
  WorldQuatConstraint(robot,l_foot,lfoot_pos1(4:7),0)};
rfoot_cnstr1 = {WorldPositionConstraint(robot,r_foot,[0;0;0],rfoot_pos1(1:3),rfoot_pos1(1:3)),...
  WorldQuatConstraint(robot,r_foot,rfoot_pos1(4:7),0)};
rtoe_cnstr1 = WorldPositionConstraint(robot,r_foot,r_toe,rtoe_pos1,rtoe_pos1);
rfoot_cnstr2 = {WorldPositionConstraint(robot,r_foot,[0;0;0],rfoot_pos2(1:3),rfoot_pos2(1:3));...
  WorldQuatConstraint(robot,r_foot,rfoot_pos2(4:7),0)};
lhand_cnstr = {WorldPositionConstraint(robot,l_hand,lhand_pt,lhand_pos,lhand_pos),lhand_dir_cnstr};
rhand_cnstr = {WorldPositionConstraint(robot,r_hand,rhand_pt,rhand_pos,rhand_pos),rhand_dir_cnstr};

lfoot_above_ground = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);0.01*ones(1,4)],[nan(1,4);(l_wall_pos(2)-wall_size(2)/2-0.01)*ones(1,4);nan(1,4)]);
rfoot_above_ground = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(1,4);(r_wall_pos(2)+wall_size(2)/2+0.01)*ones(1,4);0.01*ones(1,4)],nan(3,4));
lheel_above_ground = WorldPositionConstraint(robot,l_foot,l_heel,[nan(2,2);zeros(1,2)],nan(3,2));
rheel_above_ground = WorldPositionConstraint(robot,r_foot,r_heel,[nan(2,2);zeros(1,2)],nan(3,2));

lhand_off_wall = WorldPositionConstraint(robot,l_hand,lhand_pt,nan(3,1),[nan;l_wall_pos(2)-wall_size(2)/2;nan]);
rhand_off_wall = WorldPositionConstraint(robot,r_hand,rhand_pt,[nan;r_wall_pos(2)+wall_size(2)/2;nan],nan(3,1));

lufarm_cnstr = WorldPositionConstraint(robot,l_ufarm,lufarm_pts,nan(3,4),[nan(1,4);(l_wall_pos(2)-wall_size(2)/2)*ones(1,4);nan(1,4)]);
rufarm_cnstr = WorldPositionConstraint(robot,r_ufarm,rufarm_pts,[nan(1,4);(r_wall_pos(2)+wall_size(2)/2)*ones(1,4);nan(1,4)],nan(3,4));

t_knot = [0;cumsum(sos_sol.dt)];
qtraj_knot = PPTrajectory(foh(t_knot,sos_sol.q));

t1 = linspace(t_knot(1),t_knot(2),10);
t1 = t1(2:end-1);
q1 = zeros(nq,length(t1));
for i = 1:length(t1)
  q_nom = qtraj_knot.eval(t1(i));
  ik = InverseKinematics(robot,q_nom,lfoot_cnstr0{:},rtoe_cnstr0,rheel_above_ground,lhand_cnstr{:},rhand_cnstr{:});
  [q1(:,i),~,info] = ik.solve(q_nom);
end
v1 = [diff(q1,1,2)./repmat(diff(t1),nq,1) sos_sol.v(:,2)];

t2 = linspace(t_knot(2),t_knot(3),10);
t2 = t2(2:end-1);
q2 = zeros(nq,length(t2));
for i = 1:length(t2)
  q_nom = qtraj_knot.eval(t2(i));
  ik = InverseKinematics(robot,q_nom,lfoot_cnstr0{:},rfoot_above_ground,lhand_cnstr{:},rhand_cnstr{:});
  [q2(:,i),~,info] = ik.solve(q_nom);
end
v2 = [diff(q2,1,2)./repmat(diff(t2),nq,1) sos_sol.v(:,3)];

t3 = linspace(t_knot(3),t_knot(4),10);
t3 = t3(2:end-1);
q3 = zeros(nq,length(t3));
for i = 1:length(t3)
  q_nom = qtraj_knot.eval(t3(i));
  ik = InverseKinematics(robot,q_nom,lfoot_cnstr0{:},rfoot_above_ground,lhand_cnstr{:},rhand_cnstr{:});
  [q3(:,i),~,info] = ik.solve(q_nom);
end
v3 = [diff(q3,1,2)./repmat(diff(t3),nq,1) sos_sol.v(:,4)];

t4 = linspace(t_knot(4),t_knot(5),10);
t4 = t4(2:end-1);
q4 = zeros(nq,length(t4));
for i = 1:length(t4)
  q_nom = qtraj_knot.eval(t4(i));
  ik = InverseKinematics(robot,q_nom,ltoe_cnstr0,lheel_above_ground,rfoot_cnstr1{:},lhand_cnstr{:},rhand_cnstr{:});
  [q4(:,i),~,info] = ik.solve(q_nom);
end
v4 = [diff(q4,1,2)./repmat(diff(t4),nq,1) sos_sol.v(:,5)];

t5 = linspace(t_knot(5),t_knot(6),10);
t5 = t5(2:end-1);
q5 = zeros(nq,length(t5));
for i = 1:length(t5)
  q_nom = qtraj_knot.eval(t5(i));
  ik = InverseKinematics(robot,q_nom,lfoot_above_ground,rfoot_cnstr1{:},lhand_cnstr{:},rhand_cnstr{:});
  [q5(:,i),~,info] = ik.solve(q_nom);
end
v5 = [diff(q5,1,2)./repmat(diff(t5),nq,1) sos_sol.v(:,6)];

t6 = linspace(t_knot(6),t_knot(7),10);
t6 = t6(2:end-1);
q6 = zeros(nq,length(t6));
for i = 1:length(t6)
  q_nom = qtraj_knot.eval(t6(i));
  ik = InverseKinematics(robot,q_nom,lfoot_above_ground,rfoot_cnstr1{:},lhand_cnstr{:},rhand_cnstr{:});
  [q6(:,i),~,info] = ik.solve(q_nom);
end
v6 = [diff(q6,1,2)./repmat(diff(t6),nq,1) sos_sol.v(:,7)];

t7 = linspace(t_knot(7),t_knot(8),10);
t7 = t7(2:end-1);
q7 = zeros(nq,length(t7));
for i = 1:length(t7)
  q_nom = qtraj_knot.eval(t7(i));
  ik = InverseKinematics(robot,q_nom,lfoot_cnstr1{:},rheel_above_ground,rtoe_cnstr1,lhand_cnstr{:},rhand_cnstr{:});
  [q7(:,i),~,info] = ik.solve(q_nom);
end
v7 = [diff(q7,1,2)./repmat(diff(t7),nq,1) sos_sol.v(:,8)];

t8 = linspace(t_knot(8),t_knot(9),10);
t8 = t8(2:end-1);
q8 = zeros(nq,length(t8));
for i = 1:length(t8)
  q_nom = qtraj_knot.eval(t8(i));
  ik = InverseKinematics(robot,q_nom,lfoot_cnstr1{:},rfoot_above_ground,lhand_cnstr{:},rhand_cnstr{:});
  [q8(:,i),~,info] = ik.solve(q_nom);
end
v8 = [diff(q8,1,2)./repmat(diff(t8),nq,1) sos_sol.v(:,9)];

t9 = linspace(t_knot(9),t_knot(10),10);
t9 = t9(2:end-1);
q9 = zeros(nq,length(t9));
for i = 1:length(t9)
  q_nom = qtraj_knot.eval(t9(i));
  ik = InverseKinematics(robot,q_nom,lfoot_cnstr1{:},rfoot_above_ground,lhand_cnstr{:},rhand_cnstr{:});
  [q9(:,i),~,info] = ik.solve(q_nom);
end
v9 = [diff(q9,1,2)./repmat(diff(t9),nq,1) sos_sol.v(:,10)];

t10 = linspace(t_knot(10),t_knot(11),10);
t10 = t10(2:end-1);
q10 = zeros(nq,length(t10));
for i = 1:length(t10)
  q_nom = qtraj_knot.eval(t10(i));
  ik = InverseKinematics(robot,q_nom,lfoot_cnstr1{:},rfoot_cnstr2{:},lhand_off_wall,rhand_off_wall,lufarm_cnstr,rufarm_cnstr);
  [q10(:,i),~,info] = ik.solve(q_nom);
end
v10 = [diff(q10,1,2)./repmat(diff(t10),nq,1) sos_sol.v(:,11)];

t_all = [t_knot(1) t1 t_knot(2) t2 t_knot(3) t3 t_knot(4) t4 t_knot(5) t5 t_knot(6) t6 t_knot(7) t7 t_knot(8) t8 t_knot(9) t9 t_knot(10) t10 t_knot(11)];
q_all = [sos_sol.q(:,1) q1 sos_sol.q(:,2) q2 sos_sol.q(:,3) q3 sos_sol.q(:,4) q4 sos_sol.q(:,5) q5 sos_sol.q(:,6) q6 sos_sol.q(:,7) q7 sos_sol.q(:,8) q8 sos_sol.q(:,9) q9 sos_sol.q(:,10) q10 sos_sol.q(:,11)];
v_all = [sos_sol.v(:,1) v1 sos_sol.v(:,2) v2 sos_sol.v(:,3) v3 sos_sol.v(:,4) v4 sos_sol.v(:,5) v5 sos_sol.v(:,6) v6 sos_sol.v(:,7) v7 sos_sol.v(:,8) v8 sos_sol.v(:,9) v9 sos_sol.v(:,10) v10 sos_sol.v(:,11)];

xtraj = PPTrajectory(foh(t_all,[q_all;v_all]));
xtraj = xtraj.setOutputFrame(robot.getStateFrame());
end