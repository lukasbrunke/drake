function testAtlasTrapStepping(options)
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
r_toe = robot.getBody(r_foot).getTerrainContactPoints('toe');
r_heel = robot.getBody(r_foot).getTerrainContactPoints('heel');
r_foot_contact_pts = [r_toe r_heel];
r_foot_outer_edge = [r_toe(:,2) r_heel(:,2)];
r_foot_center = mean(r_foot_contact_pts,2);
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
com0 = robot.getCOM(kinsol0);

% find the end posture
lfoot_on_ground2 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[(trap_pos(1)+trap_size(1)/2+0.03)*ones(1,4);zeros(1,4)+0.05;zeros(1,4)],[nan(2,4);zeros(1,4)]);
rfoot_on_ground2 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[(trap_pos(1)+trap_size(1)/2+0.03)*ones(1,4);nan(1,4);zeros(1,4)],[nan(1,4);zeros(1,4)-0.05;zeros(1,4)]);
lhand_pos_cnstr = WorldPositionConstraint(robot,l_hand,lhand_pt,lhand_pos0(1:3),lhand_pos0(1:3));
rhand_pos_cnstr = WorldPositionConstraint(robot,r_hand,rhand_pt,rhand_pos0(1:3),rhand_pos0(1:3));
utorso_upright2 = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],pi/10);
ik = InverseKinematics(robot,qstar,lfoot_on_ground2,rfoot_on_ground2,lhand_pos_cnstr,rhand_pos_cnstr,utorso_upright2);
ik = ik.addConstraint(BoundingBoxConstraint(trap_pos(1)+trap_size(1)/2+0.05,inf),ik.q_idx(1));
[q2,F,info] = ik.solve(q0);
kinsol2 = robot.doKinematics(q2);
lfoot_pos2 = robot.forwardKin(kinsol2,l_foot,l_foot_center,2);
rfoot_pos2 = robot.forwardKin(kinsol2,r_foot,r_foot_center,2);
lfoot_outer_pos2 = robot.forwardKin(kinsol2,l_foot,l_foot_outer_edge,0);
rfoot_outer_pos2 = robot.forwardKin(kinsol2,r_foot,r_foot_outer_edge,0);
com2 = robot.getCOM(kinsol2);

lfoot_on_ground1 = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,repmat([(trap_pos(1)-trap_size(1)/2);(trap_pos(2)+trap_size(2)/2+0.01);0],1,4),repmat([(trap_pos(1)+trap_size(1)/2);(l_wall_pos(2)-wall_size(2)/2);0],1,4));
rfoot_on_ground1 = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,repmat([(trap_pos(1)-trap_size(1)/2);(r_wall_pos(2)+wall_size(2)/2);0],1,4),repmat([(trap_pos(1)+trap_size(1)/2);(trap_pos(2)-trap_size(2)/2-0.03);0],1,4));
utorso_upright1 = WorldGazeDirConstraint(robot,utorso,[0;0;1],[0;0;1],pi/10);
ik = InverseKinematics(robot,qstar,lfoot_on_ground1,rfoot_on_ground1,lhand_pos_cnstr,rhand_pos_cnstr,utorso_upright1);
[q1,F,info] = ik.solve(q0);
kinsol1 = robot.doKinematics(q1);
lfoot_pos1 = robot.forwardKin(kinsol1,l_foot,l_foot_center,2);
rfoot_pos1 = robot.forwardKin(kinsol1,r_foot,r_foot_center,2);

num_ground_fc_edges = 3;
num_wall_fc_edges = 3;
ground_mu = 2;
wall_mu = 2;
ground_fc_theta = linspace(0,2*pi,num_ground_fc_edges+1);
ground_fc_theta = ground_fc_theta(1:end-1);
wall_fc_theta = linspace(0,2*pi,num_wall_fc_edges+1);
wall_fc_theta = wall_fc_theta(1:end-1);
ground_fc = [cos(ground_fc_theta)*ground_mu;sin(ground_fc_theta)*ground_mu;ones(1,num_ground_fc_edges)];
lwall_fc = rotateVectorToAlign([0;0;1],[0;-1;0])*[cos(wall_fc_theta)*wall_mu;sin(wall_fc_theta)*wall_mu;ones(1,num_wall_fc_edges)];
rwall_fc = rotateVectorToAlign([0;0;1],[0;1;0])*[cos(wall_fc_theta)*wall_mu;sin(wall_fc_theta)*wall_mu;ones(1,num_wall_fc_edges)];
lhand_force_max = 400;
rhand_force_max = 400;

rfoot_takeoff0 = 2;
rfoot_land0 = 3;
lfoot_takeoff0 = 4;
lfoot_land0 = 5;
rfoot_takeoff1 = 6;
rfoot_land1 = 7;
lfoot_takeoff1 = 8;
lfoot_land1 = 9;
hand_off = 10;
nT = 10;

lfoot_cw = LinearFrictionConeWrench(robot,l_foot,l_foot_center,ground_fc);
rfoot_cw = LinearFrictionConeWrench(robot,r_foot,r_foot_center,ground_fc);
lhand_cw = GraspWrenchPolytope(robot,l_hand,lhand_pt,lhand_force_max*[zeros(6,1) [lwall_fc;zeros(3,num_wall_fc_edges)]]);
rhand_cw = GraspWrenchPolytope(robot,r_hand,rhand_pt,rhand_force_max*[zeros(6,1) [rwall_fc;zeros(3,num_wall_fc_edges)]]);
lfoot_cw2 = LinearFrictionConeWrench(robot,l_foot,l_foot_outer_edge,ground_fc);
rfoot_cw2 = LinearFrictionConeWrench(robot,r_foot,r_foot_outer_edge,ground_fc);

lfoot_contact_wrench0 = struct('active_knot',1:lfoot_takeoff0-1,'cw',lfoot_cw,'contact_pos',lfoot_pos0(1:3));
rfoot_contact_wrench0 = struct('active_knot',1:rfoot_takeoff0-1,'cw',rfoot_cw,'contact_pos',rfoot_pos0(1:3));
lfoot_contact_wrench1 = struct('active_knot',lfoot_land0:lfoot_takeoff1-1,'cw',lfoot_cw,'contact_pos',lfoot_pos1(1:3));
rfoot_contact_wrench1 = struct('active_knot',rfoot_land0:rfoot_takeoff1-1,'cw',rfoot_cw,'contact_pos',rfoot_pos1(1:3));
lfoot_contact_wrench2 = struct('active_knot',lfoot_land1:hand_off-1,'cw',lfoot_cw,'contact_pos',lfoot_pos2(1:3));
rfoot_contact_wrench2 = struct('active_knot',rfoot_land1:hand_off-1,'cw',rfoot_cw,'contact_pos',rfoot_pos2(1:3));
lfoot_contact_wrench3 = struct('active_knot',hand_off:nT,'cw',lfoot_cw2,'contact_pos',lfoot_outer_pos2);
rfoot_contact_wrench3 = struct('active_knot',hand_off:nT,'cw',rfoot_cw2,'contact_pos',rfoot_outer_pos2);
lhand_contact_wrench = struct('active_knot',1:hand_off-1,'cw',lhand_cw,'contact_pos',lhand_pos0(1:3));
rhand_contact_wrench = struct('active_knot',1:hand_off-1,'cw',rhand_cw,'contact_pos',rhand_pos0(1:3));


Q_comddot = eye(3);
Qv = 0.1*ones(nv,1);
Qv = diag(Qv);
Q = eye(nq);
Q(1,1) = 0;
Q(2,2) = 0;
Q(5,5) = 10;
Q(6,6) = 10;
T_lb = 2;
T_ub = 3;
tf_range = [T_lb T_ub];

cws_margin_cost = 10;
q_nom = repmat(q0,1,nT);
contact_wrench_struct = [lfoot_contact_wrench0 rfoot_contact_wrench0 lfoot_contact_wrench1 rfoot_contact_wrench1 lfoot_contact_wrench2 rfoot_contact_wrench2 lfoot_contact_wrench3 rfoot_contact_wrench3 lhand_contact_wrench rhand_contact_wrench];

disturbance_pos = repmat(com0,1,nT)+bsxfun(@times,(0:nT-1),(com2-com0)/(nT-1));
Qw = eye(6);
fccdfkp_options = struct();
fccdfkp = FixedContactsFixedDisturbanceComDynamicsFullKinematicsPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos,fccdfkp_options);
sccdfkp_sos_options = struct('num_fc_edges',num_ground_fc_edges,'l1_normalizer',1e3);
sccdfkp_sos = SearchContactsFixedDisturbanceFullKinematicsSOSPlanner(robot,nT,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos,sccdfkp_sos_options);

% add feet contact constraint 
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_on_ground0,{1});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_on_ground0,{1});
lfoot_fixed = WorldFixedBodyPoseConstraint(robot,l_foot);
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_fixed,{1:lfoot_takeoff0-1});
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_fixed,{lfoot_land0:lfoot_takeoff1-1});
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_fixed,{lfoot_land1:nT});
rfoot_fixed = WorldFixedBodyPoseConstraint(robot,r_foot);
if(rfoot_takeoff0>2)
  sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_fixed,{1:rfoot_takeoff0-1});
end
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_fixed,{rfoot_land0:rfoot_takeoff1-1});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_fixed,{rfoot_land1:nT});
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_on_ground1,{lfoot_land0});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_on_ground1,{rfoot_land0});
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_on_ground2,{lfoot_land1});
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_on_ground2,{rfoot_land1});

% add hand contact position constraint
lhand_on_wall1 = WorldPositionConstraint(robot,l_hand,lhand_pt,[trap_pos(1)-trap_size(1)/2;l_wall_pos(2)-wall_size(2)/2;0.8],[l_wall_pos(1)+wall_size(1)/2;l_wall_pos(2)-wall_size(2)/2;nan]);
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_on_wall1,{1});
rhand_on_wall1 = WorldPositionConstraint(robot,r_hand,rhand_pt,[trap_pos(1)-trap_size(1)/2;r_wall_pos(2)+wall_size(2)/2;0.8],[r_wall_pos(1)+wall_size(1)/2;r_wall_pos(2)+wall_size(2)/2;nan]);
sccdfkp_sos = sccdfkp_sos.addConstraint(rhand_on_wall1,{1});

% fix the foot orientation
lfoot_orient_cnstr0 = WorldQuatConstraint(robot,l_foot,lfoot_pos0(4:7),0);
fccdfkp = fccdfkp.addConstraint(lfoot_orient_cnstr0,num2cell(1:lfoot_takeoff0-1));
rfoot_orient_cnstr0 = WorldQuatConstraint(robot,r_foot,rfoot_pos0(4:7),0);
fccdfkp = fccdfkp.addConstraint(rfoot_orient_cnstr0,num2cell(1:rfoot_takeoff0-1));

lfoot_orient_cnstr1 = WorldQuatConstraint(robot,l_foot,lfoot_pos1(4:7),0);
fccdfkp = fccdfkp.addConstraint(lfoot_orient_cnstr1,num2cell(lfoot_land0:lfoot_takeoff1-1));
rfoot_orient_cnstr1 = WorldQuatConstraint(robot,r_foot,rfoot_pos1(4:7),0);
fccdfkp = fccdfkp.addConstraint(rfoot_orient_cnstr1,num2cell(rfoot_land0:rfoot_takeoff1-1));

lfoot_orient_cnstr1 = WorldQuatConstraint(robot,l_foot,lfoot_pos2(4:7),0);
fccdfkp = fccdfkp.addConstraint(lfoot_orient_cnstr1,num2cell(lfoot_land1:nT));
rfoot_orient_cnstr1 = WorldQuatConstraint(robot,r_foot,rfoot_pos2(4:7),0);
fccdfkp = fccdfkp.addConstraint(rfoot_orient_cnstr1,num2cell(rfoot_land1:nT));

% dt bound
dt_bnd = BoundingBoxConstraint(0.2*ones(nT-1,1),0.3*ones(nT-1,1));
fccdfkp = fccdfkp.addConstraint(dt_bnd,fccdfkp.h_inds);
sccdfkp_sos = sccdfkp_sos.addConstraint(dt_bnd,sccdfkp_sos.h_inds);

% foot above ground
lfoot_above_ground = WorldPositionConstraint(robot,l_foot,l_foot_contact_pts,[nan(2,4);0.03*ones(1,4)],nan(3,4));
rfoot_above_ground = WorldPositionConstraint(robot,r_foot,r_foot_contact_pts,[nan(2,4);0.03*ones(1,4)],nan(3,4));
fccdfkp = fccdfkp.addConstraint(lfoot_above_ground,num2cell(lfoot_takeoff0:lfoot_land0-1));
fccdfkp = fccdfkp.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff0:rfoot_land0-1));
fccdfkp = fccdfkp.addConstraint(lfoot_above_ground,num2cell(lfoot_takeoff1:lfoot_land1-1));
fccdfkp = fccdfkp.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff1:rfoot_land1-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_above_ground,num2cell(lfoot_takeoff0:lfoot_land0-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff0:rfoot_land0-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(lfoot_above_ground,num2cell(lfoot_takeoff1:lfoot_land1-1));
sccdfkp_sos = sccdfkp_sos.addConstraint(rfoot_above_ground,num2cell(rfoot_takeoff1:rfoot_land1-1));
% hand off wall
lhand_off_wall = WorldPositionConstraint(robot,l_hand,lhand_pt,nan(3,1),[nan;l_wall_pos(2)-wall_size(2)/2-0.05;nan]);
rhand_off_wall = WorldPositionConstraint(robot,r_hand,rhand_pt,[nan;r_wall_pos(2)+wall_size(2)/2+0.05;nan],nan(3,1));
fccdfkp = fccdfkp.addConstraint(lhand_off_wall,num2cell(hand_off:nT));
fccdfkp = fccdfkp.addConstraint(rhand_off_wall,num2cell(hand_off:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(lhand_off_wall,num2cell(hand_off:nT));
sccdfkp_sos = sccdfkp_sos.addConstraint(rhand_off_wall,num2cell(hand_off:nT));

x_init = zeros(fccdfkp.num_vars,1);
x_init(fccdfkp.q_inds) = reshape(repmat(q0,1,nT),1,[]);

fccdfkp = fccdfkp.setSolverOptions('snopt','majoriterationslimit',200);
fccdfkp = fccdfkp.setSolverOptions('snopt','majoroptimalitytolerance',1e-4);
fccdfkp = fccdfkp.setSolverOptions('snopt','print','test_trap_fccdfkp.out');

if(options.mode == 1)
  [x_sol,F,info] = fccdfkp.solve(x_init);
  sol = fccdfkp.retrieveSolution(x_sol);
  if(info<10)
    save('test_trap_fccdfkp.mat','sol','x_sol');
  end
else
  load('test_trap_fccdfkp.mat');
end
keyboard;

robot_mass = robot.getMass();
prog_lagrangian = FixedMotionSearchCWSmarginLinFC(num_ground_fc_edges,robot_mass,nT,Qw,sol.num_fc_pts,sol.num_grasp_pts,sol.num_grasp_wrench_vert);
if(options.mode == 2)
  [cws_margin_sol,l0,l1,l2,l3,l4,V,solver_sol,info] = prog_lagrangian.findCWSmargin(0,sol.friction_cones,sol.grasp_pos,sol.grasp_wrench_vert,disturbance_pos,sol.momentum_dot,sol.com);
  save('test_trap_lagrangian.mat','cws_margin_sol','l0','l1','l2','l3','l4','V');
else
  load('test_trap_lagrangian.mat');
end
keyboard

x_init = sccdfkp_sos.getInitialVars(sol.q,sol.v,sol.dt);
x_init = sccdfkp_sos.setMomentumDot(x_init,sol.momentum_dot);
x_init(sccdfkp_sos.cws_margin_ind) = cws_margin_sol;
x_init = sccdfkp_sos.setL0GramVarVal(x_init,l0);
x_init = sccdfkp_sos.setL1GramVarVal(x_init,l1);
x_init = sccdfkp_sos.setL2GramVarVal(x_init,l2);
x_init = sccdfkp_sos.setL3GramVarVal(x_init,l3);
x_init = sccdfkp_sos.setL4GramVarVal(x_init,l4);
x_init = sccdfkp_sos.setVGramVarVal(x_init,clean(V,1e-4));

sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','print','test_trap_sccdfkp_sos.out');
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','majoriterationslimit',300);
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','superbasicslimit',10000);
sccdfkp_sos = sccdfkp_sos.setSolverOptions('snopt','majoroptimalitytolerance',3e-4);

[x_sol,F,info] = sccdfkp_sos.solve(x_init);
sos_sol = sccdfkp_sos.retrieveSolution(x_sol);
keyboard;
end