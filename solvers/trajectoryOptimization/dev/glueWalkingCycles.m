function glueWalkingCycles(num_steps)
warning('off','Drake:RigidBody:SimplifiedCollisionGeometry');
warning('off','Drake:RigidBody:NonPositiveInertiaMatrix');
warning('off','Drake:RigidBodyManipulator:UnsupportedContactPoints');
warning('off','Drake:RigidBodyManipulator:UnsupportedJointLimits');
warning('off','Drake:RigidBodyManipulator:UnsupportedVelocityLimits');
urdf = [getDrakePath,'/examples/Atlas/urdf/atlas_minimal_contact.urdf'];
options.floating = true;
robot = RigidBodyManipulator(urdf,options);
halfcycle = load('test_halfcycle_walk2','-mat','xtraj_sol','com_sol','comdot_sol','comddot_sol');
t_halfcycle = halfcycle.xtraj_sol.getBreaks();
x_halfcycle = halfcycle.xtraj_sol.eval(t_halfcycle);
[x_cycle,t_cycle,com_cycle,comdot_cycle,comddot_cycle] = getFullCycle(robot,x_halfcycle,t_halfcycle,halfcycle.com_sol,halfcycle.comdot_sol,halfcycle.comddot_sol);
nT_steps = 1+(length(t_cycle)-1)*num_steps;
x_steps = zeros(size(x_cycle,1),nT_steps);
t_steps = zeros(1,nT_steps);
com_steps = zeros(3,nT_steps);
comdot_steps = zeros(3,nT_steps);
comddot_steps = zeros(3,nT_steps);
t_count = length(t_cycle);
x_steps(:,1:t_count) = x_cycle;
t_steps(:,1:t_count) = t_cycle;
com_steps(:,1:t_count) = com_cycle;
comdot_steps(:,1:t_count) = comdot_cycle;
comddot_steps(:,1:t_count) = comddot_cycle;
for i = 2:num_steps
  x_cycle_tmp = x_cycle(:,2:end);
  x_cycle_tmp(1,:) = x_cycle_tmp(1,:)+x_steps(1,t_count);
  x_steps(:,t_count+(1:length(t_cycle)-1)) = x_cycle_tmp;
  t_cycle_tmp = t_cycle(2:end)+t_steps(t_count);
  t_steps(:,t_count+(1:length(t_cycle)-1)) = t_cycle_tmp;
  com_cycle_tmp = com_cycle(:,2:end);
  com_cycle_tmp(1,:) = com_cycle_tmp(1,:)+com_steps(1,t_count);
  com_steps(:,t_count+(1:length(t_cycle)-1)) = com_cycle_tmp;
  comdot_steps(:,t_count+(1:length(t_cycle)-1)) = comdot_cycle(:,2:end);
  comddot_steps(:,t_count+(1:length(t_cycle)-1)) = comddot_cycle(:,2:end);
  t_count = t_count+length(t_cycle)-1;
end
xtraj = PPTrajectory(foh(t_steps,x_steps));
xtraj = xtraj.setOutputFrame(robot.getStateFrame);
v = robot.constructVisualizer();
v.playback(xtraj,struct('slider',true));
end

function [x_cycle,t_cycle,com_cycle,comdot_cycle,comddot_cycle] = getFullCycle(robot,x_halfcycle,t_halfcycle,com_halfcycle,comdot_halfcycle,comddot_halfcycle)
% take the first half cycle, flip it to get the second half cycle, and glue
% them together
nq = robot.getNumPositions();
nv = robot.getNumVelocities();
nT = length(t_halfcycle);
q_halfcycle = x_halfcycle(1:nq,:);
v_halfcycle = x_halfcycle(nq+(1:nv),:);
q_halfcycle2 = zeros(nq,nT-1);
v_halfcycle2 = zeros(nv,nT-1);
t_halfcycle2 = t_halfcycle(2:end)+t_halfcycle(end);
q_halfcycle2(1,:) = q_halfcycle(1,2:end)+x_halfcycle(1,end);
v_halfcycle2(1,:) = v_halfcycle(1,2:end);
q_halfcycle2(2,:) = -q_halfcycle(2,2:end);
v_halfcycle2(2,:) = -v_halfcycle(2,2:end);
q_halfcycle2(3,:) = q_halfcycle(3,2:end);
v_halfcycle2(3,:) = v_halfcycle(3,2:end);
q_halfcycle2(4,:) = -q_halfcycle(4,2:end);
v_halfcycle2(4,:) = -v_halfcycle(4,2:end);
q_halfcycle2(5,:) = q_halfcycle(5,2:end);
v_halfcycle2(5,:) = v_halfcycle(5,2:end);
q_halfcycle2(6,:) = -q_halfcycle(6,2:end);
v_halfcycle2(6,:) = -v_halfcycle(6,2:end);
back_bkz_position = robot.getBody(robot.findJointInd('back_bkz')).position_num;
back_bkz_velocity = robot.getBody(robot.findJointInd('back_bkz')).velocity_num;
q_halfcycle2(back_bkz_position,:) = -q_halfcycle(back_bkz_position,2:end);
v_halfcycle2(back_bkz_velocity,:) = -v_halfcycle(back_bkz_velocity,2:end);
back_bky_position = robot.getBody(robot.findJointInd('back_bky')).position_num;
back_bky_velocity = robot.getBody(robot.findJointInd('back_bky')).velocity_num;
q_halfcycle2(back_bky_position,:) = q_halfcycle(back_bky_position,2:end);
v_halfcycle2(back_bky_velocity,:) = v_halfcycle(back_bky_velocity,2:end);
back_bkx_position = robot.getBody(robot.findJointInd('back_bkx')).position_num;
back_bkx_velocity = robot.getBody(robot.findJointInd('back_bkx')).velocity_num;
q_halfcycle2(back_bkx_position,:) = -q_halfcycle(back_bkx_position,2:end);
v_halfcycle2(back_bkx_velocity,:) = -v_halfcycle(back_bkx_velocity,2:end);

l_arm_usy_position = robot.getBody(robot.findJointInd('l_arm_usy')).position_num;
r_arm_usy_position = robot.getBody(robot.findJointInd('r_arm_usy')).position_num;
l_arm_usy_velocity = robot.getBody(robot.findJointInd('l_arm_usy')).velocity_num;
r_arm_usy_velocity = robot.getBody(robot.findJointInd('r_arm_usy')).velocity_num;
q_halfcycle2(l_arm_usy_position,:) = q_halfcycle(r_arm_usy_position,2:end);
v_halfcycle2(l_arm_usy_velocity,:) = v_halfcycle(r_arm_usy_velocity,2:end);
q_halfcycle2(r_arm_usy_position,:) = q_halfcycle(l_arm_usy_position,2:end);
v_halfcycle2(r_arm_usy_velocity,:) = v_halfcycle(l_arm_usy_velocity,2:end);

l_arm_shx_position = robot.getBody(robot.findJointInd('l_arm_shx')).position_num;
r_arm_shx_position = robot.getBody(robot.findJointInd('r_arm_shx')).position_num;
l_arm_shx_velocity = robot.getBody(robot.findJointInd('l_arm_shx')).velocity_num;
r_arm_shx_velocity = robot.getBody(robot.findJointInd('r_arm_shx')).velocity_num;
q_halfcycle2(l_arm_shx_position,:) = -q_halfcycle(r_arm_shx_position,2:end);
v_halfcycle2(l_arm_shx_velocity,:) = -v_halfcycle(r_arm_shx_velocity,2:end);
q_halfcycle2(r_arm_shx_position,:) = -q_halfcycle(l_arm_shx_position,2:end);
v_halfcycle2(r_arm_shx_velocity,:) = -v_halfcycle(l_arm_shx_velocity,2:end);

l_arm_ely_position = robot.getBody(robot.findJointInd('l_arm_ely')).position_num;
r_arm_ely_position = robot.getBody(robot.findJointInd('r_arm_ely')).position_num;
l_arm_ely_velocity = robot.getBody(robot.findJointInd('l_arm_ely')).velocity_num;
r_arm_ely_velocity = robot.getBody(robot.findJointInd('r_arm_ely')).velocity_num;
q_halfcycle2(l_arm_ely_position,:) = q_halfcycle(r_arm_ely_position,2:end);
v_halfcycle2(l_arm_ely_velocity,:) = v_halfcycle(r_arm_ely_velocity,2:end);
q_halfcycle2(r_arm_ely_position,:) = q_halfcycle(l_arm_ely_position,2:end);
v_halfcycle2(r_arm_ely_velocity,:) = v_halfcycle(l_arm_ely_velocity,2:end);

l_arm_elx_position = robot.getBody(robot.findJointInd('l_arm_elx')).position_num;
r_arm_elx_position = robot.getBody(robot.findJointInd('r_arm_elx')).position_num;
l_arm_elx_velocity = robot.getBody(robot.findJointInd('l_arm_elx')).velocity_num;
r_arm_elx_velocity = robot.getBody(robot.findJointInd('r_arm_elx')).velocity_num;
q_halfcycle2(l_arm_elx_position,:) = -q_halfcycle(r_arm_elx_position,2:end);
v_halfcycle2(l_arm_elx_velocity,:) = -v_halfcycle(r_arm_elx_velocity,2:end);
q_halfcycle2(r_arm_elx_position,:) = -q_halfcycle(l_arm_elx_position,2:end);
v_halfcycle2(r_arm_elx_velocity,:) = -v_halfcycle(l_arm_elx_velocity,2:end);

l_arm_uwy_position = robot.getBody(robot.findJointInd('l_arm_uwy')).position_num;
r_arm_uwy_position = robot.getBody(robot.findJointInd('r_arm_uwy')).position_num;
l_arm_uwy_velocity = robot.getBody(robot.findJointInd('l_arm_uwy')).velocity_num;
r_arm_uwy_velocity = robot.getBody(robot.findJointInd('r_arm_uwy')).velocity_num;
q_halfcycle2(l_arm_uwy_position,:) = q_halfcycle(r_arm_uwy_position,2:end);
v_halfcycle2(l_arm_uwy_velocity,:) = v_halfcycle(r_arm_uwy_velocity,2:end);
q_halfcycle2(r_arm_uwy_position,:) = q_halfcycle(l_arm_uwy_position,2:end);
v_halfcycle2(r_arm_uwy_velocity,:) = v_halfcycle(l_arm_uwy_velocity,2:end);

l_arm_mwx_position = robot.getBody(robot.findJointInd('l_arm_mwx')).position_num;
r_arm_mwx_position = robot.getBody(robot.findJointInd('r_arm_mwx')).position_num;
l_arm_mwx_velocity = robot.getBody(robot.findJointInd('l_arm_mwx')).velocity_num;
r_arm_mwx_velocity = robot.getBody(robot.findJointInd('r_arm_mwx')).velocity_num;
q_halfcycle2(l_arm_mwx_position,:) = -q_halfcycle(r_arm_mwx_position,2:end);
v_halfcycle2(l_arm_mwx_velocity,:) = -v_halfcycle(r_arm_mwx_velocity,2:end);
q_halfcycle2(r_arm_mwx_position,:) = -q_halfcycle(l_arm_mwx_position,2:end);
v_halfcycle2(r_arm_mwx_velocity,:) = -v_halfcycle(l_arm_mwx_velocity,2:end);

l_leg_hpz_position = robot.getBody(robot.findJointInd('l_leg_hpz')).position_num;
r_leg_hpz_position = robot.getBody(robot.findJointInd('r_leg_hpz')).position_num;
l_leg_hpz_velocity = robot.getBody(robot.findJointInd('l_leg_hpz')).velocity_num;
r_leg_hpz_velocity = robot.getBody(robot.findJointInd('r_leg_hpz')).velocity_num;
q_halfcycle2(l_leg_hpz_position,:) = -q_halfcycle(r_leg_hpz_position,2:end);
v_halfcycle2(l_leg_hpz_velocity,:) = -v_halfcycle(r_leg_hpz_velocity,2:end);
q_halfcycle2(r_leg_hpz_position,:) = -q_halfcycle(l_leg_hpz_position,2:end);
v_halfcycle2(r_leg_hpz_velocity,:) = -v_halfcycle(l_leg_hpz_velocity,2:end);

l_leg_hpx_position = robot.getBody(robot.findJointInd('l_leg_hpx')).position_num;
r_leg_hpx_position = robot.getBody(robot.findJointInd('r_leg_hpx')).position_num;
l_leg_hpx_velocity = robot.getBody(robot.findJointInd('l_leg_hpx')).velocity_num;
r_leg_hpx_velocity = robot.getBody(robot.findJointInd('r_leg_hpx')).velocity_num;
q_halfcycle2(l_leg_hpx_position,:) = -q_halfcycle(r_leg_hpx_position,2:end);
v_halfcycle2(l_leg_hpx_velocity,:) = -v_halfcycle(r_leg_hpx_velocity,2:end);
q_halfcycle2(r_leg_hpx_position,:) = -q_halfcycle(l_leg_hpx_position,2:end);
v_halfcycle2(r_leg_hpx_velocity,:) = -v_halfcycle(l_leg_hpx_velocity,2:end);

l_leg_hpy_position = robot.getBody(robot.findJointInd('l_leg_hpy')).position_num;
r_leg_hpy_position = robot.getBody(robot.findJointInd('r_leg_hpy')).position_num;
l_leg_hpy_velocity = robot.getBody(robot.findJointInd('l_leg_hpy')).velocity_num;
r_leg_hpy_velocity = robot.getBody(robot.findJointInd('r_leg_hpy')).velocity_num;
q_halfcycle2(l_leg_hpy_position,:) = q_halfcycle(r_leg_hpy_position,2:end);
v_halfcycle2(l_leg_hpy_velocity,:) = v_halfcycle(r_leg_hpy_velocity,2:end);
q_halfcycle2(r_leg_hpy_position,:) = q_halfcycle(l_leg_hpy_position,2:end);
v_halfcycle2(r_leg_hpy_velocity,:) = v_halfcycle(l_leg_hpy_velocity,2:end);

l_leg_kny_position = robot.getBody(robot.findJointInd('l_leg_kny')).position_num;
r_leg_kny_position = robot.getBody(robot.findJointInd('r_leg_kny')).position_num;
l_leg_kny_velocity = robot.getBody(robot.findJointInd('l_leg_kny')).velocity_num;
r_leg_kny_velocity = robot.getBody(robot.findJointInd('r_leg_kny')).velocity_num;
q_halfcycle2(l_leg_kny_position,:) = q_halfcycle(r_leg_kny_position,2:end);
v_halfcycle2(l_leg_kny_velocity,:) = v_halfcycle(r_leg_kny_velocity,2:end);
q_halfcycle2(r_leg_kny_position,:) = q_halfcycle(l_leg_kny_position,2:end);
v_halfcycle2(r_leg_kny_velocity,:) = v_halfcycle(l_leg_kny_velocity,2:end);

l_leg_aky_position = robot.getBody(robot.findJointInd('l_leg_aky')).position_num;
r_leg_aky_position = robot.getBody(robot.findJointInd('r_leg_aky')).position_num;
l_leg_aky_velocity = robot.getBody(robot.findJointInd('l_leg_aky')).velocity_num;
r_leg_aky_velocity = robot.getBody(robot.findJointInd('r_leg_aky')).velocity_num;
q_halfcycle2(l_leg_aky_position,:) = q_halfcycle(r_leg_aky_position,2:end);
v_halfcycle2(l_leg_aky_velocity,:) = v_halfcycle(r_leg_aky_velocity,2:end);
q_halfcycle2(r_leg_aky_position,:) = q_halfcycle(l_leg_aky_position,2:end);
v_halfcycle2(r_leg_aky_velocity,:) = v_halfcycle(l_leg_aky_velocity,2:end);

l_leg_akx_position = robot.getBody(robot.findJointInd('l_leg_akx')).position_num;
r_leg_akx_position = robot.getBody(robot.findJointInd('r_leg_akx')).position_num;
l_leg_akx_velocity = robot.getBody(robot.findJointInd('l_leg_akx')).velocity_num;
r_leg_akx_velocity = robot.getBody(robot.findJointInd('r_leg_akx')).velocity_num;
q_halfcycle2(l_leg_akx_position,:) = -q_halfcycle(r_leg_akx_position,2:end);
v_halfcycle2(l_leg_akx_velocity,:) = -v_halfcycle(r_leg_akx_velocity,2:end);
q_halfcycle2(r_leg_akx_position,:) = -q_halfcycle(l_leg_akx_position,2:end);
v_halfcycle2(r_leg_akx_velocity,:) = -v_halfcycle(l_leg_akx_velocity,2:end);

neck_ay_position = robot.getBody(robot.findJointInd('neck_ay')).position_num;
neck_ay_velocity = robot.getBody(robot.findJointInd('neck_ay')).velocity_num;
q_halfcycle2(neck_ay_position,:) = q_halfcycle(neck_ay_position,2:end);
v_halfcycle2(neck_ay_velocity,:) = v_halfcycle(neck_ay_velocity,2:end);

com_halfcycle2 = com_halfcycle(:,2:end);
com_halfcycle2(1,:) = com_halfcycle2(1,:)+com_halfcycle(1,end);
com_halfcycle2(2,:) = -com_halfcycle2(2,:);
comdot_halfcycle2 = comdot_halfcycle(:,2:end);
comdot_halfcycle2(2,:) = -comdot_halfcycle2(2,:);
comddot_halfcycle2 = comddot_halfcycle(:,2:end);
comddot_halfcycle2(2,:) = -comddot_halfcycle2(2,:);

x_cycle = [x_halfcycle [q_halfcycle2;v_halfcycle2]];
t_cycle = [t_halfcycle t_halfcycle2];
com_cycle = [com_halfcycle com_halfcycle2];
comdot_cycle = [comdot_halfcycle comdot_halfcycle2];
comddot_cycle = [comddot_halfcycle comddot_halfcycle2];
end