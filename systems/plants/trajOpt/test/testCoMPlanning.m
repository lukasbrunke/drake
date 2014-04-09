function testCoMPlanning
% NOTEST
r = RigidBodyManipulator([getDrakePath,'/examples/Atlas/urdf/atlas_minimal_contact.urdf'],struct('floating',true));
nq = r.getNumDOF();
v = r.constructVisualizer();

r_foot = r.findLinkInd('r_foot');
l_foot = r.findLinkInd('l_foot');
r_hand = r.findLinkInd('r_hand');
l_hand = r.findLinkInd('l_hand');
head = r.findLinkInd('head');
pelvis = r.findLinkInd('pelvis');

r_foot_contact_pts = getContactPoints(getBody(r,r_foot));
r_foot_pt = zeros(3,1);
l_foot_contact_pts = getContactPoints(getBody(r,l_foot));
l_foot_pt = zeros(3,1);
r_hand_pts = mean(getContactPoints(getBody(r,r_hand)),2);
l_hand_pts = mean(getContactPoints(getBody(r,l_hand)),2);

nomdata = load([getDrakePath,'/examples/Atlas/data/atlas_fp.mat']);
qstar = nomdata.xstar(1:nq);
v.draw(0,[nomdata.xstar]);

kinsol0 = r.doKinematics(qstar,false,false);
rfoot_pos0 = r.forwardKin(kinsol0,r_foot,r_foot_pt,1);
lfoot_pos0 = r.forwardKin(kinsol0,l_foot,l_foot_pt,1);
com0 = r.getCOM(kinsol0);

% test walking with only friction cone constraint
% right foot takes off first
num_steps = 4;
step_length = 0.3;
lfoot_pos = zeros(6,num_steps);
rfoot_pos = zeros(6,num_steps);
lfoot_contact_pos = zeros(3,size(l_foot_contact_pts,2),num_steps);
rfoot_contact_pos = zeros(3,size(r_foot_contact_pts,2),num_steps);
lfoot_pos(:,1) = lfoot_pos0;
rfoot_pos(:,1) = rfoot_pos0;
rfoot_pos(:,2) = rfoot_pos(:,1)+[step_length;0;0;0;0;0]+[0.03*(rand(3,1)-0.5*ones(3,1));0.1*pi*(rand(3,1)-0.5*ones(3,1))];
lfoot_pos(:,2) = lfoot_pos(:,1)+[2*step_length;0;0;0;0;0]+[0.03*(rand(3,1)-0.5*ones(3,1));0.1*pi*(rand(3,1)-0.5*ones(3,1))];
for i = 3:num_steps
  lfoot_pos(:,i) = lfoot_pos(:,i-1)+[2*step_length;0;0;0;0;0]+[0.03*(rand(3,1)-0.5*ones(3,1));0.1*pi*(rand(3,1)-0.5*ones(3,1))];
  rfoot_pos(:,i) = rfoot_pos(:,i-1)+[2*step_length;0;0;0;0;0]+[0.03*(rand(3,1)-0.5*ones(3,1));0.1*pi*(rand(3,1)-0.5*ones(3,1))];
end
lfoot_fc_axis = zeros(3,num_steps);
rfoot_fc_axis = zeros(3,num_steps);
checkDependency('lcmgl');
for i = 1:num_steps
  lfoot_contact_pos(:,:,i) = contactPosition(lfoot_pos(:,i),l_foot_contact_pts);
  lcmgl = drake.util.BotLCMGLClient(lcm.lcm.LCM.getSingleton(),sprintf('lfoot%d',i));
  lcmgl.glColor3f(255,0,0);
  plotFoot(lcmgl,lfoot_contact_pos(:,:,i));
  lcmgl.switchBuffers();
  rfoot_contact_pos(:,:,i) = contactPosition(rfoot_pos(:,i),r_foot_contact_pts);
  lcmgl = drake.util.BotLCMGLClient(lcm.lcm.LCM.getSingleton(),sprintf('rfoot%d',i));
  lcmgl.glColor3f(255,0,0);
  plotFoot(lcmgl,rfoot_contact_pos(:,:,i));
  lcmgl.switchBuffers();
  lfoot_fc_axis(:,i) = rpy2rotmat(lfoot_pos(4:6,i))*[0;0;1];
  rfoot_fc_axis(:,i) = rpy2rotmat(rfoot_pos(4:6,i))*[0;0;1];
end
lfoot_landing_time = 0;
rfoot_landing_time = 0;
ts_end = 0;
lfoot_contact_wrench_constraint = cell(1,num_steps);
rfoot_contact_wrench_constraint = cell(1,num_steps);
mu = 1;
t_knot = 0;
for i = 1:num_steps-1
  [ts,~,takeoff_time,landing_time] = planSwing(r,rfoot_pos(:,i),rfoot_pos(:,i+1),struct('ignore_terrain',true,'step_height',0.1));
  takeoff_time = ts_end+takeoff_time;
  landing_time = ts_end+landing_time;
  t_knot = [t_knot ts_end+ts(2:end)];
  ts_end = ts_end+ts(end);
  rfoot_contact_wrench_constraint{i} = FrictionConeWrenchConstraint(r,r_foot,r_foot_contact_pts,mu,rfoot_fc_axis(:,1),[rfoot_landing_time,takeoff_time]);
  rfoot_landing_time = landing_time;
  [ts,~,takeoff_time,landing_time] = planSwing(r,lfoot_pos(:,i),lfoot_pos(:,i+1),struct('ignore_terrain',true,'step_height',0.1));
  takeoff_time = ts_end+takeoff_time;
  landing_time = ts_end+landing_time;
  t_knot = [t_knot ts_end+ts(2:end)];
  ts_end = ts_end+ts(end);
  lfoot_contact_wrench_constraint{i} = FrictionConeWrenchConstraint(r,l_foot,r_foot_contact_pts,mu,lfoot_fc_axis(:,1),[lfoot_landing_time,takeoff_time]);
  lfoot_landing_time = landing_time;
end
rfoot_contact_wrench_constraint{end} = FrictionConeWrenchConstraint(r,r_foot,r_foot_contact_pts,mu,rfoot_fc_axis(:,end),[rfoot_landing_time,ts_end]);
lfoot_contact_wrench_constraint{end} = FrictionConeWrenchConstraint(r,l_foot,l_foot_contact_pts,mu,lfoot_fc_axis(:,end),[lfoot_landing_time,ts_end]);

t_knot = [reshape([t_knot(1:end-1);sum([t_knot(1:end-1);t_knot(2:end)],1)/2],1,[]) t_knot(end)];
contact_args = cell(2*num_steps*2,1);
for i = 1:num_steps
  contact_args{4*(i-1)+1} = rfoot_contact_wrench_constraint{i};
  contact_args{4*(i-1)+2} = rfoot_contact_pos(:,:,i);
  contact_args{4*(i-1)+3} = lfoot_contact_wrench_constraint{i};
  contact_args{4*(i-1)+4} = lfoot_contact_pos(:,:,i);
end
kc_final = {WorldPositionConstraint(r,r_foot,r_foot_pt,rfoot_pos(1:3,end),rfoot_pos(1:3,end),[t_knot(end) t_knot(end)]),...
  WorldEulerConstraint(r,r_foot,rfoot_pos(4:6,end),rfoot_pos(4:6,end),[t_knot(end) t_knot(end)]),...
  WorldPositionConstraint(r,l_foot,l_foot_pt,lfoot_pos(1:3,end),lfoot_pos(1:3,end),[t_knot(end) t_knot(end)]),...
  WorldEulerConstraint(r,l_foot,lfoot_pos(4:6,end),lfoot_pos(4:6,end),[t_knot(end) t_knot(end)])};
qfinal = inverseKin(r,qstar,qstar,kc_final{:});
kinsol_final = r.doKinematics(qfinal,false,false);
com_final = getCOM(r,kinsol_final);
Q_com = eye(3);
Q_comddot = eye(3);
com_des = bsxfun(@times,com0,ones(1,length(t_knot)-1))+bsxfun(@times,(com_final-com0)/(length(t_knot)-1),(1:length(t_knot)-1));
com_des = [com0 com_des];
tic
com_planning = CoMPlanning(r.getMass,t_knot,Q_com,Q_comddot,com_des,1,true,true,contact_args{:});

com_planning = com_planning.setXbounds(com0,com0,com_planning.com_idx(:,1));
com_planning = com_planning.setXbounds(com_final,com_final,com_planning.com_idx(:,end));
com_planning = com_planning.setXbounds(zeros(3,1),zeros(3,1),com_planning.comdot_idx(:,1));
com_planning = com_planning.setXbounds(zeros(3,1),zeros(3,1),com_planning.comdot_idx(:,end));
com_planning = com_planning.setXbounds(0.8*ones(com_planning.nT,1),1.2*ones(com_planning.nT,1),com_planning.com_idx(3,:));
[com,comdot,comddot,F,info] = com_planning.solve();
toc
end

function pos = contactPosition(body_pos,body_pts)
pos = [rpy2rotmat(body_pos(4:6)) body_pos(1:3);0 0 0 1]*[body_pts;ones(1,size(body_pts,2))];
pos = pos(1:3,:);
end

function plotFoot(lcmgl,foot_contact_pos)
conv_idx = convhull(foot_contact_pos(1,:),foot_contact_pos(2,:));
for i = 1:length(conv_idx)-1
  lcmgl.line3(foot_contact_pos(1,conv_idx(i)),foot_contact_pos(2,conv_idx(i)),foot_contact_pos(3,conv_idx(i)),...
    foot_contact_pos(1,conv_idx(i+1)),foot_contact_pos(2,conv_idx(i+1)),foot_contact_pos(3,conv_idx(i+1)));
end
end