function testCoMPlanning
r = RigidBodyManipulator([getDrakePath,'/examples/Atlas/urdf/atlas_minimal_contact.urdf'],struct('floating',true));
nq = r.getNumDOF();

r_foot = r.findLinkInd('r_foot');
l_foot = r.findLinkInd('l_foot');
r_hand = r.findLinkInd('r_hand');
l_hand = r.findLinkInd('l_hand');
head = r.findLinkInd('head');
pelvis = r.findLinkInd('pelvis');

r_foot_contact_pts = getContactPoints(getBody(r,r_foot));
r_foot_pts = r_foot_contact_pts(:,1);
l_foot_contact_pts = getContactPoints(getBody(r,l_foot));
l_foot_pts = l_foot_contact_pts(:,1);
r_hand_pts = mean(getContactPoints(getBody(r,r_hand)),2);
l_hand_pts = mean(getContactPoints(getBody(r,l_hand)),2);

l_leg_kny = r.body(r.findJointInd('l_leg_kny')).dofnum;
r_leg_kny = r.body(r.findJointInd('r_leg_kny')).dofnum;
nom_data = load([getDrakePath,'/examples/Atlas/data/atlas_fp.mat']);
q0 = nom_data.xstar(1:nq);
qdot0 = zeros(nq,1);
kinsol0 = doKinematics(r,q0,false,false);
r_foot_pos = forwardKin(r,kinsol0,r_foot,r_foot_pts,2);
r_foot_pos(3,:) = 0;
l_foot_pos = forwardKin(r,kinsol0,l_foot,l_foot_pts,2);
l_foot_pos(3,:) = 0;
r_hand_pos = forwardKin(r,kinsol0,r_hand,r_hand_pts,0);
l_hand_pos = forwardKin(r,kinsol0,l_hand,l_hand_pts,0);
com_pos0 = getCOM(r,kinsol0,1);
com_height = com_pos0(3);
tspan = [0,1];
kc1 = {WorldPositionConstraint(r,r_foot,r_foot_pts,r_foot_pos(1:3),r_foot_pos(1:3),tspan),...
  WorldQuatConstraint(r,r_foot,r_foot_pos(4:7),0,tspan)};
kc2 = {WorldPositionConstraint(r,l_foot,l_foot_pts,l_foot_pos(1:3),l_foot_pos(1:3),tspan),...
  WorldQuatConstraint(r,l_foot,l_foot_pos(4:7),0,tspan)};
kc3 = WorldPositionConstraint(r,r_hand,r_hand_pts,r_hand_pos+[0.1;0.05;0.75],r_hand_pos+[0.1;0.05;1],[tspan(end) tspan(end)]);
kc4 = WorldPositionConstraint(r,l_hand,l_hand_pts,l_hand_pos,l_hand_pos,[tspan(end) tspan(end)]);
kc5 = WorldCoMConstraint(r,[-inf;-inf;com_height],[inf;inf;com_height+0.5],tspan,1);
pc_knee = PostureConstraint(r,tspan);

pc_knee = pc_knee.setJointLimits([l_leg_kny;r_leg_kny],[0.2;0.2],[inf;inf]);


nT = 8;
t = linspace(tspan(1),tspan(end),nT);
q_nom_traj = PPTrajectory(foh(t,repmat(q0,1,nT)));
q_seed_traj = PPTrajectory(zoh(t,repmat(q0,1,nT)+[zeros(nq,1) 1e-3*randn(nq,nT-1)]));

lfoot_fc = FrictionConeWrenchConstraint(r,l_foot,l_foot_contact_pts,1,[0;0;1],tspan);
rfoot_fc = FrictionConeWrenchConstraint(r,r_foot,r_foot_contact_pts,1,[0;0;1],tspan);
com_planning = CoMPlanning(r,t,q_nom_traj,true,[q0;qdot0],kc1{:},kc2{:},kc3,kc4,kc5,pc_knee,lfoot_fc,rfoot_fc);
com_planning = com_planning.setSolverOptions('snopt','print','planning.out');
com_planning = com_planning.addBoundingBoxConstraint(BoundingBoxConstraint(zeros(3,1),zeros(3,1)),com_planning.comdot_idx(:,end));
tic
[xtraj,comtraj,force_traj,F,indo,infeasible_constraint] = com_planning.solve(q_seed_traj);
toc
end