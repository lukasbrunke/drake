function testConvexObjectMinDistConstraint()
r = RigidBodyManipulator([getDrakePath,'/examples/Atlas/urdf/atlas_convex_hull.urdf'],struct('floating',true));
box1_pos = [0.3;-0.45;1];
box1_dim = [0.05;0.05;0.05];
box1_pt = bsxfun(@times,box1_pos,ones(1,8)) + bsxfun(@times,box1_dim,ones(1,8)).*[1 1 1 1 -1 -1 -1 -1;1 1 -1 -1 1 1 -1 -1;1 -1 1 -1 1 -1 1 -1];
lcmgl = drake.util.BotLCMGLClient(lcm.lcm.LCM.getSingleton(),'box1');
lcmgl.glColor3f(1,0,0);
lcmgl.box(box1_pos,box1_dim*2);
lcmgl.switchBuffers();
r = r.addRobotFromURDF('block.urdf',box1_pos,zeros(3,1),struct('floating',false'));
r = r.compile();
v = r.constructVisualizer(struct('use_collision_geometry',true));
nq = r.getNumPositions();
l_foot = r.findLinkId('l_foot');
r_foot = r.findLinkId('r_foot');
l_foot_pts = r.getBody(l_foot).getTerrainContactPoints();
r_foot_pts = r.getBody(r_foot).getTerrainContactPoints();

r_uarm = r.findLinkId('r_uarm');
r_larm = r.findLinkId('r_larm');
r_uarm_geo = r.getBody(r_uarm).getCollisionGeometry();
r_uarm_pts = r_uarm_geo{1}.getPoints();
r_larm_geo = r.getBody(r_larm).getCollisionGeometry();
r_larm_pts = r_larm_geo{1}.getBoundingBoxPoints();
l_hand = r.findLinkId('l_hand');
l_hand_geo = r.getBody(l_hand).getCollisionGeometry();
l_hand_pts = l_hand_geo{1}.getPoints();
r_hand = r.findLinkId('r_hand');
r_hand_geo = r.getBody(r_hand).getCollisionGeometry();
r_hand_pts = r_hand_geo{1}.getPoints();
l_hand_pt = [0;0.2;0];
r_hand_pt = [0;-0.2;0];
r_lfarm = r.findLinkId('r_lfarm');
r_lfarm_geo = r.getBody(r_lfarm).getCollisionGeometry();
r_lfarm_pts = r_lfarm_geo{1}.getPoints();


nom_data = load([getDrakePath,'/examples/Atlas/data/atlas_fp.mat']);
qstar = nom_data.xstar(1:nq);
kinsol_star = r.doKinematics(qstar);
l_foot_pos0 = r.forwardKin(kinsol_star,l_foot,zeros(3,1),2);
r_foot_pos0 = r.forwardKin(kinsol_star,r_foot,zeros(3,1),2);
l_foot_cnstr = {WorldPositionConstraint(r,l_foot,zeros(3,1),l_foot_pos0(1:3),l_foot_pos0(1:3)),...
  WorldQuatConstraint(r,l_foot,l_foot_pos0(4:7),0)};
r_foot_cnstr = {WorldPositionConstraint(r,r_foot,zeros(3,1),r_foot_pos0(1:3),r_foot_pos0(1:3)),...
  WorldQuatConstraint(r,r_foot,r_foot_pos0(4:7),0)};
qsc = QuasiStaticConstraint(r);
qsc = qsc.addContact(l_foot,l_foot_pts,r_foot,r_foot_pts);
pelvis_cnstr = PostureConstraint(r);
pelvis_cnstr = pelvis_cnstr.setJointLimits([1:3]',[0;0;-inf],[0;0;inf]);

r_hand_cnstr = WorldPositionConstraint(r,r_hand,r_hand_pt,[0.5;-0.5;1],[5;-0.5;1]);
p = InverseKinematics(r,qstar,l_foot_cnstr{:},r_foot_cnstr{:},qsc,r_hand_cnstr);
tic;[q,F,info] = p.solve(qstar);toc;

min_dist1 = PolygonMinDistConstraint(r,[r_hand,1],r_hand_pts,box1_pt,0.03);
min_dist2 = PolygonMinDistConstraint(r,[r_larm,1],r_larm_pts,box1_pt,0.03);
min_dist3 = PolygonMinDistConstraint(r,[r_lfarm,1],r_lfarm_pts,box1_pt,0.03);
p = p.addPolygonMinDistConstraint(min_dist1);
p = p.addPolygonMinDistConstraint(min_dist2);
p = p.addPolygonMinDistConstraint(min_dist3);
p = p.setSolverOptions('snopt','MajorIterationsLimit',1000);
tic;[q2,F,info] = p.solve(q);toc;

end