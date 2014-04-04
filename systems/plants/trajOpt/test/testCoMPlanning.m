function testCoMPlanning
% NOTEST
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

end