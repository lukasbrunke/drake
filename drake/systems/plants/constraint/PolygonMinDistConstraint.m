classdef PolygonMinDistConstraint < Constraint
  % Enforce the closest distance between two convex objects should be no
  % smaller than a certain value. This constraint is implemented by
  % searching over the separating hyperplane between the convex objects.
  properties(SetAccess = protected)
    rbm
    body_pair % A 1 x 2 vector, the constraint is that the distance between the convex hull of body_pair(1) and body_pair(2) is no smaller than min_dist
    body1_pts % The contact points on body_pair(1) in the body frame
    body2_pts % The contact points on body_pair(2) in the body frame
    min_dist % A scalar. the min_dist is the smallest distance between body_pair(1) and body_pair(2)
    nq
  end
  
  properties(Access = protected)
    
    num_pts1
    num_pts2
  end
  
  methods
    function obj = PolygonMinDistConstraint(rbm,body_pair,body1_pts,body2_pts,min_dist)
      if(~isa(rbm,'RigidBodyManipulator') && ~isa(rbm,'TimeSteppingRigidBodyManipulator'))
        error('rbm should be a RigidBodyManipulator or a TimeSteppingRigidBodyManipulator');
      end
      m_nq = rbm.getNumPositions();
      if(any(size(body_pair) ~= [1,2]))
        error('body_pair should be a 1 x 2 vector');
      end
      m_num_pts1 = size(body1_pts,2);
      m_num_pts2 = size(body2_pts,2);
      if(numel(min_dist) ~= 1)
        error('min_dist should be a scalar');
      end
      obj = obj@Constraint([1;0.5*min_dist*ones(m_num_pts1,1);-inf(m_num_pts2,1)],[1;inf(m_num_pts1,1);-0.5*min_dist*ones(m_num_pts2,1)],m_nq+4,1);
      obj.rbm = rbm;
      obj.body_pair = body_pair;
      obj.min_dist = min_dist;
      obj.nq = m_nq;
      obj.body1_pts = body1_pts;
      obj.body2_pts = body2_pts;
      obj.num_pts1 = m_num_pts1;
      obj.num_pts2 = m_num_pts2;
      joint_idx1 = obj.kinematicsPathJoints(body_pair(1));
      joint_idx2 = obj.kinematicsPathJoints(body_pair(2));
      iGfun = [ones(3,1);reshape(bsxfun(@times,1+(1:obj.num_pts1)',ones(1,length(joint_idx1)+4)),[],1);...
        reshape(bsxfun(@times,1+obj.num_pts1+(1:obj.num_pts2)',ones(1,length(joint_idx2)+4)),[],1)];
      jGvar = [obj.nq+(1:3)';reshape(bsxfun(@times,ones(obj.num_pts1,1),[joint_idx1 obj.nq+(1:4)]),[],1);...
        reshape(bsxfun(@times,ones(obj.num_pts2,1),[joint_idx2 obj.nq+(1:4)]),[],1)];
      obj = obj.setSparseStructure(iGfun,jGvar);
    end
  end
  
  methods(Access = protected)
    function [v,dv] = constraintEval(obj,q,c,d,kinsol)
      v = zeros(obj.num_cnstr,1);
      dv = zeros(obj.num_cnstr,obj.xdim);
      v(1) = c'*c;
      dv(1,obj.nq+(1:3)) = 2*c';
      [pts1_pos,dJ1] = obj.rbm.forwardKin(kinsol,obj.body_pair(1),obj.body1_pts);
      [pts2_pos,dJ2] = obj.rbm.forwardKin(kinsol,obj.body_pair(2),obj.body2_pts);
      v(1+(1:obj.num_pts1)) = pts1_pos'*c + d*ones(obj.num_pts1,1);
      dv(1+(1:obj.num_pts1),1:obj.nq) = sparse(reshape(bsxfun(@times,ones(3,1),1:obj.num_pts1),[],1),(1:3*obj.num_pts1)',reshape(bsxfun(@times,c,ones(1,obj.num_pts1)),[],1),obj.num_pts1,3*obj.num_pts1)*dJ1;
      dv(1+(1:obj.num_pts1),obj.nq+(1:4)) = [pts1_pos' ones(obj.num_pts1,1)];
      v(1+obj.num_pts1+(1:obj.num_pts2)) = pts2_pos'*c+d*ones(obj.num_pts2,1);
      dv(1+obj.num_pts1+(1:obj.num_pts2),1:obj.nq) = sparse(reshape(bsxfun(@times,ones(3,1),1:obj.num_pts2),[],1),(1:3*obj.num_pts2)',reshape(bsxfun(@times,c,ones(1,obj.num_pts2)),[],1),obj.num_pts2,3*obj.num_pts2)*dJ2;
      dv(1+obj.num_pts1+(1:obj.num_pts2),obj.nq+(1:4)) = [pts2_pos' ones(obj.num_pts2,1)];
    end
    
    function joint_idx = kinematicsPathJoints(obj,body_idx)
      [~,joint_path] = obj.rbm.findKinematicPath(1,body_idx);
      if isa(obj.rbm,'TimeSteppingRigidBodyManipulator')
        joint_idx = vertcat(obj.rbm.getManipulator().body(joint_path).position_num)';
      else
        joint_idx = vertcat(obj.rbm.body(joint_path).position_num)';
      end
    end
  end
end