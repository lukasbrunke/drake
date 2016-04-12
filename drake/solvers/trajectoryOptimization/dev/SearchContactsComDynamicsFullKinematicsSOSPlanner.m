classdef SearchContactsComDynamicsFullKinematicsSOSPlanner < ContactWrenchSetDynamicsFullKineamticsPlanner
  % This planner searches for the contact locations, and maximize the
  % contact wrench set margin, using the sos condition
  properties(SetAccess = protected)
    use_lin_fc % A flag. true if we are going to use linearized friction cone
    num_fc_pts % a obj.N x 1 vector. num_fc_pts(i) is the number of friction cone contact points at knot i
    num_grasp_pts % A obj.N x 1 vector. num_grasp_pts(i) is the number of grasping point at knot i
    num_grasp_wrench_vert % A obj.N x 1 cell, num_grasp_wrench_vert{i} is a num_grasp_pts(i) x 1 vector, num_grasp_wrench_vert{i}(j) is the number of vertices in the wrench polytope at j'th grasping point at knot i
    fc_contact_pos_inds % A obj.N x 1 cell, fc_contact_pos_inds{i} is a 3 x num_fc_pts(i) matrix
    grasp_contact_pos_inds % A obj.N x 1 cell, grasp_contact_pos_inds{i} is a 3 x num_grasp_pts(i) matrix
    l0_gram_var_inds % A 36 x nT matrix
    l1_gram_var_inds % A 36 x nT matrix
    l2_gram_var_inds % A nT x 1 cell, l2_gram_var_inds{i} is a 36 x (num_fc_pts*num_fc_edges) matrix
    l3_gram_var_inds % A 36 x nT matrix
    l4_gram_var_inds % A nT x 1 cell, l4_gram_var_inds{i} is a 36 x sum(num_grasp_wrench_vert{i}) matrix
    
    Qw % A 6 x 6 matrix, w'*Qw*w<=1 is the unit ball in the wrench space for wrench disturbance Qw
    
    fc_cw % A cell of FrictionConeWrench
    grasp_cw % A cell of GraspWrenchPolytope
    fc_cw_pos_inds % A length(fc_cw) x 1 cell, fc_cw_pos_inds{i} is a 3 x fc_cw.num_pts matrix
    grasp_cw_pos_inds % A length(grasp_cw) x 1 cell, grasp_cw_pos_inds{i} is a 3 x grasp_cw.num_pts matrix
  end
  
  properties(Access = protected)
    Qw_inv
    
    l0
    l1
    l2
    l3
    l4
  end
  
  methods
    function obj = SearchContactsComDynamicsFullKinematicsSOSPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,options)
      if(nargin< 11)
        options = struct();
      end
      if(~isfield(options,'use_lin_fc'))
        options.use_lin_fc = true;
      end
      obj = obj@ContactWrenchSetDynamicsFullKineamticsPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,options);
      
      obj = obj.parseContactWrenchStruct(obj,contact_wrench_struct);
      sizecheck(Qw,[6,6]);
      Qw = (Qw+Qw')/2;
      if(any(eig(Qw)<=0))
        error('Qw should be positive definite');
      end
      obj.Qw = Qw;
      obj.Qw_inv = inv(obj.Qw);
    end
  end
  
  methods(Access = protected)
    function obj = parseContactWrenchStruct(obj,contact_wrench_struct)
      num_cw = numel(contact_wrench_struct);
      obj.num_fc_pts = zeros(obj.N,1);
      obj.num_grasp_pts = zeros(obj.N,1);
      obj.num_grasp_wrench_vert = zeros(obj.N,1);
      obj.fc_cw = {};
      obj.grasp_cw = {};
      obj.fc_contact_pos_inds = cell(obj.N,1);
      obj.grasp_contact_pos_inds = cell(obj.N,1);
      obj.fc_cw_pos_inds = {};
      obj.grasp_cw_pos_inds = {};
      for i = 1:num_cw
        if(~isstruct(contact_wrench_struct) || ~isfield(contact_wrench_struct(i),'active_knot')...
            || ~isfield(contact_wrench_struct(i),'cw'))
          error('expect a struct with active_knot and cw fields');
        end
        if(isa(contact_wrench_struct(i).cw,'FrictionConeWrench'))
          obj.fc_cw{end+1} = contact_wrench_struct(i).cw;
          num_pts = contact_wrench_struct(i).cw.num_pts;
          obj.num_fc_pts(contact_wrench_struct(i).active_knot) = obj.num_fc_pts(contact_wrench_struct(i).active_knot)+num_pts;
          x_names = repmat({sprintf('%s contact pos',contact_wrench_struct(i).cw.body_name)},3*num_pts,1);
          [obj,tmp_idx] = obj.addDecisionVariable(3*num_pts,x_names);
          obj.fc_cw_pos_inds{end+1} = reshape(tmp_idx,3,num_pts);
          for j = reshape(contact_wrench_struct(i).active_knot,1,[])
            obj.fc_contact_pos_inds{j} = [obj.fc_contact_pos_inds{j} obj.fc_cw_pos_inds{end}];
          end
          % check if the body points are colinear. If yes, fix the body
          % pose, otherwise fix the body point positions
          co_linear_flag = false;
          body_pts_diff = diff(contact_wrench_struct(i).cw.body_pts,2);
          if(size(body_pts_diff,2)<=1 || all(sum(cross(body_pts_diff(:,1:end-1),body_pts_diff(:,2:end)).^2,1)<1e-4))
            co_linear_flag = true;
          end
          if(co_linear_flag)
            cnstr = WorldFixedPositionConstraint(obj.robot,contact_wrench_struct(i).cw.body,contact_wrench_struct(i).cw.body_pts);
            obj = obj.addRigidBodyConstraint(cnstr,{contact_wrench_struct(i).active_knot});
          else
            cnstr = WorldFixedBodyPoseConstraint(obj.robot,contact_wrench_struct(i).cw.body);
            obj = obj.addRigidBodyConstraint(cnstr,{contact_wrench_struct(i).active_knot});
          end
        elseif(isa(contact_wrench_struct(i).cw,'GraspWrenchPolytope'))
          obj.grasp_cw{end+1} = contact_wrench_struct(i).cw;
          valuecheck(contact_wrench_struct(i).cw.num_pts,1);
          obj.num_grasp_pts(contact_wrench_struct(i).active_knot) = obj.num_grasp_pts(contact_wrench_struct(i).active_knot)+1;
          x_names = repmat({sprintf('%s contact pos',contact_wrench_struct(i).cw.body_name)},3,1);
          [obj,tmp_idx] = obj.addDecisionVariable(3,x_names);
          obj.grasp_cw_pos_inds{end+1} = tmp_idx;
          for j = reshape(contact_wrench_struct(i).active_knot,1,[])
            obj.grasp_contact_pos_inds{j} = [obj.grasp_contact_pos_inds{j} obj.grasp_cw_pos_inds{end}];
          end
          cnstr = WorldFixedPositionConstraint(obj.robot,obj.grasp_cw{end}.body,obj.grasp_cw{end}.body_pts);
          obj = obj.addRigidBodyConstraint(cnstr,{obj.contact_wrench_struct(i).active_knot});
        end
      end
    end
  end
end