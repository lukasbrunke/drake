classdef SearchContactsComDynamicsFullKinematicsSOSPlanner < ContactWrenchSetDynamicsFullKineamticsPlanner
  % This planner searches for the contact locations, and maximize the
  % contact wrench set margin, using the sos condition
  properties(SetAccess = protected)
    use_lin_fc % A flag. true if we are going to use linearized friction cone
    num_fc_edges % A scalar. If choosing to use linearized friction cone, the number of edges in each linearized friction cone
    num_fc_pts % a obj.N x 1 vector. num_fc_pts(i) is the number of friction cone contact points at knot i
    num_grasp_pts % A obj.N x 1 vector. num_grasp_pts(i) is the number of grasping point at knot i
    num_grasp_wrench_vert % A obj.N x 1 cell, num_grasp_wrench_vert{i} is a num_grasp_pts(i) x 1 vector, num_grasp_wrench_vert{i}(j) is the number of vertices in the wrench polytope at j'th grasping point at knot i
    fc_contact_pos_inds % A obj.N x 1 cell, fc_contact_pos_inds{i} is a 3 x num_fc_pts(i) matrix
    grasp_contact_pos_inds % A obj.N x 1 cell, grasp_contact_pos_inds{i} is a 3 x num_grasp_pts(i) matrix
    l0_gram_var_inds % A 36 x nT matrix
    l1_gram_var_inds % A 36 x nT matrix
    l2_gram_var_inds % A nT x 1 cell, l2_gram_var_inds{i} is a 36 x num_fc_pts x num_fc_edges) matrix
    l3_gram_var_inds % A 36 x nT matrix
    l4_gram_var_inds % A nT x 1 cell, l4_gram_var_inds{i} is a 36 x prod(num_grasp_wrench_vert{i}) matrix
    
    V_gram_var_inds % A 666 x nT matrix
    
    Qw % A 6 x 6 matrix, w'*Qw*w<=1 is the unit ball in the wrench space for wrench disturbance Qw
    
    fc_cw % A cell of FrictionConeWrench
    grasp_cw % A cell of GraspWrenchPolytope
    fc_cw_pos_inds % A length(fc_cw) x 1 cell, fc_cw_pos_inds{i} is a 3 x fc_cw.num_pts matrix
    grasp_cw_pos_inds % A length(grasp_cw) x 1 cell, grasp_cw_pos_inds{i} is a 3 x grasp_cw.num_pts matrix
  end
  
  properties(Access = protected)
    Qw_inv
    
    l0_gram_var
    l1_gram_var
    l2_gram_var
    l3_gram_var
    l4_gram_var
    V_gram_var
    momentum_dot_var
    com_var
    fc_pos_var
    grasp_pos_var
    cws_margin_var
    
    fc_axis
    fc_mu
    grasp_wrench_vert
    
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
      obj.use_lin_fc = options.use_lin_fc;
      obj.num_fc_edges = 4;
      
      obj = obj.parseContactWrenchStruct(contact_wrench_struct);
      sizecheck(Qw,[6,6]);
      Qw = (Qw+Qw')/2;
      if(any(eig(Qw)<=0))
        error('Qw should be positive definite');
      end
      obj.Qw = Qw;
      obj.Qw_inv = inv(obj.Qw);
    end
    
    function obj = addRunningCost(running_cost_fun)
    end
    
  end
  
  methods(Access = protected)
    function obj = parseContactWrenchStruct(obj,contact_wrench_struct)
      num_cw = numel(contact_wrench_struct);
      obj.num_fc_pts = zeros(obj.N,1);
      obj.num_grasp_pts = zeros(obj.N,1);
      obj.num_grasp_wrench_vert = cell(obj.N,1);
      obj.fc_cw = {};
      obj.grasp_cw = {};
      obj.fc_contact_pos_inds = cell(obj.N,1);
      obj.grasp_contact_pos_inds = cell(obj.N,1);
      obj.fc_cw_pos_inds = {};
      obj.grasp_cw_pos_inds = {};
      obj.fc_axis = cell(obj.N,1);
      obj.fc_mu = cell(obj.N,1);
      obj.grasp_wrench_vert = cell(obj.N,1);
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
            obj.fc_axis{j} = [obj.fc_axis{j} obj.fc_cw{end}.FC_axis];
            obj.fc_mu{j} = [obj.fc_mu{j} obj.fc_cw{end}.FC_mu];
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
            obj.num_grasp_wrench_vert{j} = [obj.num_grasp_wrench_vert{j} obj.grasp_cw{end}.num_wrench_vert];
            obj.grasp_wrench_vert{j} = [obj.grasp_wrench_vert{j} {obj.grasp_cw{end}.wrench_vert}];
          end
          cnstr = WorldFixedPositionConstraint(obj.robot,obj.grasp_cw{end}.body,obj.grasp_cw{end}.body_pts);
          obj = obj.addRigidBodyConstraint(cnstr,{contact_wrench_struct(i).active_knot});
        else
          error('Not supported');
        end
      end
    end
    
    function obj = addSOScondition(obj)
      obj.com_var = reshape(msspoly('r',3*obj.N),3,obj.N);
      obj.momentum_dot_var = reshape(msspoly('h',6*obj.N),6,obj.N);
      obj.cws_margin_var = msspoly('s',1);
      x_name0 = cell(36*obj.N,1);
      x_name1 = cell(36*obj.N,1);
      x_name3 = cell(36*obj.N,1);
      for i = 1:obj.N
        x_name0((i-1)*36+(1:36)) = repmat({sprintf('l0_gram_var[%d]',i)},36,1);
        x_name1((i-1)*36+(1:36)) = repmat({sprintf('l1_gram_var[%d]',i)},36,1);
        x_name3((i-1)*36+(1:36)) = repmat({sprintf('l3_gram_var[%d]',i)},36,1);
      end
      [obj,tmp_idx] = obj.addDecisionVariable(36*obj.N,x_name0);
      obj.l0_gram_var_inds = reshape(tmp_idx,36,obj.N);
      [obj,tmp_idx] = obj.addDecisionVariable(36*obj.N,x_name1);
      obj.l1_gram_var_inds = reshape(tmp_idx,36,obj.N);
      [obj,tmp_idx] = obj.addDecisionVariable(36*obj.N,x_name3);
      obj.l3_gram_var_inds = reshape(tmp_idx,36,obj.N);
      total_num_grasp_wrench = 0;
      for i = 1:obj.N
        total_num_grasp_wrench = total_num_grasp_wrench + prod(obj.num_grasp_wrench_vert{i});
      end
      if(obj.use_lin_fc)
        l_gram_var = msspoly('l',108*obj.N+36*sum(obj.num_fc_pts)*obj.num_fc_edges+36*total_num_grasp_wrench);
      else
        l_gram_var = msspoly('l',108*obj.N+36*sum(obj.num_fc_pts)*2+36*total_num_grasp_wrench);
      end
      obj.l0_gram_var = reshape(l_gram_var(1:36*obj.N),36,obj.N);
      obj.l1_gram_var = reshape(l_gram_var(36*obj.N+(1:36*obj.N)),36,obj.N);
      obj.l3_gram_var = reshape(l_gram_var(72*obj.N+(1:36*obj.N)),36,obj.N);

      obj.l2_gram_var_inds = cell(obj.N,1);
      obj.l4_gram_var_inds = cell(obj.N,1);
      obj.l2_gram_var = cell(obj.N,1);
      obj.l4_gram_var = cell(obj.N,1);
      if(obj.use_lin_fc)
        cws_sos = CWSMarginSOSconditionLinFC(obj.num_fc_edges,obj.robot_mass,obj.N,obj.Qw,obj.num_fc_pts,obj.num_grasp_pts,obj.num_grasp_wrench_vert);
      else
        cws_sos = CWSMarginSOSconditionNonlinearFC(obj.robot_mass,obj.N,obj.Qw,obj.num_fc_pts,obj.num_grasp_pts,obj.num_grasp_wrench_vert);
      end
      ab_monomials1 = monomials([cws_sos.a_indet;cws_sos.b_indet],0:1);
      l0_gram = cell(obj.N,1);
      l1_gram = cell(obj.N,1);
      l3_gram = cell(obj.N,1);
      tril_mask = tril(ones(8))~=0;

      l0 = msspoly.zeros(obj.N,1);
      l1 = msspoly.zeros(obj.N,1);
      l2 = cell(obj.N,1);
      l3 = msspoly.zeros(obj.N,1);
      l4 = cell(obj.N,1);
      l_gram_var_count = 108*obj.N;
      for i = 1:obj.N
        l0_gram{i} = msspoly.zeros(8,8);
        l0_gram{i}(tril_mask) = obj.l0_gram_var(:,i);
        l0_gram{i} = l0_gram{i}'*l0_gram{i}+1;
        l0(i) = ab_monomials1'*l0_gram{i}*ab_monomials1;

        l1_gram{i} = msspoly.zeros(8,8);
        l1_gram{i}(tril_mask) = obj.l1_gram_var(:,i);
        l1(i) = ab_monomials1'*l1_gram{i}*ab_monomials1;

        l3_gram{i} = msspoly.zeros(8,8);
        l3_gram{i}(tril_mask) = obj.l3_gram_var(:,i);
        l3_gram{i} = l3_gram{i}'*l3_gram{i};
        l3(i) = ab_monomials1'*l3_gram{i}*ab_monomials1;

        if(obj.use_lin_fc)
          l2{i} = msspoly.zeros(obj.num_fc_pts(i),obj.num_fc_edges);
          x_name2 = repmat({sprintf('l2_gram_var[%d]',i)},36*obj.num_fc_pts(i)*obj.num_fc_edges,1);
          [obj,tmp_idx] = obj.addDecisionVariable(36*obj.num_fc_pts(i)*obj.num_fc_edges,x_name2);
          obj.l2_gram_var_inds{i} = reshape(tmp_idx,36,obj.num_fc_pts(i),obj.num_fc_edges);
          obj.l2_gram_var{i} = reshape(l_gram_var(l_gram_var_count+(1:36*obj.num_fc_pts(i)*obj.num_fc_edges)),36,obj.num_fc_pts(i)*obj.num_fc_edges);
          l_gram_var_count = l_gram_var_count+36*obj.num_fc_pts(i)*obj.num_fc_edges;
          for j = 1:obj.num_fc_pts(i)
            for k = 1:obj.num_fc_edges
              l2_gram = msspoly.zeros(8,8);
              l2_gram(tril_mask) = obj.l2_gram_var{i}(:,(j-1)*obj.num_fc_edges+k);
              l2_gram = l2_gram'*l2_gram;
              l2{i}(j,k) = ab_monomials1'*l2_gram*ab_monomials1;
            end
          end
        else
        end
        num_grasp_wrench_vert_i = prod(obj.num_grasp_wrench_vert{i});
        l4{i} = msspoly.zeros(num_grasp_wrench_vert_i,1);
        x_name4 = repmat({sprintf('l4_gram_var[%d]',i)},36*num_grasp_wrench_vert_i,1);
        [obj,tmp_idx] = obj.addDecisionVariable(36*num_grasp_wrench_vert_i,x_name4);
        obj.l4_gram_var_inds{i} = reshape(tmp_idx,36,num_grasp_wrench_vert_i);
        obj.l4_gram_var{i} = reshape(l_gram_var(l_gram_var_count+(1:36*num_grasp_wrench_vert_i)),36,num_grasp_wrench_vert_i);
        l_gram_var_count = l_gram_var_count+36*num_grasp_wrench_vert_i;
        for j = 1:num_grasp_wrench_vert_i
          l4_gram = msspoly.zeros(8,8);
          l4_gram(tril_mask) = obj.l4_gram_var{i}(:,j);
          l4_gram = l4_gram'*l4_gram;
          l4{i}(j) = ab_monomials1'*l4_gram*ab_monomials1;
        end
      end

      friction_cones = cell(obj.N,1);
      contact_pos = msspoly('p',3*sum(obj.num_fc_pts)+3*sum(obj.num_grasp_pts));
      contact_pos_count = 0;
      obj.fc_pos_var = cell(obj.N,1);
      obj.grasp_pos_var = cell(obj.N,1);
      if(obj.use_lin_fc)
        fc_theta = linspace(0,2*pi,obj.num_fc_edges+1);
        fc_theta = fc_theta(1:end-1);
        fc_edges0 = obj.robot_mass*obj.gravity*[cos(fc_theta);sin(fc_theta);ones(1,obj.num_fc_edges)];
      end
      for i = 1:obj.N
        obj.fc_pos_var{i} = reshape(contact_pos(contact_pos_count+(1:3*obj.num_fc_pts(i))),3,obj.num_fc_pts(i));
        contact_pos_count = contact_pos_count+3*obj.num_fc_pts(i);
        obj.grasp_pos_var{i} = reshape(contact_pos(contact_pos_count+(1:3*obj.num_grasp_pts(i))),3,obj.num_grasp_pts(i));
        contact_pos_count = contact_pos_count+3*obj.num_grasp_pts(i);
        if(obj.use_lin_fc)
          friction_cones{i} = LinearizedFrictionCone.empty(0,obj.num_fc_pts(i));
          for j = 1:obj.num_fc_pts(i)
            fc_edges = fc_edges0;
            fc_edges(1:2,:) = fc_edges(1:2,:)*obj.fc_mu{i}(j);
            fc_edges = rotateVectorToAlign([0;0;1],obj.fc_axis{i}(:,j))*fc_edges;
            friction_cones{i}(j) = LinearizedFrictionCone(obj.fc_pos_var{i}(:,j),obj.fc_axis{i}(:,j),obj.fc_mu{i}(j),fc_edges);
          end
        end
      end

      V = cws_sos.CWSMarginSOScondition(l0,l1,l2,l3,l4,obj.cws_margin_var,friction_cones,obj.grasp_pos_var,obj.grasp_wrench_vert,obj.disturbance_pos,obj.momentum_dot_var,obj.com_var,zeros(3,obj.N));

      x_name = cell(666*obj.N,1);
      for i = 1:obj.N
        x_name((i-1)*666+(1:666)) = repmat({sprintf('V_gram_var[%d]',i)},666,1);
      end
      [obj,tmp_idx] = obj.addDecisionVariable(666*obj.N,x_name);
      obj.V_gram_var_inds = reshape(tmp_idx,666,obj.N);
      obj.V_gram_var = reshape(msspoly('v',666*obj.N),666,obj.N);
      ab_monomials2 = monomials([cws_sos.a_indet;cws_sos.b_indet],0:2);
      for i = 1:obj.N
        V_gram = msspoly.zeros(36,36);
        tril_mask = tril(ones(36))~=0;
        V_gram(tril_mask) = obj.V_gram_var(:,i);
        V_gram = V_gram'*V_gram;
        V(i) = V(i)-ab_monomials2'*V_gram*ab_monomials2;
        decision_var = [obj.l0_gram_var(:,i);obj.l1_gram_var(:,i);obj.l2_gram_var{i}(:);obj.l3_gram_var(:,i);obj.l4_gram_var{i}(:);obj.V_gram_var(:,i);obj.momentum_dot_var(:,i);obj.com_var(:,i);obj.fc_pos_var{i}(:);obj.grasp_pos_var{i}(:);obj.cws_margin_var];
        decision_var_inds = [obj.l0_gram_var_inds(:,i);obj.l1_gram_var_inds(:,i);obj.l2_gram_var_inds{i}(:);obj.l3_gram_var_inds(:,i);obj.l4_gram_var_inds{i}(:);obj.V_gram_var_inds(:,i);obj.world_momentum_dot_inds(:,i);obj.com_inds(:,i);obj.fc_contact_pos_inds{i}(:);obj.grasp_contact_pos_inds{i}(:);obj.cws_margin_ind];
        [V_indet,~,V_coeff] = decomp(V(i),decision_var);
        mtch = match([cws_sos.a_indet;cws_sos.b_indet],V_indet);
        valuecheck(numel(unique(mtch)),7);
        sparse_pattern = zeros(length(V_coeff),length(decision_var));
        dV_coeff = diff(V_coeff',decision_var);
        for j = 1:length(V_coeff)
          coeff_var_j = decomp(V_coeff(j));
          match_ij = match(decision_var,coeff_var_j);
          sparse_pattern(j,match_ij) = 1;
        end
        [coeff_var,coeff_power,coeff_M] = decomp(V_coeff);
        [dcoeff_var,dcoeff_power,dcoeff_M] = decomp(dV_coeff);
        coeff_match = match(decision_var,coeff_var);
        dcoeff_match = match(decision_var,dcoeff_var);
        cnstr = FunctionHandleConstraint(zeros(length(V_coeff),1),zeros(length(V_coeff),1),length(decision_var_inds),@(x) obj.recompV(x,coeff_match,coeff_power,coeff_M,dcoeff_match,dcoeff_power,dcoeff_M));
        [iCfun,jCvar] = find(sparse_pattern);
        cnstr = cnstr.setSparseStructure(iCfun,jCvar);
        name = repmat({sprintf('sos[%d]',i)},cnstr.num_cnstr,1);
        cnstr = cnstr.setName(name);
        obj = obj.addConstraint(cnstr,decision_var_inds);
      end
    end
    
    function [c,dc] = recompV(obj,x,coeff_match,coeff_power,coeff_M,dcoeff_match,dcoeff_power,dcoeff_M)
      c = coeff_M*prod(bsxfun(@times,x(coeff_match)',ones(size(coeff_power,1),1)).^coeff_power,2);
      dc = reshape(dcoeff_M*prod(bsxfun(@times,x(dcoeff_match)',ones(size(dcoeff_power,1),1)).^dcoeff_power,2),length(c),length(x));
    end
  end
end