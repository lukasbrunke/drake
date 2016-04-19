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
    l1_gram_var_inds % A 8 x nT matrix
    l2_gram_var_inds % A nT x 1 cell, l2_gram_var_inds{i} is a 36 x (num_fc_pts * num_fc_edges) matrix
    l3_gram_var_inds % A 36 x nT matrix
    l4_gram_var_inds % A nT x 1 cell, l4_gram_var_inds{i} is a 36 x prod(num_grasp_wrench_vert{i}) matrix
    
    V_gram_var_inds % A 36 x nT matrix
    
    Qw % A 6 x 6 matrix, w'*Qw*w<=1 is the unit ball in the wrench space for wrench disturbance Qw
    
    fc_cw % A cell of FrictionConeWrench
    fc_cw_active_knot % A length(fc_cw) x 1 cell
    grasp_cw % A cell of GraspWrenchPolytope
    grasp_cw_active_knot % A length(grasp_cw) x 1 cell
    fc_cw_pos_inds % A length(fc_cw) x 1 cell, fc_cw_pos_inds{i} is a 3 x fc_cw.num_pts matrix
    grasp_cw_pos_inds % A length(grasp_cw) x 1 cell, grasp_cw_pos_inds{i} is a 3 x grasp_cw.num_pts matrix
    
    cws_margin_sos % A CWSMarginSOSconditionBase object
    
    grasp_wrench_vert % A nT x 1 cell, grasp_wrench_vert{i} is a num_grasp_pts(i) x 1 cell
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
    
    friction_cones % A nT x 1 cell, friction_cones{i} is a num_fc_pts(i) x 1 FrictionCone array
    
    a_indet
    b_indet
    
    sos_cnstr_normalizer;
  end
  
  properties(Access = private)
    % if we do recomp(V_indet,V_power,V_res_coeff), then we get the residue of
    % the sos condition, we want this residue to be 0
    V_res_indet % A 7 x obj.N matrix
    V_res_power % A obj.N x 1 cell
    
    sos_con_id % A obj.N x 1 vector
    
    l0
    l1
    l2
    l3
    l4
    
    l_gram_var_count
    
    l2_bbcon_id
  end
  
  methods
    function obj = SearchContactsComDynamicsFullKinematicsSOSPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,options)
      if(nargin< 11)
        options = struct();
      end
      if(~isfield(options,'use_lin_fc'))
        options.use_lin_fc = true;
      end
      if(~isfield(options,'sos_cnstr_normalizer'))
        options.sos_cnstr_normalizer = robot.getMass()*9.81;
      end
      if(~isfield(options,'num_fc_edges'))
        options.num_fc_edges = 4;
      end
      obj = obj@ContactWrenchSetDynamicsFullKineamticsPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,options);
      obj.use_lin_fc = options.use_lin_fc;
      obj.num_fc_edges = options.num_fc_edges;
      obj.sos_cnstr_normalizer = options.sos_cnstr_normalizer;
      
      obj = obj.parseContactWrenchStruct(contact_wrench_struct);
      sizecheck(Qw,[6,6]);
      Qw = (Qw+Qw')/2;
      if(any(eig(Qw)<=0))
        error('Qw should be positive definite');
      end
      obj.Qw = Qw;
      obj.Qw_inv = inv(obj.Qw);
      
      obj = obj.setSolverOptions('snopt','majorfeasibilitytolerance',1e-6);
      obj = obj.setSolverOptions('snopt','superbasicslimit',1e4);
      obj = obj.setSolverOptions('snopt','majoroptimalitytolerance',3e-4);
      obj = obj.setSolverOptions('snopt','iterationslimit',1e6);
      obj = obj.setSolverOptions('snopt','majoriterationslimit',1e3);
    end
    
    function obj = addRunningCost(obj,running_cost_fun)
    end
    
    function sol = retrieveSolution(obj,x)
      sol = retrieveSolution@ContactWrenchSetDynamicsFullKineamticsPlanner(obj,x);
      sol.friction_cones = obj.friction_cones;
      sol.num_fc_pts = obj.num_fc_pts;
      sol.num_grasp_pts = obj.num_grasp_pts;
      sol.num_grasp_wrench_vert = obj.num_grasp_wrench_vert;
      sol.grasp_pos = cell(obj.N,1);
      sol.grasp_wrench_vert = obj.grasp_wrench_vert;
      for i = 1:obj.N
        for j = 1:obj.num_fc_pts(i)
          sol.friction_cones{i}(j) = sol.friction_cones{i}(j).setContactPos(x(obj.fc_contact_pos_inds{i}(:,j)));
        end
        sol.grasp_pos{i} = reshape(x(obj.grasp_contact_pos_inds{i}),3,obj.num_grasp_pts(i));
      end
      sol.l2 = cell(obj.N,1);
      sol.l4 = cell(obj.N,1);
      sol.l0 = subs(obj.l0,obj.l0_gram_var(:),reshape(x(obj.l0_gram_var_inds),[],1));
      sol.l1 = subs(obj.l1,obj.l1_gram_var(:),reshape(x(obj.l1_gram_var_inds),[],1));
      sol.l3 = subs(obj.l3,obj.l3_gram_var(:),reshape(x(obj.l3_gram_var_inds),[],1));
      sol.V = msspoly.zeros(obj.N,1);
      ab_monomials2 = [obj.a_indet;obj.b_indet;1];
      triu_mask = triu(ones(8))~=0;
      for i = 1:obj.N
        sol.l2{i} = subs(obj.l2{i},obj.l2_gram_var{i}(:),reshape(x(obj.l2_gram_var_inds{i}),[],1));
        sol.l4{i} = subs(obj.l4{i},obj.l4_gram_var{i}(:),reshape(x(obj.l4_gram_var_inds{i}),[],1));
        V_gram_var_val = x(obj.V_gram_var_inds(:,i));
        V_gram = zeros(8);
        V_gram(triu_mask) = V_gram_var_val;
        V_gram = V_gram'*V_gram;
        sol.V(i) = ab_monomials2'*V_gram*ab_monomials2;
      end
    end
    
    function x_guess = getInitialVars(obj,q,v,dt)
      x_guess = getInitialVars@ContactWrenchSetDynamicsFullKineamticsPlanner(obj,q,v,dt);
      kinsol = cell(obj.N,1);
      for i = 1:obj.N
        kinsol{i} = obj.robot.doKinematics(q(:,i),v(:,i),struct('use_mex',false'));
      end
      for i = 1:length(obj.fc_cw)
        fc_contact_pos = obj.robot.forwardKin(kinsol{obj.fc_cw_active_knot{i}(1)},obj.fc_cw{i}.body,obj.fc_cw{i}.body_pts);
        x_guess(obj.fc_cw_pos_inds{i}(:)) = fc_contact_pos(:);
      end
      for i = 1:length(obj.grasp_cw)
        grasp_pos = obj.robot.forwardKin(kinsol{obj.grasp_cw_active_knot{i}(1)},obj.grasp_cw{i}.body,obj.grasp_cw{i}.body_pts);
        x_guess(obj.grasp_cw_pos_inds{i}(:)) = grasp_pos(:);
      end
    end
    
    function obj = fixL0(obj,l0)
      l0_gram_var_val = obj.getL0GramVarVal(l0);
      obj = obj.addConstraint(ConstantConstraint(l0_gram_var_val(:)),obj.l0_gram_var_inds(:));
    end
    
    function obj = fixL1(obj,l1)
      l1_gram_var_val = obj.getL1GramVarVal(l1);
      obj = obj.addConstraint(ConstantConstraint(l1_gram_var_val(:)),obj.l1_gram_var_inds(:));
    end
    
    function obj = fixL2(obj,l2)
      l2_gram_var_val = obj.getL2GramVarVal(l2);
      if(isempty(obj.l2_bbcon_id))
        obj.l2_bbcon_id = zeros(obj.N,1);
        for i = 1:obj.N
          [obj,obj.l2_bbcon_id(i)] = obj.addConstraint(ConstantConstraint(l2_gram_var_val{i}(:)),obj.l2_gram_var_inds{i}(:));
        end
      else
        for i = 1:obj.N
          [obj,obj.l2_bbcon_id(i)] = obj.updateBoundingBoxConstraint(obj.l2_bbcon_id(i),ConstantConstraint(l2_gram_var_val{i}(:)),obj.l2_gram_var_inds{i}(:));
        end
      end
    end
    
    function obj = fixL3(obj,l3)
      l3_gram_var_val = obj.getL3GramVarVal(l3);
      obj = obj.addConstraint(ConstantConstraint(l3_gram_var_val(:)),obj.l3_gram_var_inds(:));
    end
    
    function obj = fixL4(obj,l4)
      l4_gram_var_val = obj.getL4GramVarVal(l4);
      for i = 1:obj.N
        if(~isempty(obj.num_grasp_wrench_vert{i}))
          obj = obj.addConstraint(ConstantConstraint(l4_gram_var_val{i}(:)),obj.l4_gram_var_inds{i}(:));
        end
      end
    end
    
    function x = setL0GramVarVal(obj,x,l0)
      l0_gram_var_val = obj.getL0GramVarVal(l0);
      x(obj.l0_gram_var_inds(:)) = l0_gram_var_val(:);
    end
    
    function x = setL1GramVarVal(obj,x,l1)
      l1_gram_var_val = obj.getL1GramVarVal(l1);
      x(obj.l1_gram_var_inds(:)) = l1_gram_var_val(:);
    end
    
    function x = setL2GramVarVal(obj,x,l2)
      l2_gram_var_val = obj.getL2GramVarVal(l2);
      for i = 1:obj.N
        x(obj.l2_gram_var_inds{i}(:)) = l2_gram_var_val{i}(:);
      end
    end
    
    function x = setL3GramVarVal(obj,x,l3)
      l3_gram_var_val = obj.getL3GramVarVal(l3);
      x(obj.l3_gram_var_inds(:)) = l3_gram_var_val(:);
    end
    
    function x = setL4GramVarVal(obj,x,l4)
      l4_gram_var_val = obj.getL4GramVarVal(l4);
      for i = 1:obj.N
        if(~isempty(obj.num_grasp_wrench_vert{i}))
          x(obj.l4_gram_var_inds{i}(:)) = l4_gram_var_val{i}(:);
        end
      end
    end
    
    function x = setVGramVarVal(obj,x,V)
      V_gram_var_val = obj.getVGramVarVal(V);
      x(obj.V_gram_var_inds(:)) = V_gram_var_val(:);
    end
    
    function V_res = computeSOSconditionFromVar(obj,x)
      V_res = msspoly.zeros(obj.N,1);
      shared_data = obj.evaluateSharedDataFunctions(x);
      for i = 1:obj.N
        [~,cnstr_idx] = obj.isNonlinearConstraintID(obj.sos_con_id(i));
        args = [obj.getArgumentArray(x,obj.nlcon_xind{cnstr_idx});shared_data(obj.nlcon_dataind{cnstr_idx})];
        V_res_coeff = obj.nlcon{cnstr_idx}.eval(args{:})';
        V_res(i) = recomp(obj.V_res_indet(:,i),obj.V_res_power{i},V_res_coeff);
      end
    end
    
    function V_res = computeSOSconditionFromSolution(obj,q,v,dt,momentum_dot,cws_margin,friction_cones,grasp_pos,l0,l1,l2,l3,l4,V)
      x = obj.getInitialVars(q,v,dt);
      x(obj.world_momentum_dot_inds) = momentum_dot(:);
      x(obj.cws_margin_ind) = cws_margin;
      for i = 1:obj.N
        for j = 1:obj.num_fc_pts(i)
          x(obj.fc_contact_pos_inds{i}(:,j)) = friction_cones{i}(j).contact_pos;
        end
        x(obj.grasp_contact_pos_inds{i}) = grasp_pos{i}(:);
      end
      x = obj.setL0GramVarVal(x,l0);
      x = obj.setL1GramVarVal(x,l1);
      x = obj.setL2GramVarVal(x,l2);
      x = obj.setL3GramVarVal(x,l3);
      x = obj.setL4GramVarVal(x,l4);
      x = obj.setVGramVarVal(x,V);
      V_res = obj.computeSOSconditionFromVar(x);
    end
  end
  
  methods(Access = protected)
    function obj = parseContactWrenchStruct(obj,contact_wrench_struct)
      num_cw = numel(contact_wrench_struct);
      obj.num_fc_pts = zeros(obj.N,1);
      obj.num_grasp_pts = zeros(obj.N,1);
      obj.num_grasp_wrench_vert = cell(obj.N,1);
      obj.fc_cw = {};
      obj.fc_cw_active_knot = {};
      obj.grasp_cw = {};
      obj.grasp_cw_active_knot = {};
      obj.fc_contact_pos_inds = cell(obj.N,1);
      obj.grasp_contact_pos_inds = cell(obj.N,1);
      obj.fc_cw_pos_inds = {};
      obj.grasp_cw_pos_inds = {};
      obj.friction_cones = cell(obj.N,1);
      for i = 1:obj.N
        if(obj.use_lin_fc)
          obj.friction_cones{i} = LinearizedFrictionCone.empty(0,0);
        else
          obj.friction_cones{i} = FrictionCone.empty(0,0);
        end
      end
      obj.grasp_wrench_vert = cell(obj.N,1);
      
      for i = 1:num_cw
        if(~isstruct(contact_wrench_struct) || ~isfield(contact_wrench_struct(i),'active_knot')...
            || ~isfield(contact_wrench_struct(i),'cw'))
          error('expect a struct with active_knot and cw fields');
        end
        if(isa(contact_wrench_struct(i).cw,'FrictionConeWrench') || isa(contact_wrench_struct(i).cw,'LinearFrictionConeWrench'))
          obj.fc_cw{end+1} = contact_wrench_struct(i).cw;
          obj.fc_cw_active_knot{end+1} = contact_wrench_struct(i).active_knot;
          num_pts = contact_wrench_struct(i).cw.num_pts;
          obj.num_fc_pts(contact_wrench_struct(i).active_knot) = obj.num_fc_pts(contact_wrench_struct(i).active_knot)+num_pts;
          x_names = repmat({sprintf('%s contact pos',contact_wrench_struct(i).cw.body_name)},3*num_pts,1);
          [obj,tmp_idx] = obj.addDecisionVariable(3*num_pts,x_names);
          obj.fc_cw_pos_inds{end+1} = reshape(tmp_idx,3,num_pts);
          if(obj.use_lin_fc)
            if(~isa(contact_wrench_struct(i).cw,'LinearFrictionConeWrench') || size(contact_wrench_struct(i).cw.FC_edge,2)~=obj.num_fc_edges)
              error('The contact wrench is not a linearized friction cone wrench of %d edges',obj.num_fc_edges);
            end
            friction_cone_i = LinearizedFrictionCone.empty(0,num_pts);
            for j = 1:num_pts
              friction_cone_i(j) = LinearizedFrictionCone(nan(3,1),obj.fc_cw{end}.normal_dir(:,j),obj.fc_cw{end}.mu_face(j),obj.fc_cw{end}.FC_edge);
            end
          else
            error('Not implemented yet');
          end
          for j = reshape(contact_wrench_struct(i).active_knot,1,[])
            obj.fc_contact_pos_inds{j} = [obj.fc_contact_pos_inds{j} obj.fc_cw_pos_inds{end}];
            obj.friction_cones{j}(end+(1:num_pts)) = friction_cone_i;
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
          obj.grasp_cw_active_knot{end+1} = contact_wrench_struct(i).active_knot;
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
      
      if(obj.use_lin_fc)
        obj.cws_margin_sos = CWSMarginSOSconditionLinFC(obj.num_fc_edges,obj.robot_mass,obj.N,obj.Qw,obj.num_fc_pts,obj.num_grasp_pts,obj.num_grasp_wrench_vert);
      else
        obj.cws_margin_sos = CWSMarginSOSconditionNonlinearFC(obj.robot_mass,obj.N,obj.Qw,obj.num_fc_pts,obj.num_grasp_pts,obj.num_grasp_wrench_vert);
      end
      
      total_num_grasp_wrench = 0;
      for i = 1:obj.N
        if(~isempty(obj.num_grasp_wrench_vert{i}))
          total_num_grasp_wrench = total_num_grasp_wrench + prod(obj.num_grasp_wrench_vert{i});
        end
      end
      if(obj.use_lin_fc)
        l_gram_var = msspoly('l',80*obj.N+36*sum(obj.num_fc_pts)*obj.num_fc_edges+36*total_num_grasp_wrench);
      else
        l_gram_var = msspoly('l',80*obj.N+36*sum(obj.num_fc_pts)*2+36*total_num_grasp_wrench);
      end
      obj.a_indet = obj.cws_margin_sos.a_indet;
      obj.b_indet = obj.cws_margin_sos.b_indet;
      ab_monomials1 = [obj.cws_margin_sos.a_indet;obj.cws_margin_sos.b_indet;1];
      triu_mask = triu(ones(8))~=0;

      obj.l_gram_var_count = 0;
      
      obj = obj.addL0(l_gram_var,ab_monomials1,triu_mask);
      
      obj = obj.addL1(l_gram_var,ab_monomials1);
      
      obj = obj.addL2(l_gram_var,ab_monomials1,triu_mask);
      
      obj = obj.addL3(l_gram_var,ab_monomials1,triu_mask);
      
      obj = obj.addL4(l_gram_var,ab_monomials1,triu_mask);

      contact_pos = msspoly('p',3*sum(obj.num_fc_pts)+3*sum(obj.num_grasp_pts));
      contact_pos_count = 0;
      obj.fc_pos_var = cell(obj.N,1);
      obj.grasp_pos_var = cell(obj.N,1);
      for i = 1:obj.N
        obj.fc_pos_var{i} = reshape(contact_pos(contact_pos_count+(1:3*obj.num_fc_pts(i))),3,obj.num_fc_pts(i));
        contact_pos_count = contact_pos_count+3*obj.num_fc_pts(i);
        obj.grasp_pos_var{i} = reshape(contact_pos(contact_pos_count+(1:3*obj.num_grasp_pts(i))),3,obj.num_grasp_pts(i));
        contact_pos_count = contact_pos_count+3*obj.num_grasp_pts(i);
        if(obj.use_lin_fc)
          for j = 1:obj.num_fc_pts(i)
            obj.friction_cones{i}(j) = obj.friction_cones{i}(j).setContactPos(obj.fc_pos_var{i}(:,j));
          end
        end
      end

      V = obj.cws_margin_sos.CWSMarginSOScondition(obj.l0,obj.l1,obj.l2,obj.l3,obj.l4,obj.cws_margin_var,obj.friction_cones,obj.grasp_pos_var,obj.grasp_wrench_vert,obj.disturbance_pos,obj.momentum_dot_var,obj.com_var,zeros(3,obj.N));

      x_name = cell(36*obj.N,1);
      for i = 1:obj.N
        x_name((i-1)*36+(1:36)) = repmat({sprintf('V_gram_var[%d]',i)},36,1);
      end
      [obj,tmp_idx] = obj.addDecisionVariable(36*obj.N,x_name);
      obj.V_gram_var_inds = reshape(tmp_idx,36,obj.N);
      obj.V_gram_var = reshape(msspoly('v',36*obj.N),36,obj.N);
      ab_monomials2 = [obj.cws_margin_sos.a_indet;obj.cws_margin_sos.b_indet;1];
      obj.V_res_indet = msspoly.zeros(7,obj.N);
      obj.V_res_power = cell(obj.N,1);

      obj.sos_con_id = zeros(obj.N,1);
      V_res = msspoly.zeros(obj.N,1);
      for i = 1:obj.N
        V_gram = msspoly.zeros(8,8);
        triu_mask = triu(ones(8))~=0;
        V_gram(triu_mask) = obj.V_gram_var(:,i);
        V_gram = V_gram'*V_gram;
        V_res(i) = V(i)-ab_monomials2'*V_gram*ab_monomials2;
        decision_var = [obj.l0_gram_var(:,i);obj.l1_gram_var(:,i);obj.l2_gram_var{i}(:);obj.l3_gram_var(:,i);obj.l4_gram_var{i}(:);obj.V_gram_var(:,i);obj.momentum_dot_var(:,i);obj.com_var(:,i);obj.fc_pos_var{i}(:);obj.grasp_pos_var{i}(:);obj.cws_margin_var];
        decision_var_inds = [obj.l0_gram_var_inds(:,i);obj.l1_gram_var_inds(:,i);obj.l2_gram_var_inds{i}(:);obj.l3_gram_var_inds(:,i);obj.l4_gram_var_inds{i}(:);obj.V_gram_var_inds(:,i);obj.world_momentum_dot_inds(:,i);obj.com_inds(:,i);obj.fc_contact_pos_inds{i}(:);obj.grasp_contact_pos_inds{i}(:);obj.cws_margin_ind];
        [obj.V_res_indet(:,i),obj.V_res_power{i},V_res_coeff] = decomp(V_res(i),decision_var);
        mtch = match([obj.cws_margin_sos.a_indet;obj.cws_margin_sos.b_indet],obj.V_res_indet(:,i));
        valuecheck(numel(unique(mtch)),7);
        sparse_pattern = zeros(length(V_res_coeff),length(decision_var));
        dV_coeff = diff(V_res_coeff',decision_var);
        for j = 1:length(V_res_coeff)
          coeff_var_j = decomp(V_res_coeff(j));
          match_ij = match(decision_var,coeff_var_j);
          sparse_pattern(j,match_ij) = 1;
        end
        [coeff_var,coeff_power,coeff_M] = decomp(V_res_coeff);
        [dcoeff_var,dcoeff_power,dcoeff_M] = decomp(dV_coeff);
        coeff_M = sparse(coeff_M);
        dcoeff_M = sparse(dcoeff_M);
        coeff_match = match(decision_var,coeff_var);
        dcoeff_match = match(decision_var,dcoeff_var);
        cnstr = FunctionHandleConstraint(zeros(length(V_res_coeff),1),zeros(length(V_res_coeff),1),length(decision_var_inds),@(x) recompVmex(x,coeff_match,coeff_power,coeff_M,dcoeff_match,dcoeff_power,dcoeff_M,obj.sos_cnstr_normalizer));
        [iCfun,jCvar] = find(sparse_pattern);
        cnstr = cnstr.setSparseStructure(iCfun,jCvar);
        name = repmat({sprintf('sos[%d]',i)},cnstr.num_cnstr,1);
        cnstr = cnstr.setName(name);
        [obj,obj.sos_con_id(i)] = obj.addConstraint(cnstr,decision_var_inds);
      end
    end
    
    function [c,dc] = recompV(obj,x,coeff_match,coeff_power,coeff_M,dcoeff_match,dcoeff_power,dcoeff_M)
      x_coeff = bsxfun(@times,x(coeff_match)',ones(size(coeff_power,1),1));
      nonzero_coeff_power = coeff_power~=0;
      x_coeff_power = ones(size(x_coeff));
      x_coeff_power(nonzero_coeff_power) = x_coeff(nonzero_coeff_power).^coeff_power(nonzero_coeff_power);
      c = coeff_M*prod(x_coeff_power,2);
      
      x_dcoeff = bsxfun(@times,x(dcoeff_match)',ones(size(dcoeff_power,1),1));
      nonzero_dcoeff_power = dcoeff_power~=0;
      x_dcoeff_power = ones(size(x_dcoeff));
      x_dcoeff_power(nonzero_dcoeff_power) = x_dcoeff(nonzero_dcoeff_power).^dcoeff_power(nonzero_dcoeff_power);
      dc = reshape(dcoeff_M*prod(x_dcoeff_power,2),length(c),length(x));
    end
    
    function l0_gram_var_val = getL0GramVarVal(obj,l0)
      triu_mask = triu(ones(8))~=0;
      l0_gram_var_val = zeros(36,obj.N);
      for i = 1:obj.N
        Q = double(decompQuadraticPoly(l0(i)-1,[obj.a_indet;obj.b_indet]));
        R = chol(Q+eps*eye(8));
        l0_gram_var_val(:,i) = R(triu_mask);
      end
    end
    
    function l1_gram_var_val = getL1GramVarVal(obj,l1)
      [~,q] = linear(l1,[obj.a_indet;obj.b_indet]);
      l1_gram_var_val = [double(q(:,2:end)) double(q(:,1))]';
    end
    
    function l2_gram_var_val = getL2GramVarVal(obj,l2)
      triu_mask = triu(ones(8))~=0;
      l2_gram_var_val = cell(obj.N,1);
      for i = 1:obj.N
        if(obj.use_lin_fc)
          l2_gram_var_val{i} = zeros(36,obj.num_fc_pts(i)*obj.num_fc_edges);
        end
        for j = 1:obj.num_fc_pts(i)
          if(obj.use_lin_fc)
            for k = 1:obj.num_fc_edges
              Q = double(decompQuadraticPoly(l2{i}(j,k),[obj.a_indet;obj.b_indet]));
              R = chol(Q+eps*eye(8));
              l2_gram_var_val{i}(:,(j-1)*obj.num_fc_edges+k) = R(triu_mask);
            end
          end
        end
      end
    end
    
    function l3_gram_var_val = getL3GramVarVal(obj,l3)
      triu_mask = triu(ones(8))~=0;
      l3_gram_var_val = zeros(36,obj.N);
      for i = 1:obj.N
        Q = double(decompQuadraticPoly(l3(i),[obj.a_indet;obj.b_indet]));
        R = chol(Q+eps*eye(8));
        l3_gram_var_val(:,i) = R(triu_mask);
      end
    end
    
    function l4_gram_var_val = getL4GramVarVal(obj,l4)
      triu_mask = triu(ones(8))~=0;
      l4_gram_var_val = cell(obj.N,1);
      for i = 1:obj.N
        if(~isempty(obj.num_grasp_wrench_vert{i}))
          for j = 1:prod(obj.num_grasp_wrench_vert{i})
            Q = double(decompQuadraticPoly(l4{i}(j),[obj.a_indet;obj.b_indet]));
            R = chol(Q+eps*eye(8));
            l4_gram_var_val{i}(:,j) = R(triu_mask);
          end
        end
      end
    end
    
    function V_gram_var_val = getVGramVarVal(obj,V)
      if(deg(V)>2)
        error('V should be quadratic');
      end
      V_gram_var_val = zeros(36,obj.N);
      triu_mask = triu(ones(8))~=0;
      for i = 1:obj.N
        Q = double(decompQuadraticPoly(V(i),[obj.a_indet;obj.b_indet]));
        R = chol(Q+eps*eye(8));
        V_gram_var_val(:,i) = R(triu_mask);
      end
    end
    
    function obj = addL0(obj,l_gram_var,ab_monomials1,triu_mask)
      x_name0 = cell(36*obj.N,1);
      for i = 1:obj.N
        x_name0((i-1)*36+(1:36)) = repmat({sprintf('l0_gram_var[%d]',i)},36,1);
      end
      [obj,tmp_idx] = obj.addDecisionVariable(36*obj.N,x_name0);
      obj.l0_gram_var_inds = reshape(tmp_idx,36,obj.N);
      obj.l0_gram_var = reshape(l_gram_var(obj.l_gram_var_count+(1:36*obj.N)),36,obj.N);
      l0_gram = cell(obj.N,1);
      obj.l0 = msspoly.zeros(obj.N,1);
      for i = 1:obj.N
        l0_gram{i} = msspoly.zeros(8,8);
        l0_gram{i}(triu_mask) = obj.l0_gram_var(:,i);
        l0_gram{i} = l0_gram{i}'*l0_gram{i};
        obj.l0(i) = ab_monomials1'*l0_gram{i}*ab_monomials1+1;
      end
      obj.l_gram_var_count = obj.l_gram_var_count+36*obj.N;
    end
    
    function obj = addL1(obj,l_gram_var,ab_monomials1)
      x_name1 = cell(8*obj.N,1);
      for i = 1:obj.N
        x_name1((i-1)*8+(1:8)) = repmat({sprintf('l1_gram_var[%d]',i)},8,1);
      end
      [obj,tmp_idx] = obj.addDecisionVariable(8*obj.N,x_name1);
      obj.l1_gram_var_inds = reshape(tmp_idx,8,obj.N);
      obj.l1_gram_var = reshape(l_gram_var(obj.l_gram_var_count+(1:8*obj.N)),8,obj.N);
      obj.l1 = msspoly.zeros(obj.N,1);
      for i = 1:obj.N
        obj.l1(i) = ab_monomials1'*obj.l1_gram_var(:,i);
      end
      obj.l_gram_var_count = obj.l_gram_var_count+8*obj.N;
    end
    
    function obj = addL2(obj,l_gram_var,ab_monomials1,triu_mask)
      obj.l2_gram_var_inds = cell(obj.N,1);
      obj.l2_gram_var = cell(obj.N,1);
      obj.l2 = cell(obj.N,1);
      for i = 1:obj.N
        if(obj.use_lin_fc)
          obj.l2{i} = msspoly.zeros(obj.num_fc_pts(i),obj.num_fc_edges);
          x_name2 = repmat({sprintf('l2_gram_var[%d]',i)},36*obj.num_fc_pts(i)*obj.num_fc_edges,1);
          [obj,tmp_idx] = obj.addDecisionVariable(36*obj.num_fc_pts(i)*obj.num_fc_edges,x_name2);
          obj.l2_gram_var_inds{i} = reshape(tmp_idx,36,obj.num_fc_pts(i)*obj.num_fc_edges);
          obj.l2_gram_var{i} = reshape(l_gram_var(obj.l_gram_var_count+(1:36*obj.num_fc_pts(i)*obj.num_fc_edges)),36,obj.num_fc_pts(i)*obj.num_fc_edges);
          obj.l_gram_var_count = obj.l_gram_var_count+36*obj.num_fc_pts(i)*obj.num_fc_edges;
          for j = 1:obj.num_fc_pts(i)
            for k = 1:obj.num_fc_edges
              l2_gram = msspoly.zeros(8,8);
              l2_gram(triu_mask) = obj.l2_gram_var{i}(:,(j-1)*obj.num_fc_edges+k);
              l2_gram = l2_gram'*l2_gram;
              obj.l2{i}(j,k) = ab_monomials1'*l2_gram*ab_monomials1;
            end
          end
        else
          error('Not implemented yet');
        end
      end
    end
    
    function obj = addL3(obj,l_gram_var,ab_monomials1,triu_mask)
      x_name3 = cell(36*obj.N,1);
      for i = 1:obj.N
        x_name3((i-1)*36+(1:36)) = repmat({sprintf('l3_gram_var[%d]',i)},36,1);
      end
      
      [obj,tmp_idx] = obj.addDecisionVariable(36*obj.N,x_name3);
      obj.l3_gram_var_inds = reshape(tmp_idx,36,obj.N);
      obj.l3_gram_var = reshape(l_gram_var(obj.l_gram_var_count+(1:36*obj.N)),36,obj.N);
      obj.l3 = msspoly.zeros(obj.N,1);
      for i = 1:obj.N
        l3_gram = msspoly.zeros(8,8);
        l3_gram(triu_mask) = obj.l3_gram_var(:,i);
        l3_gram = l3_gram'*l3_gram;
        obj.l3(i) = ab_monomials1'*l3_gram*ab_monomials1;
      end
      obj.l_gram_var_count = obj.l_gram_var_count+36*obj.N;
    end
    
    function obj = addL4(obj,l_gram_var,ab_monomials1,triu_mask)
      obj.l4_gram_var_inds = cell(obj.N,1);
      obj.l4_gram_var = cell(obj.N,1);
      obj.l4 = cell(obj.N,1);
      for i = 1:obj.N
        if(~isempty(obj.num_grasp_wrench_vert{i}))
          num_grasp_wrench_vert_i = prod(obj.num_grasp_wrench_vert{i});
          obj.l4{i} = msspoly.zeros(num_grasp_wrench_vert_i,1);
          x_name4 = repmat({sprintf('l4_gram_var[%d]',i)},36*num_grasp_wrench_vert_i,1);
          [obj,tmp_idx] = obj.addDecisionVariable(36*num_grasp_wrench_vert_i,x_name4);
          obj.l4_gram_var_inds{i} = reshape(tmp_idx,36,num_grasp_wrench_vert_i);
          obj.l4_gram_var{i} = reshape(l_gram_var(obj.l_gram_var_count+(1:36*num_grasp_wrench_vert_i)),36,num_grasp_wrench_vert_i);
          obj.l_gram_var_count = obj.l_gram_var_count+36*num_grasp_wrench_vert_i;
          for j = 1:num_grasp_wrench_vert_i
            l4_gram = msspoly.zeros(8,8);
            l4_gram(triu_mask) = obj.l4_gram_var{i}(:,j);
            l4_gram = l4_gram'*l4_gram;
            obj.l4{i}(j) = ab_monomials1'*l4_gram*ab_monomials1;
          end
        end
      end
    end
  end
end