classdef CoMPlanning
  % plan the CoM trajectory and contact forces only. Suppose we know the contact position
  % already.
  % @param robot_mass    -- The mass of the robot   
  % @param interpolation_order   -- The order of interpolation for CoM trajectory
  % @param t_knot        -- The time knot points
  % @param nT            -- The total time knot points
  % @param contact_wrench_constr   -- A 1 x nT cell. contact_wrench_constr{i} are all the
  % ContactWrenchConstraint being active at time t_knot(i)
  % @param contact_pos     -- A 1 x nT cell. contact_pos{i}{j} is the contact position for
  % contact_wrench_constr{i}{j}
  % @param g            -- Gravitational acceleration.
  properties(SetAccess = protected)
    robot_mass
    interpolation_order
    t_knot
    nT
    contact_wrench_constr
    contact_pos
    contact_force_rotmat
    g
    com_idx
    comdot_idx
    comddot_idx
    contact_F_idx
    x_name
    A_name
    num_vars
    x_lb
    x_ub
    cones
  end
  
  properties(Access = protected)
    iAfun;
    jAvar;
    Aval;
    b_lb;
    b_ub;
    num_A;
  end
  
  methods
    function obj = CoMPlanning(robot_mass,t_knot,interpolation_order,fix_time,minimize_angular_momentum,varargin)
      % obj =
      % CoMPlanning(robot_mass,t_knot,fix_time,minimize_angular_momentum,...
      %         contact_wrench_constraint1,contact_pos1,...
      %         contact_wrench_constraint2,contact_pos2,...
      %                    ....................
      %         contact_wrench_constraintN,contact_posN)
      % @param robot_mass   -- The mass of the robot
      % @param t_knot       -- The time knot points
      % @param interpolation_order   -- The interpolation order of CoM trajectory. Accept
      % 1 or 2
      % @param fix_time     -- A boolean, True if we are not optimizing over time. False
      % otherwise
      % @param minimize_angular_momentum     -- A boolean. False if only the linear
      % acceleration is considered. True if we consider to minimize the rate of angular
      % momentum.
      % @param conact_wrench_constraint     -- A ContactWrenchConstraint object
      % @param contact_pos                  -- The contact position
      sizecheck(robot_mass,[1,1]);
      if(~isnumeric(robot_mass) || robot_mass<=0)
        error('Drake:CoMPlanning: robot mass should be a positive scalar');
      end
      obj.robot_mass = robot_mass;
      sizecheck(interpolation_order,[1,1]);
      if(interpolation_order ~= 1 && interpolation_order ~= 2)
        error('Drake:CoMPlanning: interpolation order not supported yet');
      end
      obj.interpolation_order = interpolation_order;
      t_knot = reshape(unique(t_knot),1,[]);
      obj.t_knot = t_knot;
      obj.nT = length(t_knot);
      sizecheck(fix_time,[1 1]);
      typecheck(fix_time,'logical');
      sizecheck(minimize_angular_momentum,[1 1]);
      typecheck(minimize_angular_momentum,'logical');
      obj.com_idx = reshape((1:3*obj.nT),3,obj.nT);
      obj.comdot_idx = reshape(3*obj.nT+(1:3*obj.nT),3,obj.nT);
      obj.comddot_idx = reshape(6*obj.nT+(1:3*obj.nT),3,obj.nT);
      obj.contact_wrench_constr = cell(1,obj.nT);
      obj.contact_pos = cell(1,obj.nT);
      obj.contact_F_idx = cell(1,obj.nT);
      obj.contact_force_rotmat = cell(1,obj.nT);
      obj.x_name = cell(9*obj.nT,1);
      for i = 1:obj.nT
        obj.x_name{(i-1)*9+1} = sprintf('com_x[%d]',i);
        obj.x_name{(i-1)*9+2} = sprintf('com_y[%d]',i);
        obj.x_name{(i-1)*9+3} = sprintf('com_z[%d]',i);
        obj.x_name{(i-1)*9+4} = sprintf('comdot_x[%d]',i);
        obj.x_name{(i-1)*9+5} = sprintf('comdot_y[%d]',i);
        obj.x_name{(i-1)*9+6} = sprintf('comdot_z[%d]',i);
        obj.x_name{(i-1)*9+7} = sprintf('comddot_x[%d]',i);
        obj.x_name{(i-1)*9+8} = sprintf('comddot_y[%d]',i);
        obj.x_name{(i-1)*9+9} = sprintf('comddot_z[%d]',i);
      end
      obj.num_vars = 9*obj.nT;
      obj.x_lb = -inf(9*obj.nT,1);
      obj.x_ub = inf(9*obj.nT,1);
      obj.iAfun = [];
      obj.jAvar = [];
      obj.Aval = [];
      obj.num_A = 0;
      obj.b_lb = [];
      obj.b_ub = [];
      obj.A_name = {};
      obj.cones = [];
      for i = 1:length(varargin)/2
        if(~isa(varargin{2*i-1},'ContactWrenchConstraint'))
          error('Drake:CoMPlanning: expect a ContactWrenchConstraint object');
        end
        if(~isnumeric(varargin{2*i}))
          error('Drake:CoMPlanning: expect the contact_pos being numeric');
        end
        sizecheck(varargin{2*i},[3 varargin{2*i-1}.num_contact_pt]);
        for j = 1:obj.nT
          if(isa(varargin{2*i-1},'FrictionConeWrenchConstraint'))
            obj = obj.addFrictionCone(varargin{2*i-1},varargin{2*i},j);
          else
            obj = obj.addContactConstraint(varargin{2*i-1},varargin{2*i},j,eye(prod(varargin{2*i-1}.F_size)));
          end
        end
      end
      % The interpolation of CoM
      if(fix_time)
        dt = reshape(diff(obj.t_knot),1,[]);
        if(obj.interpolation_order == 1)
          iAfun_com = [(1:3*(obj.nT-1))';(1:3*(obj.nT-1))';(1:3*(obj.nT-1))'];
          jAvar_com = [reshape(obj.com_idx(:,2:end),[],1);reshape(obj.com_idx(:,1:end-1),[],1); reshape(obj.comdot_idx(:,2:end),[],1)];
          Aval_com = [ones(3*(obj.nT-1),1); -ones(3*(obj.nT-1),1);-reshape(bsxfun(@times,ones(3,1),dt),[],1)];
          iAfun_com = [iAfun_com;3*(obj.nT-1)+iAfun_com];
          jAvar_com = [jAvar_com;reshape(obj.comdot_idx(:,2:end),[],1);reshape(obj.comdot_idx(:,1:end-1),[],1);reshape(obj.comddot_idx(:,2:end),[],1)];
          Aval_com = [Aval_com;Aval_com];
          com_name = repmat({sprintf('com interpolation')},6*(obj.nT-1),1);
        else
          error('Not implemented yet');
        end
      else
        error('Not implemented yet');
      end
      obj = obj.addLinearConstraint(iAfun_com,jAvar_com,Aval_com,zeros(6*(obj.nT-1),1),zeros(6*(obj.nT-1),1),com_name);
      % Newton law for acceleration
      A_newton = zeros(3*obj.nT,obj.num_vars);
      newton_name = cell(3*obj.nT,1);
      for i = 1:obj.nT
        A_newton(3*(i-1)+(1:3),obj.comddot_idx(:,i)) = obj.robot_mass*eye(3);
        for j = 1:length(obj.contact_wrench_constr{i})
          A_newton(3*(i-1)+(1:3),obj.contact_F_idx{i}{j}) = -obj.contact_wrench_constr{i}{j}.force(obj.t_knot(i))*obj.contact_force_rotmat{i}{j};
        end
        newton_name(3*(i-1)+(1:3)) = repmat({sprintf('newton law for acceleration at %5.2f',obj.t_knot(i))},3,1);
      end
      [iAfun_newton,jAvar_newton,Aval_newton] = find(A_newton);
      
      obj = obj.addLinearConstraint(iAfun_newton,jAvar_newton,Aval_newton,zeros(3*obj.nT,1),zeros(3*obj.nT,1),newton_name);
      % Add constraint on the contact force. 
      for i = 1:obj.nT
        for j = 1:length(obj.contact_wrench_constr{i})
          if(isa(obj.contact_wrench_constr{i},'FrictionConeWrenchConstraint'))
            % add a cone constraint
          end
        end
      end
    end
    
    function obj = setXbounds(obj,lb,ub,xind)
      if(any(ub<lb))
        error('lb should be no larger than ub');
      end
      obj.x_lb(xind) = lb;
      obj.x_ub(xind) = ub;
    end
    
    function [com,comdot,comddot,F,info] = solve(obj)
      A = sparse(obj.iAfun,obj.jAvar,obj.Aval,obj.num_A,obj.num_vars);
      Aeq_idx = obj.b_lb == obj.b_ub;
      Aub_idx = ~isinf(obj.b_ub) & (obj.b_lb ~= obj.b_lb);
      Alb_idx = ~isinf(obj.b_lb) & (obj.b_lb ~= obj.b_ub);
      model.A = sparse([A(Aeq_idx,:);A(Aub_idx,:);A(Alb_idx,:)]);
      model.rhs = [obj.b_ub(Aeq_idx);obj.b_ub(Aub_idx);obj.b_lb(Alb_idx)];
      model.sense = [repmat('=',sum(Aeq_idx),1);repmat('<',sum(Aub_idx),1);repmat('>',sum(Alb_idx),1)];
      if(~isempty(obj.cones))
        model.cones = obj.cones;
      end
      model.lb = obj.x_lb;
      model.ub = obj.x_ub;
      model.obj = zeros(1,obj.num_vars);
      model.modelsense = 'min';
      params = struct();
      checkDependency('gurobi');
      result = gurobi(model,params);
    end
  end
    
  
  methods(Access = protected)
    function obj = addFrictionCone(obj,friction_cone,contact_pos,t_idx)
      % Add a second order cone constraint
      % @param friction_cone   A FrictionConeWrenchConstraint
      if(~isa(friction_cone,'FrictionConeWrenchConstraint'))
        error('Should be a FrictionConeWrenchConstraint object');
      end
      if(friction_cone.isTimeValid(obj.t_knot(t_idx)))
        num_F = prod(friction_cone.F_size);
        force_rot_axis = cross(friction_cone.FC_axis,repmat([0;0;1],1,friction_cone.num_pts));
        force_rotmat = zeros(num_F);
        for k = 1:friction_cone.num_pts
          if(norm(force_rot_axis(:,k))<1e-2)
            force_rotmat(3*(k-1)+(1:3),3*(k-1)+(1:3)) = eye(3);
          else
            force_rot_angle = asin(norm(force_rot_axis(:,k)));
            force_rotmat(3*(k-1)+(1:3),3*(k-1)+(1:3)) = axis2rotmat([force_rot_axis(:,k);-force_rot_angle]);
          end
        end
        obj = obj.addContactConstraint(friction_cone,contact_pos,t_idx,force_rotmat);
        obj.num_vars = obj.num_vars+friction_cone.num_pts;
        obj.x_name = [obj.x_name;repmat({'mu*normal_force'},friction_cone.num_pts,1)];
        obj.x_lb = [obj.x_lb;zeros(friction_cone.num_pts,1)];
        obj.x_ub = [obj.x_ub;inf(friction_cone.num_pts,1)];
        iAfun_mu = reshape(bsxfun(@times,(1:friction_cone.num_pts)',ones(1,2)),[],1);
        F_idx = reshape(obj.contact_F_idx{t_idx}{end},3,[]);
        jAvar_mu = [F_idx(3,:)';(obj.num_vars-friction_cone.num_pts+1:obj.num_vars)'];
        Aval_mu = [ones(friction_cone.num_pts,1);reshape(-friction_cone.FC_mu,[],1)];
        mu_name = repmat({sprintf('mu constraint for cone at %5.2f',obj.t_knot(t_idx))},friction_cone.num_pts,1);
        obj = obj.addLinearConstraint(iAfun_mu,jAvar_mu,Aval_mu,zeros(friction_cone.num_pts,1),zeros(friction_cone.num_pts,1),mu_name);
        for i = 1:friction_cone.num_pts
          if(isempty(obj.cones))
            obj.cones = struct('index',[obj.num_vars-friction_cone.num_pts+i F_idx(1:2,i)']);
          else
            obj.cones = [obj.cones;struct('index',[obj.num_vars-friction_cone.num_pts+i F_idx(1:2,i)'])];
          end
        end
      end
    end
    
    function obj = addContactConstraint(obj,contact_cnstr,contact_pos,t_idx,force_rotmat)
      % @param contact_cnstr   A ContactWrenchConstraint object
      num_F = prod(contact_cnstr.F_size);
      if(isempty(obj.contact_wrench_constr{t_idx}))
        obj.contact_wrench_constr{t_idx} = {contact_cnstr};
        obj.contact_pos{t_idx} = {contact_pos};
        obj.contact_F_idx{t_idx} = {obj.num_vars+(1:num_F)'};
        obj.contact_force_rotmat{t_idx} = {force_rotmat};
      else
        obj.contact_wrench_constr{t_idx} = [obj.contact_wrench_constr{t_idx} {contact_cnstr}];
        obj.contact_pos{t_idx} = [obj.contact_pos{t_idx} {contact_pos}];
        obj.contact_F_idx{t_idx} = [obj.contact_F_idx{t_idx};{obj.num_vars+(1:num_F)'}];
        obj.contact_force_rotmat{t_idx} = [obj.contact_force_rotmat{t_idx},{force_rotmat}];
      end
      obj.num_vars = obj.num_vars+num_F;
      obj.x_name = [obj.x_name;contact_cnstr.forceParamName(obj.t_knot(t_idx))];
      cnstr = contact_cnstr.generateConstraint(obj.t_knot(t_idx));
      obj.x_lb = [obj.x_lb;cnstr{2}.lb];
      obj.x_ub = [obj.x_ub;cnstr{2}.ub];
    end
    
    function obj = addLinearConstraint(obj,iAfun,jAvar,Aval,b_lb,b_ub,constr_name)
      obj.iAfun = [obj.iAfun;iAfun+obj.num_A];
      obj.jAvar = [obj.jAvar;jAvar];
      obj.Aval = [obj.Aval;Aval];
      obj.b_lb = [obj.b_lb;b_lb];
      obj.b_ub = [obj.b_ub;b_ub];
      obj.num_A = obj.num_A+length(b_lb);
      if(length(b_lb) ~= length(constr_name))
        error('linear constraint length does not match');
      end
      if(~iscellstr(constr_name))
        error('should be a cell string');
      end
      obj.A_name = [obj.A_name;constr_name];
    end
  end
end