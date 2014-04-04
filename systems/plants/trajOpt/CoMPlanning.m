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
    g
    com_idx
    comdot_idx
    comddot_idx
    contact_F_idx
    x_name
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
      if(~isnumeric(robot_mass) || robot_mas<=0)
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
      typecheck(t_knot,'logical');
      sizecheck(minimize_angular_momentum,[1 1]);
      typecheck(minimize_angular_momentum,'logical');
      obj.com_idx = reshape((1:3*obj.nT),3,obj.nT);
      obj.comdot_idx = reshape(3*obj.nT+(1:3*obj.nT),3,obj.nT);
      obj.comddot_idx = reshape(6*obj.nT+(1:3*obj.nT),3,obj.nT);
      obj.contact_wrench_constr = cell(1,obj.nT);
      obj.contact_pos = cell(1,obj.nT);
      obj.contact_F_idx = cell(1,obj.nT);
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
      num_vars = 9*obj.nT;
      for i = 1:length(varargin)/2
        if(~isa(varargin{2*i-1},'ContactWrenchConstraint'))
          error('Drake:CoMPlanning: expect a ContactWrenchConstraint object');
        end
        if(~isnumeric(varargin{2*i}))
          error('Drake:CoMPlanning: expect the contact_pos being numeric');
        end
        sizecheck(varargin{2*i},[3 varargin{2*i-1}.num_contact_pt]);
        for j = 1:obj.nT
          if(varargin{2*i-1}.isTimeValid(obj.t_knot(j)))
            if(isempty(obj.contact_wrench_constr{j}))
              obj.contact_wrench_constr{j} = varargin(2*i-1);
              obj.contact_pos{j} = varargin(2*i);
              obj.contact_F_idx{j} = {num_vars+(1:prod(varargin{2*i-1}.F_size))'};
            else
              obj.contact_wrench_constr{j} = [obj.contact_wrench_constr{j} varargin(2*i-1)];
              obj.contact_pos{j} = [obj.contact_pos{j} varargin(2*i)];
              obj.contact_F_idx{j} = [obj.contact_F_idx{j};{num_vars+(1:prod(varargin{2*i-1}.F_size))'}];
            end
            num_vars = num_vars+prod(varargin{2*i}.F_size);
          end
        end
        if(fix_time)
          dt = reshape(diff(obj.t_knot),1,[]);
          if(obj.interpolation_order == 1)
            iAfun = [(1:3*(obj.nT-1))';(1:3*(obj.nT-1))';(1:3*(obj.nT-1))'];
            jAvar = [reshape(obj.com_idx(:,2:end),[],1);reshape(obj.com_idx(:,1:end-1),[],1); reshape(obj.comdot_idx(:,2:end),[],1)];
            Aval = [ones(3*(obj.nT-1),1); -ones(3*(obj.nT-1),1);-reshape(bsxfun(@times,ones(3,1),dt),[],1)];
            iAfun = [iAfun;3*(obj.nT-1)+iAfun];
            jAvar = [jAvar;reshape(obj.comdot_idx(:,2:end),[],1);reshape(obj.comdot_idx(:,1:end-1),[],1);reshape(obj.comddot_idx(:,2:end),[],1)];
            Aval = [Aval;Aval];
            A_com = sparse(iAfun,jAvar,Aval,6*obj.nT,9*obj.nT);
          else
            error('Not implemented yet');
          end
        else
          error('Not implemented yet');
        end
      end
    end
  end
end