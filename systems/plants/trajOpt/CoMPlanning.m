classdef CoMPlanning < NonlinearProgramWKinsol
  % solve IK
%   min_q sum_i
%   qdd(:,i)'*Qa*qdd(:,i)+qd(:,i)'*Qv*qd(:,i)+(q(:,i)-q_nom(:,i))'*Q*(q(:,i)-q_nom(:,i))]+additional_cost1+additional_cost2+...
%   subject to
%          constraint1 at t_samples(i)
%          constraint2 at t_samples(i)
%          ...
%          constraint(k)   at [t_samples(2) t_samples(3) ... t_samples(nT)]
%          constraint(k+1) at [t_samples(2) t_samples(3) ... t_samples(nT)]
%          m*comddot(i) = sum_j F_j
%          comdot(i) = 
%   ....
%
% using q_seed_traj as the initial guess. q(1) would be fixed to
% q_seed_traj.eval(t(1))
% @param robot    -- A RigidBodyManipulator or a TimeSteppingRigidBodyManipulator
% @param t_knot   -- A 1 x nT double vector. The t_knot(i) is the time of the i'th knot
% point
% @param Q        -- The matrix that penalizes the posture error
% @param q_nom_traj    -- The nominal posture trajectory
% @param Qv       -- The matrix that penalizes the velocity
% @param Qa       -- The matrix that penalizes the acceleration
% @param rgc      -- A cell of RigidBodyConstraint
% @param x_name   -- A string cell, x_name{i} is the name of the i'th decision variable
% @param fix_initial_state   -- A boolean flag. Set to true if the initial q(:,1) is hold
% constant at q_seed_traj.eval(t_knot(1)) and qdot(:,1) =
% q_seed_traj.fnder(1).eval(t_knot(1)). Otherwise the initial state is free to change. The default value is false.
% @param q_idx    -- a nq x nT matrix. q(:,i) = x(q_idx(:,i))
% @param qd0_idx  -- a nq x 1 matrix. qdot0 = x(qd0_idx);
% @param qdf_idx  -- a nq x 1 matrix. qdotf = x(qdf_idx);
% @param qsc_weight_idx  -- a cell of vectors. x(qsc_weight_idx{i}) are the weights of the QuasiStaticConstraint at time t(i) 
% @param com_idx      -- a 3 x nT matrix. com(:,i) = x(com_idx(:,i))
% @param comdot_idx   -- a 3 x nT matrix. comdot(:,i) = x(comdot_idx(:,i))
% @param comddot_idx  -- a 3 x nT matrix. comddot(:,i) = x(comddot_idx(:,i))
% @param contact_wrench_cnstr     -- A 1 x nT cell. contact_wrench_cnstr{i} is a cell of all the ContactWrenchConstraint being active at time t_knot(i)
% @param contact_F_idx    -- A 1 x nT cell. x(contact_F_idx{i}{j}) are the F variable used
% in evaluating contact_wrench_cnstr{i}{j}

  properties(SetAccess = protected)
    t_knot
    Q
    q_nom_traj
    Qv
    Qa
    rgc
    fix_initial_state
    q_idx
    qd0_idx
    qdf_idx
    qsc_weight_idx
    com_idx
    comdot_idx
    comddot_idx
    contact_force_idx
    contact_F_idx
    contact_wrench_cnstr
    g
  end
  
  properties(Access = protected)
    q_nom
    cpe
  end
  
  methods
    function obj = CoMPlanning(robot,t,q_nom_traj,fix_initial_state,x0,varargin)
      % obj =
      % InverseKinTraj(robot,t,q_nom_traj,RigidBodyConstraint1,RigidBodyConstraint2,...,RigidBodyConstraintN)
      % @param robot    -- A RigidBodyManipulator or a TimeSteppingRigidBodyManipulator
      % @param t   -- A 1 x nT double vector. t(i) is the time of the i'th knot
      % point
      % @param q_nom_traj    -- The nominal posture trajectory
      % @param fix_initial_state    -- A boolean flag, true if the [q(:,1);qdot(:,q)] is
      % fixed to x0, and thus not a decision variable
      % @param x0                 -- A 2*obj.nq x 1 vector. If fix_initial_state = true,
      % then the initial state if fixed to x0, otherwise it is not used.
      % @param RigidBodyConstraint_i    -- A RigidBodyConstraint object
      t = unique(t(:)');
      obj = obj@NonlinearProgramWKinsol(robot,length(t));
      obj.t_knot = t;
      qd_name = cell(2*obj.nq,1);
      for i = 1:obj.nq
        qd_name{i} = sprintf('qd%d[1]',i);
        qd_name{i+obj.nq} = sprintf('qd%d[%d]',i,obj.nT);
      end
      obj = obj.addDecisionVariable(2*obj.nq,qd_name);
      obj.q_idx = reshape((1:obj.nq*obj.nT),obj.nq,obj.nT);
      obj.qd0_idx = obj.nq*obj.nT+(1:obj.nq)';
      obj.qdf_idx = obj.nq*(obj.nT+1)+(1:obj.nq)';
      com_name = cell(9*obj.nT,1);
      for i = 1:obj.nT
        com_name{9*(i-1)+1} = sprintf('com_x[%d]',i);
        com_name{9*(i-1)+2} = sprintf('com_y[%d]',i);
        com_name{9*(i-1)+3} = sprintf('com_z[%d]',i);
        com_name{9*(i-1)+4} = sprintf('comdot_x[%d]',i);
        com_name{9*(i-1)+5} = sprintf('comdot_y[%d]',i);
        com_name{9*(i-1)+6} = sprintf('comdot_z[%d]',i);
        com_name{9*(i-1)+7} = sprintf('comddot_x[%d]',i);
        com_name{9*(i-1)+8} = sprintf('comddot_y[%d]',i);
        com_name{9*(i-1)+9} = sprintf('comddot_z[%d]',i);
      end
      com_idx_tmp = bsxfun(@plus,(1:9)',9*(0:obj.nT-1));
      obj.com_idx = obj.num_vars+com_idx_tmp(1:3,:);
      obj.comdot_idx = obj.num_vars+com_idx_tmp(4:6,:);
      obj.comddot_idx = obj.num_vars+com_idx_tmp(7:9,:);
      obj = obj.addDecisionVariable(9*obj.nT,com_name);
      % suppose euler integration for com and comdot
      for i = 2:obj.nT
        euler_com = LinearConstraint(zeros(3,1),zeros(3,1),[eye(3) -eye(3) (obj.t_knot(i)-obj.t_knot(i-1))*eye(3)]);
        euler_com = euler_com.setName({sprintf('com_euler x at t[%d],t[%d]',i,i-1);...
          sprintf('com_euler y at t[%d],t[%d]',i,i-1);sprintf('com_euler z at t[%d],t[%d]',i,i-1)});
        obj = obj.addLinearConstraint(euler_com,[obj.com_idx(:,i);obj.com_idx(:,i-1);obj.comdot_idx(:,i)]);
        euler_comdot = LinearConstraint(zeros(3,1),zeros(3,1),[eye(3) -eye(3) (obj.t_knot(i)-obj.t_knot(i-1))*eye(3)]);
        euler_comdot = euler_comdot.setName({sprintf('comdot_euler x at t[%d],t[%d]',i,i-1);...
          sprintf('comdot_euler y at t[%d],t[%d]',i,i-1);sprintf('comdot_euler z at t[%d],t[%d]',i,i-1)});
        obj = obj.addLinearConstraint(euler_comdot,[obj.comdot_idx(:,i);obj.comdot_idx(:,i-1);obj.comddot_idx(:,i)]);
      end
      if(~isa(q_nom_traj,'Trajectory'))
        error('Drake:InverseKinTraj:q_nom_traj should be a trajectory');
      end
      obj.q_nom_traj = q_nom_traj;
      obj.q_nom = obj.q_nom_traj.eval(obj.t_knot);
      
      sizecheck(fix_initial_state,[1,1]);
      obj = obj.setFixInitialState(fix_initial_state,x0);
      obj.qsc_weight_idx = cell(1,obj.nT);
      num_rbcnstr = nargin-5;
      [q_lb,q_ub] = obj.robot.getJointLimits();
      obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint(reshape(bsxfun(@times,q_lb,ones(1,obj.nT)),[],1),...
        reshape(bsxfun(@times,q_ub,ones(1,obj.nT)),[],1)),obj.q_idx(:));
      if(obj.fix_initial_state)
        t_start = 2;
      else
        t_start = 1;
      end
      for i = t_start:obj.nT
        com_match_cnstr = CoMMatchConstraint(obj.robot);
        com_match_cnstr = com_match_cnstr.setName({sprintf('CoM x at t[%d]',i);sprintf('CoM y at t[%d]',i);sprintf('CoM z at t[%d]',i)});
        obj = obj.addNonlinearConstraint(com_match_cnstr,i,obj.com_idx(:,i),[obj.q_idx(:,i);obj.com_idx(:,i)]);
      end
      obj.contact_wrench_cnstr = cell(1,obj.nT);
      obj.contact_F_idx = cell(1,obj.nT);
      for i = 1:num_rbcnstr
        if(~isa(varargin{i},'RigidBodyConstraint'))
          error('Drake:InverseKinTraj:the input should be a RigidBodyConstraint');
        end
        if(isa(varargin{i},'SingleTimeKinematicConstraint'))
          for j = t_start:obj.nT
            if(varargin{i}.isTimeValid(obj.t_knot(j)))
              cnstr = varargin{i}.generateConstraint(obj.t_knot(j));
              obj = obj.addNonlinearConstraint(cnstr{1},j,[],obj.q_idx(:,j));
            end
          end
        elseif(isa(varargin{i},'PostureConstraint'))
          for j = t_start:obj.nT
            if(varargin{i}.isTimeValid(obj.t_knot(j)))
              cnstr = varargin{i}.generateConstraint(obj.t_knot(j));
              obj = obj.addBoundingBoxConstraint(cnstr{1},obj.q_idx(:,j));
            end
          end
        elseif(isa(varargin{i},'QuasiStaticConstraint'))
          for j = t_start:obj.nT
            if(varargin{i}.isTimeValid(obj.t_knot(j)) && varargin{i}.active)
              if(~isempty(obj.qsc_weight_idx{j}))
                error('Drake:InverseKinTraj:currently only support at most one QuasiStaticConstraint at an individual time');
              end
              cnstr = varargin{i}.generateConstraint(obj.t_knot(j));
              qsc_weight_names = cell(varargin{i}.num_pts,1);
              for k = 1:varargin{i}.num_pts
                qsc_weight_names{k} = sprintf('qsc_weight%d',k);
              end
              obj.qsc_weight_idx{j} = obj.num_vars+(1:varargin{i}.num_pts)';
              obj = obj.addDecisionVariable(varargin{i}.num_pts,qsc_weight_names);
              obj = obj.addNonlinearConstraint(cnstr{1},j,obj.qsc_weight_idx{j},[obj.q_idx(:,j);obj.qsc_weight_idx{j}]);
            end
          end
        elseif(isa(varargin{i},'SingleTimeLinearPostureConstraint'))
          for j = t_start:obj.nT
            if(varargin{i}.isTimeValid(obj.t_knot(j)))
              cnstr = varargin{i}.generateConstraint(obj.t_knot(j));
              obj = obj.addLinearConstraint(cnstr{1},obj.q_idx(:,j));
            end
          end
        elseif(isa(varargin{i},'MultipleTimeKinematicConstraint'))
          valid_t_flag = varargin{i}.isTimeValid(obj.t_knot(t_start:end));
          t_idx = (t_start:obj.nT);
          valid_t_idx = t_idx(valid_t_flag);
          cnstr = varargin{i}.generateConstraint(obj.t_knot(valid_t_idx));
          obj = obj.addNonlinearConstraint(cnstr{1},valid_t_idx,[],reshape(obj.q_idx(:,valid_t_idx),[],1));
        elseif(isa(varargin{i},'MultipleTimeLinearPostureConstraint'))
          cnstr = varargin{i}.generateConstraint(obj.t_knot(t_start:end));
          obj = obj.addLinearConstraint(cnstr{1},reshape(obj.q_idx(:,t_start:end),[],1));
        elseif(isa(varargin{i},'ContactWrenchConstraint'))
          for j = 1:obj.nT
            if(varargin{i}.isTimeValid(obj.t_knot(j)))
              num_F = prod(varargin{i}.F_size);
              if(isempty(obj.contact_wrench_cnstr{j}))
                obj.contact_wrench_cnstr{j} = varargin(i);
                obj.contact_F_idx{j} = {obj.num_vars+(1:num_F)'};
              else
                obj.contact_wrench_cnstr{j} = [obj.contact_wrench_cnstr{j},varargin(i)];
                obj.contact_F_idx{j} = [obj.contact_F_idx{j};{obj.num_vars+(1:num_F)'}];
              end
              F_name = varargin{i}.forceParamName(obj.t_knot(j));
              obj = obj.addDecisionVariable(num_F,F_name);
              cnstr = varargin{i}.generateConstraint(obj.t_knot(j));
              obj = obj.addNonlinearConstraint(cnstr{1},j,obj.contact_F_idx{j}{end},[obj.q_idx(:,j);obj.contact_F_idx{j}{end}]);
              obj = obj.addBoundingBoxConstraint(cnstr{2},obj.contact_F_idx{j}{end});
            end
          end
        else
          error('Drake:CoMPlanning:Unsupported RigidBodyConstraint');
        end
      end
      obj.g = 9.8;
      m = obj.robot.getMass();
      for i = 1:obj.nT
        % add the linear constraint that the acceleration matches with total forces
        F_idx_i = cell2mat(obj.contact_F_idx{i});
        A_accel = zeros(3,3+length(F_idx_i));
        A_accel(:,1:3) = m*eye(3);
        A_accel_row_start = 3;
        for j = 1:length(obj.contact_wrench_cnstr{i})
          A_accel(:,A_accel_row_start+(1:length(obj.contact_F_idx{i}{j}))) = ...
            -obj.contact_wrench_cnstr{i}{j}.force(obj.t_knot(i));
          A_accel_row_start = A_accel_row_start+length(obj.contact_F_idx{i}{j});
        end
        accel_bnd = -m*[0;0;obj.g];
        obj = obj.addLinearConstraint(LinearConstraint(accel_bnd,accel_bnd,A_accel),[obj.comddot_idx(:,i);F_idx_i]);
        % add the nonlinear constraint that the torque at the CoM is zero
        sctc_i = SimpleCentroidalTorqueConstraint(obj.t_knot(i),obj.nq,length(F_idx_i),obj.contact_wrench_cnstr{i}{:});
        obj = obj.addNonlinearConstraint(sctc_i,i,F_idx_i,[obj.q_idx(:,i);F_idx_i]);
      end
      obj.Q = eye(obj.nq);
      obj.Qv = 0*eye(obj.nq);
      obj.Qa = 1e-3*eye(obj.nq);
      obj = obj.setCubicPostureError(obj.Q,obj.Qv,obj.Qa);
      obj = obj.setSolverOptions('snopt','majoroptimalitytolerance',1e-4);
      obj = obj.setSolverOptions('snopt','superbasicslimit',2000);
      obj = obj.setSolverOptions('snopt','majorfeasibilitytolerance',1e-6);
      obj = obj.setSolverOptions('snopt','iterationslimit',10000);
      obj = obj.setSolverOptions('snopt','majoriterationslimit',200);
    end
    
    function obj = setCubicPostureError(obj,Q,Qv,Qa)
      % set the cost sum_i qdd(:,i)'*Qa*qdd(:,i)+qd(:,i)'*Qv*qd(:,i)+(q(:,i)-q_nom(:,i))'*Q*(q(:,i)-q_nom(:,i))]
      obj.Q = (Q+Q')/2;
      obj.Qv = (Qv+Qv')/2;
      obj.Qa = (Qa+Qa')/2;
      obj.cpe = CubicPostureError(obj.t_knot,obj.Q,obj.q_nom,obj.Qv,obj.Qa);
      if(isempty(obj.cost))
        xind = [obj.q_idx(:);obj.qd0_idx;obj.qdf_idx];
        obj = obj.addCost(obj.cpe,[],xind,xind);
      else
        xind = [obj.q_idx(:);obj.qd0_idx;obj.qdf_idx];
        obj = obj.replaceCost(obj.cpe,1,[],xind,xind);
      end
    end
    
    function obj = setFixInitialState(obj,flag,x0)
      % set obj.fix_initial_state = flag. If flag = true, then fix the initial state to x0
      % @param x0   A 2*obj.nq x 1 double vector. x0 = [q0;qdot0]. The initial state
      sizecheck(flag,[1,1]);
      flag = logical(flag);
      if(isempty(obj.bbcon))
        obj.fix_initial_state = flag;
        if(obj.fix_initial_state)
          q0 = x0(1:obj.nq);
          kinsol = obj.robot.doKinematics(q0,false,false);
          [com0,J] = obj.robot.getCOM(kinsol);
          comdot0 = J*x0(obj.nq+1:end);
          obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint([x0;com0;comdot0],[x0;com0;comdot0]),...
            [obj.q_idx(:,1);obj.qd0_idx;obj.com_idx(:,1);obj.comdot_idx(:,1)]);
        else
          obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint(-inf(2*obj.nq+6,1),inf(2*obj.nq+6,1)),...
            [obj.q_idx(:,1);obj.qd0_idx;obj.com_idx(:,1);obj.comdot_idx(:,1)]);
        end
      elseif(obj.fix_initial_state ~= flag)
        obj.fix_initial_state = flag;
        if(obj.fix_initial_state)
          q0 = x0(1:obj.nq);
          kinsol = obj.robot.doKinematics(q0,false,false);
          [com0,J] = obj.robot.getCOM(kinsol);
          comdot0 = J*x0(obj.nq+1:end);
          obj = obj.replaceBoundingBoxConstraint(BoundingBoxConstraint([x0;com0;comdot0],[x0;com0;comdot0]),...
            1,[obj.q_idx(:,1);obj.qd0_idx;obj.com_idx(:,1);obj.comdot_idx(:,1)]);
        else
          obj = obj.replaceBoundingBoxConstraint(BoundingBoxConstraint(-inf(2*obj.nq+6,1),inf(2*obj.nq+6,1)),...
            1,[obj.q_idx(:,1);obj.qd0_idx;obj.com_idx(:,1);obj.comdot_idx(:,1)]);
        end
      end
    end
  end
end