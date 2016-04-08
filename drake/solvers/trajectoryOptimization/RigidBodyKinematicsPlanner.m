classdef RigidBodyKinematicsPlanner < DirectTrajectoryOptimization
  properties(SetAccess = protected)
    robot % A RigidBodyManipulator or a TimeSteppingRigidBodyManipulator
    nq % number of positions in state
    nv % number of velocities in state
    q_inds % An nq x obj.N matrix. x(q_inds(:,i)) is the posture at i'th knot
    v_inds % An nv x obj.N matrix. x(v_inds(:,i)) is the velocity at i'th knot
    qsc_weight_inds = {}
    fix_initial_state = false
    gravity
    % N-element vector of indices into the shared_data, where
    % shared_data{kinsol_dataind(i)} is the kinsol for knot point i
    kinsol_dataind
    
    % kinematics cache pointers, one for each knot point
    kinematics_cache_ptrs
    
    robot_mass  % A double scalar.
  end
  
  properties(Access = protected)
    floating_body_idx % robot.getBody(floating_body_idx(i)) is a floating body
  end
  
  methods
    function obj = RigidBodyKinematicsPlanner(plant,robot,N,tf_range,options)
      if(nargin<5)
        options = struct();
      end
      if(~isfield(options,'time_option'))
        options.time_option = 2;
      end
      obj = obj@DirectTrajectoryOptimization(plant,N,tf_range,struct('time_option',options.time_option));
      if(~isa(robot,'RigidBodyManipulator') && ~isa(robot,'TimeSteppingRigidBodyManipulator'))
        error('Drake:SimpleDynamicsFullKinematicsPlanner:expect a RigidBodyManipulator or a TimeSteppingRigidBodyManipulator');
      end
      obj.robot = robot;
      obj.floating_body_idx = [];
      for i = 1:obj.robot.getNumBodies()
        if(obj.robot.getBody(i).floating)
          obj.floating_body_idx(end+1) = i;
        end
      end
      obj.nq = obj.robot.getNumPositions();
      obj.nv = obj.robot.getNumVelocities();
      obj.q_inds = obj.x_inds(1:obj.nq,:);
      obj.v_inds = obj.x_inds(obj.nq+(1:obj.nv),:);
      obj.qsc_weight_inds = cell(1,obj.N);
      obj.gravity = 9.81;
      % create shared data functions to calculate kinematics at the knot
      % points
      [joint_lb,joint_ub] = obj.robot.getJointLimits();
      obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint(reshape(bsxfun(@times,joint_lb,ones(1,obj.N)),[],1),...
        reshape(bsxfun(@times,joint_ub,ones(1,obj.N)),[],1)),obj.q_inds(:));
      obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint(zeros(obj.N-1,1),inf(obj.N-1,1)),obj.h_inds);
      m_kinsol_dataind = zeros(obj.N,1);
      obj.kinematics_cache_ptrs = cell(obj.N, 1);
      for i=1:obj.N,
        obj.kinematics_cache_ptrs{i} = createKinematicsCacheAutoDiffmex(robot.mex_model_ptr, robot.getNumPositions() + robot.getNumVelocities());
        [obj,m_kinsol_dataind(i)] = obj.addSharedDataFunction(@(q) obj.kinematicsData(q, obj.kinematics_cache_ptrs{i}),{obj.q_inds(:,i)});
      end
      obj.kinsol_dataind = m_kinsol_dataind;
      
      obj.robot_mass = obj.robot.getMass();
    end
    
    function data = kinematicsData(obj,q,kinematics_cache_ptr)
      options.compute_gradients = true;
      options.use_mex = true;
      options.kinematics_cache_ptr_to_use = kinematics_cache_ptr;
      data = doKinematics(obj.robot,q,[],options);
    end
    
    function obj = setFixInitialState(obj,flag,x0)
      % set obj.fix_initial_state = flag. If flag = true, then fix the initial state to x0
      % @param x0   A 2*obj.robot.getNumPositions() x 1 double vector. x0 = [q0;qdot0]. The initial state
      sizecheck(flag,[1,1]);
      flag = logical(flag);
      if(isempty(obj.bbcon))
        obj.fix_initial_state = flag;
        if(obj.fix_initial_state)
          obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint(x0,x0),obj.x_inds(:,1));
        else
          obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint(-inf(obj.robot.getNumStates(),1),inf(obj.robot.getNumStates(),1)),obj.x_inds(:,1));
        end
      elseif(obj.fix_initial_state ~= flag)
        obj.fix_initial_state = flag;
        if(obj.fix_initial_state)
          obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint(x0,x0),obj.x_inds(:,1));
        else
          obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint(-inf(obj.robot.getNumStates(),1),inf(obj.robot.getNumStates(),1)),obj.x_inds(:,1));
        end
      end
    end
    
    function obj = addConstraint(obj, constraint, varargin)
      if isa(constraint, 'RigidBodyConstraint')
        obj = addRigidBodyConstraint(obj,constraint, varargin{:});
      else
        obj = addConstraint@DirectTrajectoryOptimization(obj,constraint,varargin{:});
      end
    end
    
    function obj = addKinematicConstraint(obj,constraint,time_index)
      % Add a kinematic constraint that is a function of the state at the
      % specified time or times.
      % @param constraint  a CompositeConstraint
      % @param time_index   a cell array of time indices
      %   ex1., time_index = {1, 2, 3} means the constraint is applied
      %   individually to knot points 1, 2, and 3
      %   ex2,. time_index = {[1 2], [3 4]} means the constraint is applied to knot
      %   points 1 and 2 together (taking the combined state as an argument)
      %   and 3 and 4 together.
      if ~iscell(time_index)
        time_index = num2cell(reshape(time_index,1,[]));
      end
      for j=1:length(time_index),
        kinsol_inds = obj.kinsol_dataind(time_index{j});
        cnstr_inds = mat2cell(obj.q_inds(:,time_index{j}),size(obj.q_inds,1),ones(1,length(time_index{j})));

        % record constraint for posterity
        obj.constraints{end+1}.constraint = constraint;
        obj.constraints{end}.var_inds = cnstr_inds;
        obj.constraints{end}.kinsol_inds = kinsol_inds;
        obj.constraints{end}.time_index = time_index;

        obj = obj.addConstraint(constraint,cnstr_inds,kinsol_inds);
      end
    end
    
    function obj = addRigidBodyConstraint(obj,constraint,time_index)
      % Add a kinematic constraint that is a function of the state at the
      % specified time or times.
      % @param constraint  a RigidBodyConstraint object
      % @param time_index   a cell array of time indices
      %   ex1., time_index = {1, 2, 3} means the constraint is applied
      %   individually to knot points 1, 2, and 3
      %   ex2,. time_index = {[1 2], [3 4]} means the constraint is applied to knot
      %   points 1 and 2 together (taking the combined state as an argument)
      %   and 3 and 4 together.
      typecheck(constraint,'RigidBodyConstraint');
      if ~iscell(time_index)
        if isa(constraint,'MultipleTimeKinematicConstraint')
          % then use { time_index(1), time_index(2), ... } ,
          % aka independent constraints for each time
          time_index = {reshape(time_index,1,[])};
        else
          % then use { time_index(1), time_index(2), ... } ,
          % aka independent constraints for each time
          time_index = num2cell(reshape(time_index,1,[]));
        end
      end
      for j = 1:numel(time_index)
        if isa(constraint,'SingleTimeKinematicConstraint')
          cnstr = constraint.generateConstraint();
          obj = obj.addKinematicConstraint(cnstr{1},time_index(j));
        elseif isa(constraint, 'PostureConstraint')
          cnstr = constraint.generateConstraint();
          obj = obj.addBoundingBoxConstraint(cnstr{1}, ...
            obj.q_inds(:,time_index{j}));
        elseif isa(constraint,'QuasiStaticConstraint')
          cnstr = constraint.generateConstraint();
          if(constraint.active)
            if(~isempty(obj.qsc_weight_inds{time_index{j}}))
              error('Drake:SimpleDynamicsFullKinematicsPlanner', ...
                ['We currently support at most one ' ...
                'QuasiStaticConstraint at an individual time']);
            end
            qsc_weight_names = cell(constraint.num_pts,1);
            for k = 1:constraint.num_pts
              qsc_weight_names{k} = sprintf('qsc_weight%d',k);
            end
            obj.qsc_weight_inds{time_index{j}} = obj.num_vars+(1:constraint.num_pts)';
            obj = obj.addDecisionVariable(constraint.num_pts,qsc_weight_names);
            obj = obj.addConstraint(cnstr{1},{obj.q_inds(:,time_index{j});obj.qsc_weight_inds{time_index{j}}},obj.kinsol_dataind(time_index{j}));
            obj = obj.addConstraint(cnstr{2},obj.qsc_weight_inds{time_index{j}});
            obj = obj.addConstraint(cnstr{3},obj.qsc_weight_inds{time_index{j}});
          end
        elseif isa(constraint,'SingleTimeLinearPostureConstraint')
          cnstr = constraint.generateConstraint();
          obj = obj.addLinearConstraint(cnstr{1},obj.q_inds(:,time_index{j}));
        elseif isa(constraint,'MultipleTimeKinematicConstraint')
          cnstr = constraint.generateConstraint([],numel(time_index{j}));
          if ~isempty(cnstr)
            obj = obj.addKinematicConstraint(cnstr{1},time_index(j));
          end
        else
          id = ['Drake:SimpleDynamicsFullKinematicsPlanner:' ...
            'unknownRBConstraint'];
          msg = ['Constraints of class %s are not currently ' ...
            'supported by %s'];
          error(id,msg,class(constraint),class(obj));
        end
      end
    end
    
  end
end