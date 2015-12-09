classdef InverseKinematics < NonlinearProgram
  % solve the inverse kinematics problem
  % min_q 0.5*(q-qnom)'Q(q-qnom)+cost1(q)+cost2(q)+...
  % s.t    lb<= kc(q) <=ub
  %        q_lb<= q <=q_ub
  % where cost1, cost2 are functions defined by user
  % @param robot      -- A RigidBodyManipulator or a TimeSteppingRigidBodyManipulator
  % @param Q          -- A nq x nq double matrix, where nq is the DOF. Penalize the
  % posture error
  % @param q_nom      -- An nq x 1 double vector. The nominal posture.
  % @param rbm_joint_bnd_cnstr_id  The ID of the BoundingBoxConstraint that enforces the
  % posture to be within the joint limits of the RigidBodyManipulator
  properties(SetAccess = protected)
    Q
    q_nom
    q_idx   % q=x(q_idx), the robot posture
    qsc_weight_idx   % qsc_weight = x(qsc_weight_idx), the weight used in QuasiStaticConstraint
    nq
    robot
    kinsol_dataind
    rbm_joint_bnd_cnstr_id
  end
  
  properties(Access = protected)
    pe   % A PostureError object.
    
    polygon_min_dist_cnstr_idx
    polygon_min_dist_c_idx
    polygon_min_dist_d_idx
  end
  
  methods
    function obj = InverseKinematics(robot,q_nom,varargin)
      % InverseKinematics(robot,q_nom,RigidBodyConstraint1,RigidBodyConstraint2,...)
      % @param robot    -- A RigidBodyManipulator or a TimeSteppingRigidBodyManipulator
      % object
      % @param q_nom    -- A nq x 1 double vector. The nominal posture
      % @param RigidBodyConstraint_i  -- A RigidBodyConstraint object. Support
      % SingleTimeKinematicConstraint, PostureConstraint, QuasiStaticConstraint and
      % SingleTimeLinearPostureConstraint
      if(~isa(robot,'RigidBodyManipulator') && ~isa(robot,'TimeSteppingRigidBodyManipulator'))
        error('Drake:InverseKinematics:robot should be a RigidBodyManipulator or a TimeSteppingRigidBodyManipulator');
      end
      nq_tmp = robot.getNumPositions();
      obj = obj@NonlinearProgram(nq_tmp);
      obj.nq = nq_tmp;
      obj.robot = robot;
      obj.x_name = cell(obj.nq,1);
      for j = 1:obj.nq
        obj.x_name{j} = sprintf('q%d',j);
      end

      if(~isnumeric(q_nom))
        error('Drake:InverseKinematics:q_nom should be a numeric vector');
      end
      sizecheck(q_nom,[obj.nq,1]);
      obj.q_nom = q_nom;
      num_rbcnstr = nargin-2;
      t = [];
      obj.q_idx = (1:obj.nq)';
      obj.qsc_weight_idx = [];
      [q_lb,q_ub] = obj.robot.getJointLimits();
      [obj,obj.rbm_joint_bnd_cnstr_id] = obj.addBoundingBoxConstraint(BoundingBoxConstraint(q_lb,q_ub),obj.q_idx);

      [obj,kinsol_dataind] = obj.addSharedDataFunction(@obj.kinematicsData,{obj.q_idx});
      obj.kinsol_dataind = kinsol_dataind;

      for i = 1:num_rbcnstr
        if(~isa(varargin{i},'RigidBodyConstraint'))
          error('Drake:InverseKinematics:the input should be a RigidBodyConstraint');
        end
        if(isa(varargin{i},'SingleTimeKinematicConstraint'))
          cnstr = varargin{i}.generateConstraint(t);
          obj = obj.addConstraint(cnstr{1},obj.q_idx,obj.kinsol_dataind);
        elseif(isa(varargin{i},'PostureConstraint'))
          cnstr = varargin{i}.generateConstraint(t);
          obj = obj.addBoundingBoxConstraint(cnstr{1},obj.q_idx);
        elseif(isa(varargin{i},'QuasiStaticConstraint'))
          if(varargin{i}.active)
            obj.qsc_weight_idx = obj.num_vars+(1:varargin{i}.num_pts)';
            qsc_weight_names = cell(varargin{i}.num_pts,1);
            for j = 1:varargin{i}.num_pts
              qsc_weight_names{j} = sprintf('qsc_weight%d',j);
            end
            obj = obj.addDecisionVariable(varargin{i}.num_pts,qsc_weight_names);
            cnstr = varargin{i}.generateConstraint(t);
            obj = obj.addConstraint(cnstr{1},{obj.q_idx;obj.qsc_weight_idx},obj.kinsol_dataind);
            obj = obj.addLinearConstraint(cnstr{2},obj.qsc_weight_idx);
            obj = obj.addBoundingBoxConstraint(cnstr{3},obj.qsc_weight_idx);
          end
        elseif(isa(varargin{i},'SingleTimeLinearPostureConstraint'))
          cnstr = varargin{i}.generateConstraint(t);
          obj = obj.addLinearConstraint(cnstr{1},obj.q_idx);
        else
          error('Drake:InverseKinematics:the input RigidBodyConstraint is not accepted');
        end
      end
      obj = obj.setQ(eye(obj.nq));
      obj = obj.setSolverOptions('snopt','majoroptimalitytolerance',1e-4);
      obj = obj.setSolverOptions('snopt','superbasicslimit',2000);
      obj = obj.setSolverOptions('snopt','majorfeasibilitytolerance',1e-6);
      obj = obj.setSolverOptions('snopt','iterationslimit',10000);
      obj = obj.setSolverOptions('snopt','majoriterationslimit',200);
    end
    
    function data = kinematicsData(obj,q)
      data = doKinematics(obj.robot,q,false,false);
    end
    
    function obj = setQ(obj,Q)
      % set the Q matrix in the cost function 0.5(q-q_nom)'Q(q-q_nom)
      % @param Q    -- An nq x nq double PSD matrix
      sizecheck(Q,[obj.nq,obj.nq]);
      obj.Q = Q;
      obj.pe = PostureError(obj.Q,obj.q_nom);
      if(isempty(obj.cost))
        obj = obj.addCost(obj.pe,obj.q_idx);
      else
        obj = obj.replaceCost(obj.pe,1,obj.q_idx);
      end
    end

    function obj = setQnom(obj,q_nom)
      obj.q_nom = q_nom;
      obj.pe = PostureError(obj.Q,obj.q_nom);
      if(isempty(obj.cost))
        obj = obj.addCost(obj.pe,obj.q_idx);
      else
        obj = obj.replaceCost(obj.pe,1,obj.q_idx);
      end
    end
    
    function [q,F,info,infeasible_constraint] = solve(obj,q_seed)
      x0 = zeros(obj.num_vars,1);
      x0(obj.q_idx) = q_seed;
      if(~isempty(obj.qsc_weight_idx))
        x0(obj.qsc_weight_idx) = 1/length(obj.qsc_weight_idx);
      end
      x0 = getSeparatingHyperplaneGuess(obj,q_seed,x0);
      [x,F,info,infeasible_constraint] = solve@NonlinearProgram(obj,x0);
      q = x(obj.q_idx);
      q = max([obj.x_lb(obj.q_idx) q],[],2);
      q = min([obj.x_ub(obj.q_idx) q],[],2);
    end
    
    function obj = addPolygonMinDistConstraint(obj,cnstr)
      if(~isa(cnstr,'PolygonMinDistConstraint'))
        error('cnstr should be a PolygonMinDistConstraint object');
      end
      var_name = {'c1';'c2';'c3';'d'};
      [obj,tmp_idx] = obj.addDecisionVariable(4,var_name);
      obj.polygon_min_dist_c_idx = [obj.polygon_min_dist_c_idx tmp_idx(1:3)];
      obj.polygon_min_dist_d_idx = [obj.polygon_min_dist_d_idx tmp_idx(4)];
      obj = obj.addNonlinearConstraint(cnstr,{obj.q_idx;tmp_idx(1:3);tmp_idx(4)},obj.kinsol_dataind);
      obj.polygon_min_dist_cnstr_idx = [obj.polygon_min_dist_cnstr_idx length(obj.nlcon)];
      obj = obj.addBoundingBoxConstraint(BoundingBoxConstraint(-ones(3,1),ones(3,1)),obj.polygon_min_dist_c_idx(:,end));
    end
    
  end
  
  methods(Access = protected)
    function x0 = getSeparatingHyperplaneGuess(obj,q_seed,x0)
      kinsol = obj.robot.doKinematics(q_seed,[],struct('compute_gradients',false));
      for i = 1:length(obj.polygon_min_dist_cnstr_idx)
        nlcon_idx = obj.polygon_min_dist_cnstr_idx(i);
        x1 = obj.robot.forwardKin(kinsol,obj.nlcon{nlcon_idx}.body_pair(1),obj.nlcon{nlcon_idx}.body1_pts);
        x2 = obj.robot.forwardKin(kinsol,obj.nlcon{nlcon_idx}.body_pair(2),obj.nlcon{nlcon_idx}.body2_pts);
        [x0(obj.polygon_min_dist_c_idx(:,i)),x0(obj.polygon_min_dist_d_idx(i))] = getMaxSeparationHyperplane(x1,x2);
      end
    end
  end
end
