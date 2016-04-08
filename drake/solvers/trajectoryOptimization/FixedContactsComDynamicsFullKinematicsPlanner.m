classdef FixedContactsComDynamicsFullKinematicsPlanner < RigidBodyKinematicsPlanner
  % This planner fixes the contact locations, and compute the contact
  % wrench set at each knot. The goal is to maximize the margin in the
  % contact wrench set
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
    
    % The CWS is described as Ain_cws{i}*w<=bin_cws{i} and
    % Aeq_cws{i}*w=beq_cws{i}
    Ain_cws % A nT x 1 cell
    bin_cws % A nT x 1 cell
    Aeq_cws % A nT x 1 cell
    beq_cws % A nT x 1 cell
    cws_margin_ind % A scalar.
  end
  methods
    function obj = FixedContactsComDynamicsFullKinematicsPlanner(robot,N,tf_range,Q_comddot,Qv,Q,q_nom,contact_wrench_struct,options)
      % @param contact_wrench_struct  A cell of of structs, with fields
      % 'active_knot', 'cw' and 'contact_pos', where 'cw' fields contain the
      % RigidBodyContactWrench objects
      if(nargin<9)
        options = struct();
      end
      plant = SimpleDynamicsDummyPlant(robot.getNumPositions());
      obj = obj@RigidBodyKinematicsPlanner(plant,robot,N,tf_range,options);
      
      obj = obj.setSolverOptions('snopt','majoroptimalitytolerance',1e-5);
      obj = obj.setSolverOptions('snopt','superbasicslimit',2000);
      obj = obj.setSolverOptions('snopt','majorfeasibilitytolerance',1e-6);
      obj = obj.setSolverOptions('snopt','iterationslimit',1e5);
      obj = obj.setSolverOptions('snopt','majoriterationslimit',200);
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
    
  end
  
  methods(Access = protected)
    function obj = parseContactWrenchStruct(obj,contact_wrench_struct)
      % parse the contact_wrench_struct, to compute the H representation of
      % the contact wrench set
      num_cw = numel(contact_wrench_struct);
      obj.cws_ray = cell(obj.nT,1);
      obj.cws_vert = cell(obj.nT,1);
      for i = 1:num_cw
        if(~isstruct(contact_wrench_struct) || ~isfield(contact_wrench_struct(i),'active_knot') || ~isfield(contact_wrench_struct(i),'cw') ||...
            ~isfield(contact_wrench_struct(i),'contact_pos') || ~isa(contact_wrench_struct(i).cw,'RigidBodyContactWrench'))
          error('expect a struct with active knot, cw and contact_pos');
        end
        if(isa(contact_wrench_struct(i).cw,'LinearFrictionConeWrench'))
          num_pts = contact_wrench_struct(i).cw.num_pts;
          sizecheck(contact_wrench_struct(i).contact_pos,[3,num_pts]);
          % compute the rays in the contact wrench set
          num_fc_edges = size(contact_wrench_struct(i).cw.FC_edge,2);
          fc_edges_all = repmat(contact_wrench_struct(i).cw.FC_edge,1,num_pts);
          wrench_ray = [fc_edges_all;cross(reshape(repmat(contact_wrench_struct(i).contact_pos,num_fc_edges,1),3,[]),fc_edges_all,1)];
          for j = 1:length(contact_wrench_struct(i).active_knot)
            obj.cws_ray{contact_wrench_struct(i).active_knot(j)} = [obj.cws_ray{contact_wrench_struct(i).active_knot(j)} wrench_ray];
          end
          % add the kinematic constraint that the body should reach those
          % contact positions
          cnstr = WorldPositionConstraint(obj.robot,contact_wrench_struct(i).cw.body,contact_wrench_struct(i).cw.body_pts,contact_wrench_struct(i).contact_pos,contact_wrench_struct(i).contact_pos);
          obj = obj.addRigidBodyConstraint(cnstr,num2cell(contact_wrench_struct(i).active_knot));
        elseif(isa(contact_wrench_struct(i).cw,'GraspWrenchPolytope'))
          sizecheck(contact_wrench_struct(i).contact_pos,[3,1]);
          valuecheck(contact_wrench_struct(i).cw.num_pts,1);
          % compute the vertices in the contact wrench set
          wrench_vert = contact_wrench_struct(i).cw.wrench_vert;
          wrench_vert(4:6,:) = cross(repmat(contact_wrench_struct(i).contact_pos,1,contact_wrench_struct(i).cw.num_wrench_vert),wrench_vert(1:3,:),1)+wrench_vert(4:6,:);
          for j = 1:length(contact_wrench_struct(i).active_knot)
            obj.cws_vert{contact_wrench_struct(i).active_knot(j)} = [obj.cws_ray{contact_wrench_struct(i).active_knot(j)} wrench_vert];
          end
        else
          error('Not supported');
        end
      end
      for i = 1:obj.nT
        P = Polyhedron('V',obj.cws_vert','R',obj.cws_ray');
        P = P.minHRep();
        obj.Ain_cws{i} = P.H(:,1:6);
        obj.bin_cws{i} = P.H(:,7);
        obj.Aeq_cws{i} = P.He(:,1:6);
        obj.beq_cws{i} = P.He(:,7);
      end
    end
  end
end