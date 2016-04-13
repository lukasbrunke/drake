classdef FixedContactsComDynamicsFullKinematicsPlanner < ContactWrenchSetDynamicsFullKineamticsPlanner
  % This planner fixes the contact locations, and compute the contact
  % wrench set at each knot. The goal is to maximize the margin in the
  % contact wrench set
  properties(SetAccess = protected)
    % The CWS is described as Ain_cws{i}*w<=bin_cws{i} and
    % Aeq_cws{i}*w=beq_cws{i}
    Ain_cws % A nT x 1 cell
    bin_cws % A nT x 1 cell
    Aeq_cws % A nT x 1 cell
    beq_cws % A nT x 1 cell
    % An alternative way to describe the contact wrench set is w belongs to
    % convexcone(cws_ray{i}) + convexhull(cws_vert{i});
    cws_ray % A nT x 1 cell
    cws_vert % A nT x 1 cell
    
    Qw % A 6 x 6 matrix, w'*Qw*w<=1 is the unit ball in the wrench space for wrench disturbance Qw
    
    num_fc_pts % A obj.N x 1 vecotr
    fc_contact_pos % A obj.N x 1 cell, fc_contact_pos{i} is a 3 x num_fc_pts(i) matrix
    fc_axis % A obj.N x 1 cell, fc_axis{i} is a 3 x num_fc_pts(i) matrix
    fc_mu % A obj.N x 1 cell, fc_mu{i} is a 1 x num_fc_pts(i) vector
    fc_edges % A obj.N x 1 cell, fc_edges{i} is a num_fc_pts(i) x 1 cell, fc_edges{i}{j} is a 3 x num_fc_edges matrix
    
    num_grasp_pts % A obj.N x 1 vector
    grasp_pos % A obj.N x 1 cell, grasp_contact_pos{i} is a 3 x num_grasp_pos(i) matrix
    num_grasp_wrench_vert % A obj.N x 1 cell
    grasp_wrench_vert % A obj.N x 1 cell, grasp_wrench_vert{i} is a num_grasp_pos(i) x 1 cell, grasp_wrench_vert{i}{j} is a 6 x num_grasp_wrench_vert{i}(j) matrix
  end
  
  properties(Access = protected)
    Qw_inv;
  end
  
  methods
    function obj = FixedContactsComDynamicsFullKinematicsPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,options)
      % @param contact_wrench_struct  A cell of of structs, with fields
      % 'active_knot', 'cw' and 'contact_pos', where 'cw' fields contain the
      % RigidBodyContactWrench objects
      if(nargin<11)
        options = struct();
      end
      obj = obj@ContactWrenchSetDynamicsFullKineamticsPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,options);
      
      obj = obj.parseContactWrenchStruct(contact_wrench_struct);
      
      sizecheck(Qw,[6,6]);
      Qw = (Qw+Qw')/2;
      if(any(eig(Qw)<=0))
        error('Qw should be positive definite');
      end
      obj.Qw = Qw;
      obj.Qw_inv = inv(obj.Qw);
      
      obj = obj.addCWSconstraint();
      
      obj = obj.setSolverOptions('snopt','majoroptimalitytolerance',1e-5);
      obj = obj.setSolverOptions('snopt','superbasicslimit',2000);
      obj = obj.setSolverOptions('snopt','majorfeasibilitytolerance',1e-6);
      obj = obj.setSolverOptions('snopt','iterationslimit',1e6);
      obj = obj.setSolverOptions('snopt','majoriterationslimit',500);
    end
    
    function obj = addRunningCost(obj,running_cost_function)
    end
    
    function sol = retrieveSolution(obj,x_sol)
      sol = retrieveSolution@ContactWrenchSetDynamicsFullKineamticsPlanner(obj,x_sol);
      sol.friction_cones = cell(obj.N,1);
      sol.num_fc_pts = obj.num_fc_pts;
      sol.num_grasp_pts = obj.num_grasp_pts;
      sol.num_grasp_wrench_vert = obj.num_grasp_wrench_vert;
      sol.grasp_pos = cell(obj.N,1);
      sol.gresp_wrench_vert = cell(obj.N,1);
      for i = 1:obj.N
        sol.friction_cones{i} = LinearizedFrictionCone.empty(obj.num_fc_pts(i),0);
        for j = 1:obj.num_fc_pts(i)
          sol.friction_cones{i}(j) = LinearizedFrictionCone(obj.fc_contact_pos{i}(:,j),obj.fc_axis{i}(:,j),obj.fc_mu{i}(j),obj.fc_edges{i}{j});
        end
        sol.grasp_pos{i} = obj.grasp_pos{i};
        sol.grasp_wrench_vert{i} = obj.grasp_wrench_vert{i};
      end
    end
    function checkSolution(obj,sol)
      checkSolution@ContactWrenchSetDynamicsFullKineamticsPlanner(obj,sol);
      mg = [0;0;-obj.robot_mass*obj.gravity];
      wrench_gravity = [repmat(mg,1,obj.N);cross(sol.com,repmat(mg,1,obj.N))];
      cws_margin = -inf(obj.N,1);
      for i = 1:obj.N
        if(~isempty(obj.Aeq_cws{i}))
          valuecheck(obj.Aeq_cws{i}*(sol.momentum_dot-wrench_gravity),obj.beq_cws{i},1e-4);
        end
        if(~isempty(obj.Ain_cws{i}))
          cws_margin(i) = min(obj.bin_cws{i}-obj.Ain_cws{i}*(sol.momentum_dot(:,i)-wrench_gravity(:,i)));
        end
      end
      cws_margin = min(cws_margin);
      if(cws_margin<-1e-5)
        error('The wrench is not within the contact wrench set');
      end
    end
  end
  
  methods(Access = protected)
    function obj = parseContactWrenchStruct(obj,contact_wrench_struct)
      % parse the contact_wrench_struct, to compute the H representation of
      % the contact wrench set
      num_cw = numel(contact_wrench_struct);
      obj.cws_ray = cell(obj.N,1);
      obj.cws_vert = cell(obj.N,1);
      
      obj.num_fc_pts = zeros(obj.N,1);
      obj.fc_contact_pos = cell(obj.N,1);
      obj.fc_axis = cell(obj.N,1);
      obj.fc_mu = cell(obj.N,1);
      obj.fc_edges = cell(obj.N,1);
      obj.num_grasp_pts = zeros(obj.N,1);
      obj.grasp_pos = cell(obj.N,1);
      obj.num_grasp_wrench_vert = cell(obj.N,1);
      obj.grasp_wrench_vert = cell(obj.N,1);
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
          co_linear_flag = false;
          body_pts_diff = diff(contact_wrench_struct(i).cw.body_pts,2);
          if(size(body_pts_diff,2)<=1 || all(sum(cross(body_pts_diff(:,1:end-1),body_pts_diff(:,2:end)).^2,1)<1e-4))
            co_linear_flag = true;
          end
          if(co_linear_flag)
            cnstr = WorldPositionConstraint(obj.robot,contact_wrench_struct(i).cw.body,contact_wrench_struct(i).cw.body_pts,contact_wrench_struct(i).contact_pos,contact_wrench_struct(i).contact_pos);
            obj = obj.addRigidBodyConstraint(cnstr,num2cell(contact_wrench_struct(i).active_knot));
          else
            T_body = findHomogeneousTransform(contact_wrench_struct(i).cw.body_pts(:,1:3),contact_wrench_struct(i).contact_pos(:,1:3));
            cnstr = WorldPositionConstraint(obj.robot,contact_wrench_struct(i).cw.body,zeros(3,1),T_body(1:3,4),T_body(1:3,4));
            obj = obj.addRigidBodyConstraint(cnstr,num2cell(contact_wrench_struct(i).active_knot));
            cnstr = WorldQuatConstraint(obj.robot,contact_wrench_struct(i).cw.body,rotmat2quat(T_body(1:3,1:3)),0);
            obj = obj.addRigidBodyConstraint(cnstr,num2cell(contact_wrench_struct(i).active_knot));
          end
          obj.num_fc_pts(contact_wrench_struct(i).active_knot) = obj.num_fc_pts(contact_wrench_struct(i).active_knot)+num_pts;
          for j = reshape(contact_wrench_struct(i).active_knot,1,[])
            obj.fc_contact_pos{j} = [obj.fc_contact_pos{j} contact_wrench_struct(i).contact_pos];
            obj.fc_axis{j} = [obj.fc_axis{j} contact_wrench_struct(i).cw.normal_dir];
            obj.fc_mu{j} = [obj.fc_mu{j} contact_wrench_struct(i).cw.mu_face];
            obj.fc_edges{j} = [obj.fc_edges{j} repmat({contact_wrench_struct(i).cw.FC_edge},1,num_pts)];
          end
        elseif(isa(contact_wrench_struct(i).cw,'GraspWrenchPolytope'))
          sizecheck(contact_wrench_struct(i).contact_pos,[3,1]);
          valuecheck(contact_wrench_struct(i).cw.num_pts,1);
          % compute the vertices in the contact wrench set
          wrench_vert = contact_wrench_struct(i).cw.wrench_vert;
          wrench_vert(4:6,:) = cross(repmat(contact_wrench_struct(i).contact_pos,1,contact_wrench_struct(i).cw.num_wrench_vert),wrench_vert(1:3,:),1)+wrench_vert(4:6,:);
          for j = 1:length(contact_wrench_struct(i).active_knot)
            obj.cws_vert{contact_wrench_struct(i).active_knot(j)} = [obj.cws_ray{contact_wrench_struct(i).active_knot(j)} wrench_vert];
          end
          cnstr = WorldPositionConstraint(obj.robot,contact_wrench_struct(i).cw.body,contact_wrench_struct(i).cw.body_pts,contact_wrench_struct(i).contact_pos,contact_wrench_struct(i).contact_pos);
          obj = obj.addRigidBodyConstraint(cnstr,num2cell(contact_wrench_struct(i).active_knot));
          obj.num_grasp_pts(contact_wrench_struct(i).active_knot) = obj.num_grasp_pts(contact_wrench_struct(i).active_knot)+1;
          for j = reshape(contact_wrench_struct(i).active_knot,1,[])
            obj.grasp_pos{j} = [obj.grasp_pos{j} contact_wrench_struct(i).contact_pos];
            obj.num_grasp_wrench_vert{j} = [obj.num_grasp_wrench_vert{j} size(wrench_vert,2)];
            obj.grasp_wrench_vert{j} = [obj.grasp_wrench_vert{j} {wrench_vert}];
          end
        else
          error('Not supported');
        end
      end
      for i = 1:obj.N
        try
          P = Polyhedron('V',obj.cws_vert{i}','R',obj.cws_ray{i}');
          P = P.minHRep();
          obj.Ain_cws{i} = P.H(:,1:6);
          obj.bin_cws{i} = P.H(:,7);
          normalizer = sqrt(sum(obj.Ain_cws{i}.^2,2));
          obj.Ain_cws{i} = obj.Ain_cws{i}./bsxfun(@times,normalizer,ones(1,6));
          obj.bin_cws{i} = obj.bin_cws{i}./normalizer;
          obj.Aeq_cws{i} = P.He(:,1:6);
          obj.beq_cws{i} = P.He(:,7);
        catch
          [obj.Ain_cws{i},obj.bin_cws{i},obj.Aeq_cws{i},obj.beq_cws{i}] = vert2lcon([obj.cws_vert{i}.*bsxfun(@times,ones(6,1),(obj.robot_mass*obj.gravity*50./sqrt(sum(obj.cws_vert{i}.^2,1)))) obj.cws_vert{i}]');
        end
      end
    end
    
    function obj = addCWSconstraint(obj)
      % momentum_dot in the CWS 
      % Ain*(momentum_dot-wg) <bin
      % Aeq*(momentum_dot-wg) = beq
      for i = 1:obj.N
        if(~isempty(obj.Aeq_cws{i}))
          beq = obj.beq_cws{i}-obj.Aeq_cws{i}*[0;0;obj.robot_mass*obj.gravity;zeros(3,1)];
          cnstr = LinearConstraint(beq,beq,[obj.Aeq_cws{i} -obj.Aeq_cws{i}*[zeros(3);crossSkewSymMat([0;0;obj.robot_mass*obj.gravity])]]);
          cnstr = cnstr.setName(repmat({sprintf('CWS constraint[%d]',i)},size(obj.beq_cws{i},1),1));
          obj = obj.addLinearConstraint(cnstr,[obj.world_momentum_dot_inds(:,i);obj.com_inds(:,i)]);
        end
        if(~isempty(obj.Ain_cws{i}))
          bin = obj.bin_cws{i}-obj.Ain_cws{i}*[0;0;obj.robot_mass*obj.gravity;zeros(3,1)];
          cnstr = LinearConstraint(-inf(size(obj.Ain_cws{i},1),1),bin,[obj.Ain_cws{i} -obj.Ain_cws{i}*[zeros(3);crossSkewSymMat([0;0;obj.robot_mass*obj.gravity])]]);
          cnstr = cnstr.setName(repmat({sprintf('CWS constraint[%d]',i)},size(obj.Ain_cws{i},1),1));
          obj = obj.addLinearConstraint(cnstr,[obj.world_momentum_dot_inds(:,i);obj.com_inds(:,i)]);
        end
      end
    end
  end
end