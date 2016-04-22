classdef FixedContactsFixedDisturbanceComDynamicsFullKinematicsPlanner< FixedContactsComDynamicsFullKinematicsPlanner
  % The disturbance position is fixed
  properties(SetAccess = protected)
    disturbance_pos % A 3 x obj.N matrix. The position where the disturbance wrench is applied
  end
  
  methods
    function obj = FixedContactsFixedDisturbanceComDynamicsFullKinematicsPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,disturbance_pos,options)
      if(nargin<12)
        options = struct();
      end
      obj = obj@FixedContactsComDynamicsFullKinematicsPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,Qw,options);
      sizecheck(disturbance_pos,[3,obj.N]);
      obj.disturbance_pos = disturbance_pos;
      
      obj = obj.addCWSMarginConstraint();
    end
    
    function checkSolution(obj,sol)
      checkSolution@FixedContactsComDynamicsFullKinematicsPlanner(obj,sol);
      mg = [0;0;-obj.robot_mass*obj.gravity];
      wrench_gravity = [repmat(mg,1,obj.N);cross(sol.com,repmat(mg,1,obj.N))];
      cws_margin = inf(obj.N,1);
      for i = 1:obj.N
        if(~isempty(obj.Aeq_cws{i}))
          valuecheck(obj.Aeq_cws{i}*(sol.momentum_dot-wrench_gravity),obj.beq_cws{i},1e-4);
        end
        if(~isempty(obj.Ain_cws{i}))
          Tw = [eye(3) zeros(3);crossSkewSymMat(obj.disturbance_pos(:,i)) eye(3)];
          cws_margin(i) = min((obj.bin_cws{i}-obj.Ain_cws{i}*(sol.momentum_dot(:,i)-wrench_gravity(:,i)))./sqrt(sum((obj.Ain_cws{i}*Tw*obj.Qw_inv).*(obj.Ain_cws{i}*Tw),2)));
        end
      end
      cws_margin = min(cws_margin);
      valuecheck(cws_margin,sol.cws_margin,1e-2);
    end
  end
  
  methods
    function obj = addCWSMarginConstraint(obj)
      for i = 1:obj.N
        if(~isempty(obj.Ain_cws{i}))
          bin = obj.bin_cws{i}-obj.Ain_cws{i}*[0;0;obj.robot_mass*obj.gravity;zeros(3,1)];
          T_pw = [eye(3) zeros(3);crossSkewSymMat(obj.disturbance_pos(:,i)) eye(3)];
          cnstr = LinearConstraint(-inf(size(obj.Ain_cws{i},1),1),bin,[obj.Ain_cws{i}*obj.momentum_dot_normalizer -obj.Ain_cws{i}*[zeros(3);crossSkewSymMat([0;0;obj.robot_mass*obj.gravity])] sqrt(sum((obj.Ain_cws{i}*T_pw*obj.Qw_inv).*(obj.Ain_cws{i}*T_pw),2))]);
          cnstr = cnstr.setName(repmat({sprintf('CWS margin constraint[%d]',i)},size(obj.Ain_cws{i},1),1));
          obj = obj.addLinearConstraint(cnstr,[obj.world_momentum_dot_inds(:,i);obj.com_inds(:,i);obj.cws_margin_ind]);
        end
      end
    end
  end
end