classdef ContactWrenchSetDynamicsFullKineamticsPlanner < RigidBodyKinematicsPlanner
  properties(SetAccess = protected)
    cws_margin_ind % A scalar.
    com_inds % A 3 x obj.N matrix
    comdot_inds % A 3 x obj.N matrix
    comddot_inds % A 3 x obj.N matrix
    centroidal_momentum_inds % A 3 x obj.N matrix
    world_momentum_dot_inds % A 3 x obj.N matrix
    
    cws_margin_cost % A positive scalar. The cost is cws_margin_cost*cws_margin
    
    quat_correction_slack_inds % A size(q_quat_inds,2) x obj.N-1 matrix. The slack variable for interpolating the quaternion
  end
  
  properties(Access = protected)
    q_quat_inds % A 4 x m matrix, q(q_quat_inds(:,i)) is a quaternion
  end
  
  methods
    function obj = ContactWrenchSetDynamicsFullKineamticsPlanner(robot,N,tf_range,Q_comddot,Qv,Q,cws_margin_cost,q_nom,contact_wrench_struct,options)
      if(nargin<10)
        options = struct();
      end
      plant = SimpleDynamicsDummyPlant(robot.getNumPositions());
      obj = obj@RigidBodyKinematicsPlanner(plant,robot,N,tf_range,options);
      obj = obj.setCWSMarginCost(cws_margin_cost);
      
      obj.q_quat_inds = [];
      for i = 1:obj.robot.getNumBodies()
        if(obj.robot.getBody(i).floating == 2)
          obj.q_quat_inds(:,end+1) = obj.robot.getBody(i).position_num(4:7);
        end
      end
      [obj,obj.cws_margin_ind] = obj.addDecisionVariable(1,{'cws_margin'});
      
      x_name = cell(9*obj.N,1);
      for i = 1:obj.N
        x_name{(i-1)*3+1} = sprintf('com_x[%d]',i);
        x_name{(i-1)*3+2} = sprintf('com_y[%d]',i);
        x_name{(i-1)*3+3} = sprintf('com_z[%d]',i);
        x_name{3*obj.N+(i-1)*3+1} = sprintf('comdot_x[%d]',i);
        x_name{3*obj.N+(i-1)*3+2} = sprintf('comdot_y[%d]',i);
        x_name{3*obj.N+(i-1)*3+3} = sprintf('comdot_z[%d]',i);
        x_name{6*obj.N+(i-1)*3+1} = sprintf('comddot_x[%d]',i);
        x_name{6*obj.N+(i-1)*3+2} = sprintf('comddot_y[%d]',i);
        x_name{6*obj.N+(i-1)*3+3} = sprintf('comddot_z[%d]',i);
      end
      [obj,tmp_idx] = obj.addDecisionVariable(9*obj.N,x_name);
      obj.com_inds = reshape(tmp_idx(1:3*obj.N),3,obj.N);
      obj.comdot_inds = reshape(tmp_idx(3*obj.N+(1:3*obj.N)),3,obj.N);
      obj.comddot_inds = reshape(tmp_idx(6*obj.N+(1:3*obj.N)),3,obj.N);
      
      x_name = cell(6*obj.N,1);
      for i = 1:obj.N
        x_name{(i-1)*3+1} = sprintf('centroidal_momentum_x[%d]',i);
        x_name{(i-1)*3+2} = sprintf('centroidal_momentum_y[%d]',i);
        x_name{(i-1)*3+3} = sprintf('centroidal_momentum_z[%d]',i);
        x_name{3*obj.N+(i-1)*3+1} = sprintf('momentum_dot_x[%d]',i);
        x_name{3*obj.N+(i-1)*3+2} = sprintf('momentum_dot_y[%d]',i);
        x_name{3*obj.N+(i-1)*3+3} = sprintf('momentum_dot_z[%d]',i);
      end
      [obj,tmp_idx] = obj.addDecisionVariable(6*obj.N,x_name);
      obj.centroidal_momentum_inds = reshape(tmp_idx(1:3*obj.N),3,obj.N);
      obj.world_momentum_dot_inds = reshape(tmp_idx(3*obj.N+(1:3*obj.N)),3,obj.N);
      
      if(~isempty(obj.q_quat_inds))
        num_quat = size(obj.q_quat_inds,2);
        x_name = cell(num_quat*(obj.N-1));
        for i = 1:num_quat
          for j = 1:obj.N-1
            x_name{(i-1)*(obj.N-1)+j} = sprintf('%s_quat_correction_slack[%d]',obj.robot.getBody(obj.floating_body_idx(i)).linkname,j+1);
          end
        end
        [obj,tmp_idx] = obj.addDecisionVariable(num_quat*(obj.N-1),x_name);
        obj.quat_correction_slack_inds = reshape(tmp_idx,num_quat,obj.N-1);
      end
      
      obj = obj.parseContactWrenchStruct(contact_wrench_struct);
    end
    
    function obj = addDynamicConstraints(obj)
    end
    
    function obj = setCWSMarginCost(obj,cws_margin_cost)
      sizecheck(cws_margin_cost,[1,1])
      if(cws_margin_cost<0)
        error('cws_margin_cost should be non-negative');
      end
      obj.cws_margin_cost = cws_margin_cost;
    end
    
  end
  
  methods(Access = protected)
    function obj = addPostureInterpolation(obj)
      % Use the mid-point interpolation for joint q and v
      function [c,dc] = postureInterpolationFun(obj,kinsol_l,kinsol_r,v_l,v_r,dt,quat_correction_slack)
        [VqInv_l,dVqInv_l] = obj.robot.vToqdot(kinsol_l);
        [VqInv_r,dVqInv_r] = obj.robot.vToqdot(kinsol_r);
        qdot_l = VqInv_l*v_l;
        qdot_r = VqInv_r*v_r;
        dqdot_l = [VqInv_l matGradMult(dVqInv_l,v_l)];
        dqdot_r = [VqInv_r matGradMult(dVqInv_r,v_r)];
        c = kinsol_r.q-kinsol_l.q-0.5*(qdot_l+qdot_r)*dt;
        dc = [-eye(obj.nq)-0.5*dt*dqdot_l(:,1:nq) eye(obj.nq)-0.5*dt*dqdot_r(:,1:obj.nq) -0.5*dt*dqdot_l(:,obj.nq+(1:obj.nv)) -0.5*dt*dqdot_r(:,obj.nq+(1:obj.nv)) -0.5*(qdot_l+qdot_r)];
        if(~isempty(obj.q_quat_inds))
          num_quat = size(obj.q_quat_inds,2);
          quat_correction = reshape(reshape(kinsol_r.q(obj.q_quat_inds),4,[])*bsxfun(@times,quat_correction_slack',ones(4,1)),[],1);
          dquat_correction_dqr = sparse((1:4*num_quat)',(1:4*num_quat)',reshape(bsxfun(@times,ones(4,1),quat_correction_slack'),[],1),4*num_quat,4*num_quat);
          dquat_correction_dslack = sparse((1:4*num_quat)',reshape(bsxfun(@times,ones(4,1),1:num_quat),[],1),kinsol_r.q(obj.q_quat_inds),4*num_quat,num_quat);
          c(obj.q_quat_inds(:)) = c(obj.q_quat_inds(:))-quat_correction;
          dc(obj.q_quat_inds(:),obj.nq+(obj.q_quat_inds(:))) = -dquat_correction_dqr;
          dc(obj.q_quat_inds(:),obj.nq*2+obj.nv*2+q+(1:num_quat)) = -dquat_correction_dslack;
        end
        
      end
    end
  end
end