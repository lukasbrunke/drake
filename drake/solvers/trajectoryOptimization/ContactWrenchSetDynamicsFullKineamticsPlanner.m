classdef ContactWrenchSetDynamicsFullKineamticsPlanner < RigidBodyKinematicsPlanner
  properties(SetAccess = protected)
    cws_margin_ind % A scalar.
    com_inds % A 3 x obj.N matrix
    comdot_inds % A 3 x obj.N matrix
    comddot_inds % A 3 x obj.N matrix
    centroidal_momentum_inds % A 6 x obj.N matrix
    world_momentum_dot_inds % A 3 x obj.N matrix
    
    cws_margin_cost % A positive scalar. The cost is -cws_margin_cost*cws_margin
    
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
      
      obj = obj.addState();
      
      obj = obj.addPostureInterpolation();
      
      obj = obj.addCentroidalConstraint();
    end
    
    function obj = addDynamicConstraints(obj)
    end
    
    function obj = setCWSMarginCost(obj,cws_margin_cost)
      sizecheck(cws_margin_cost,[1,1]);
      if(cws_margin_cost<0)
        error('cws_margin_cost should be non-negative');
      end
      obj.cws_margin_cost = cws_margin_cost;
    end
    
  end
  
  methods(Access = protected)
    function obj = addState(obj)
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
        x_name{(i-1)*6+1} = sprintf('centroidal_agl_momentum_x[%d]',i);
        x_name{(i-1)*6+2} = sprintf('centroidal_agl_momentum_y[%d]',i);
        x_name{(i-1)*6+3} = sprintf('centroidal_agl_momentum_z[%d]',i);
        x_name{(i-1)*6+4} = sprintf('centroidal_lin_momentum_x[%d]',i);
        x_name{(i-1)*6+5} = sprintf('centroidal_lin_momentum_y[%d]',i);
        x_name{(i-1)*6+6} = sprintf('centroidal_lin_momentum_z[%d]',i);
      end
      [obj,tmp_idx] = obj.addDecisionVariable(6*obj.N,x_name);
      obj.centroidal_momentum_inds = reshape(tmp_idx,6,obj.N);
      
      x_name = cell(6*obj.N,1);
      for i = 1:obj.N
        x_name{(i-1)*6+1} = sprintf('lin_momentum_dot_x[%d]',i);
        x_name{(i-1)*6+2} = sprintf('lin_momentum_dot_y[%d]',i);
        x_name{(i-1)*6+3} = sprintf('lin_momentum_dot_z[%d]',i);
        x_name{(i-1)*6+4} = sprintf('agl_momentum_dot_x[%d]',i);
        x_name{(i-1)*6+5} = sprintf('agl_momentum_dot_y[%d]',i);
        x_name{(i-1)*6+6} = sprintf('agl_momentum_dot_z[%d]',i);
      end
      [obj,tmp_idx] = obj.addDecisionVariable(6*obj.N,x_name);
      obj.world_momentum_dot_inds = reshape(tmp_idx,6,obj.N);
      
      if(~isempty(obj.q_quat_inds))
        num_quat = size(obj.q_quat_inds,2);
        x_name = cell(num_quat*(obj.N-1),1);
        for i = 1:num_quat
          for j = 1:obj.N-1
            x_name{(i-1)*(obj.N-1)+j} = sprintf('%s_quat_correction_slack[%d]',obj.robot.getBody(obj.floating_body_idx(i)).linkname,j+1);
          end
        end
        [obj,tmp_idx] = obj.addDecisionVariable(num_quat*(obj.N-1),x_name);
        obj.quat_correction_slack_inds = reshape(tmp_idx,num_quat,obj.N-1);
      end
    end
    
    
    function obj = addPostureInterpolation(obj)
      % Use the mid-point interpolation for joint q and v
      m_num_quat = size(obj.q_quat_inds,2);
      cnstr = FunctionHandleConstraint(zeros(obj.nq,1),zeros(obj.nq,1),2*obj.nq+2*obj.nv+1+m_num_quat,@(~,~,v_l,v_r,dt,quat_correction_slack,kinsol_l,kinsol_r) postureInterpolationFun(obj,kinsol_l,kinsol_r,v_l,v_r,dt,quat_correction_slack));
      cnstr_name = cell(obj.nq,1);
      for i = 1:obj.nq
        cnstr_name{i} = sprintf('q%d_interpolation',i);
      end
      cnstr = cnstr.setName(cnstr_name);
      
      if(~isempty(obj.q_quat_inds))
        sparsity_pattern = [ones(obj.nq,2*obj.nq+2*obj.nv+1) sparse(obj.q_quat_inds(:),reshape(bsxfun(@times,ones(4,1),1:m_num_quat),[],1),ones(4*m_num_quat,1),obj.nq,m_num_quat)];
      else
        sparsity_pattern = [eye(obj.nq) eye(obj.nq) eye(obj.nq) eye(obj.nq) ones(obj.nq,1)];
      end
      [iCfun,jCvar] = find(sparsity_pattern);
      cnstr = cnstr.setSparseStructure(iCfun,jCvar);
      
      for i = 1:obj.N-1
        if(~isempty(obj.q_quat_inds))
          quat_slack_correction_idx = obj.quat_correction_slack_inds(:,i);
        else
          quat_slack_correction_idx = [];
        end
        obj = obj.addConstraint(cnstr,[{obj.q_inds(:,i)};{obj.q_inds(:,i+1)};{obj.v_inds(:,i)};{obj.v_inds(:,i+1)};{obj.h_inds(i)};{quat_slack_correction_idx}],[obj.kinsol_dataind(i);obj.kinsol_dataind(i+1)]);
      end
    end
    
    function [c,dc] = postureInterpolationFun(obj,kinsol_l,kinsol_r,v_l,v_r,dt,quat_correction_slack)
      [VqInv_l,dVqInv_l] = obj.robot.vToqdot(kinsol_l);
      [VqInv_r,dVqInv_r] = obj.robot.vToqdot(kinsol_r);
      qdot_l = VqInv_l*v_l;
      qdot_r = VqInv_r*v_r;
      dqdot_l = [matGradMult(dVqInv_l,v_l) VqInv_l ];
      dqdot_r = [matGradMult(dVqInv_r,v_r) VqInv_r ];
      c = kinsol_r.q-kinsol_l.q-0.5*(qdot_l+qdot_r)*dt;
      dc = [-eye(obj.nq)-0.5*dt*dqdot_l(:,1:obj.nq) eye(obj.nq)-0.5*dt*dqdot_r(:,1:obj.nq) -0.5*dt*dqdot_l(:,obj.nq+(1:obj.nv)) -0.5*dt*dqdot_r(:,obj.nq+(1:obj.nv)) -0.5*(qdot_l+qdot_r)];
      if(~isempty(obj.q_quat_inds))
        num_quat = size(obj.q_quat_inds,2);
        quat_correction = reshape(reshape(kinsol_r.q(obj.q_quat_inds),4,[]).*bsxfun(@times,quat_correction_slack',ones(4,1)),[],1);
        dquat_correction_dqr = sparse((1:4*num_quat)',(1:4*num_quat)',reshape(bsxfun(@times,ones(4,1),quat_correction_slack'),[],1),4*num_quat,4*num_quat);
        dquat_correction_dslack = sparse((1:4*num_quat)',reshape(bsxfun(@times,ones(4,1),1:num_quat),[],1),kinsol_r.q(obj.q_quat_inds),4*num_quat,num_quat);
        c(obj.q_quat_inds(:)) = c(obj.q_quat_inds(:))-quat_correction;
        dc(obj.q_quat_inds(:),obj.nq+(obj.q_quat_inds(:))) = dc(obj.q_quat_inds(:),obj.nq+(obj.q_quat_inds(:)))-dquat_correction_dqr;
        dc(obj.q_quat_inds(:),obj.nq*2+obj.nv*2+1+(1:num_quat)) = -dquat_correction_dslack;
      end
    end
    
    function obj = addCentroidalConstraint(obj)
      % com = robot.getCOM()
      % centroidal_momentum = robot.centroidalMomentumMatrix*v
      cnstr = FunctionHandleConstraint(zeros(9,1),zeros(9,1),obj.nq+obj.nv+9,@(~,v,com,centroidal_momentum,kinsol) centroidalConstraintFun(obj,kinsol,v,com,centroidal_momentum));
      name = [repmat({'com = com(q)'},3,1);repmat({'centroidal_momentum=A(q)*v'},6,1)];
      cnstr = cnstr.setName(name);
      sparse_pattern = [ones(3,obj.nq) zeros(3,obj.nv) eye(3) zeros(3,6);ones(6,obj.nq+obj.nv) zeros(6,3) eye(6)];
      [iCfun,jCvar] = find(sparse_pattern);
      cnstr = cnstr.setSparseStructure(iCfun,jCvar);
      for i = 1:obj.N
        obj = obj.addConstraint(cnstr,[{obj.q_inds(:,i)};{obj.v_inds(:,i)};{obj.com_inds(:,i)};{obj.centroidal_momentum_inds(:,i)}],obj.kinsol_dataind(i));
      end
    end
    
    function [c,dc] = centroidalConstraintFun(obj,kinsol,v,com,centroidal_momentum)
      c = zeros(9,1);
      dc = zeros(9,obj.nq+obj.nv+9);
      [com_kinsol,dcom_kinsol] = obj.robot.getCOM(kinsol);
      c(1:3) = com-com_kinsol;
      dc(1:3,1:obj.nq) = -dcom_kinsol;
      dc(1:3,obj.nq+obj.nv+(1:3)) = eye(3);
      [A,dA] = obj.robot.centroidalMomentumMatrix(kinsol);
      h = A*v;
      c(4:9) = centroidal_momentum-h;
      dc(4:9,1:obj.nq) = -matGradMult(dA,v);
      dc(4:9,obj.nq+(1:obj.nv)) = -A;
      dc(4:9,obj.nq+obj.nv+3+(1:6)) = eye(6);
    end
    
    function obj = addMomentumInterpolationConstraint(obj)
      cnstr = FunctionHandleConstraint(zeros(6*(obj.N-1),1),zeros(6*(obj.N-1),1),16*obj.N-1,@(centroidal_momentum,momentum_dot,com,dt) obj.momentumInterpolation(obj,centroidal_momentum,momentum_dot,com,dt));
      name = cell(6*obj.(N-1),1);
      for i = 1:obj.N-1
        name(6*(i-1)+1+6) = repmat({sprintf('h[%d]-h[%d]=(hdot[%d]+hdot[%d])*dt[%d]',i+1,i,i,i+1,i)},6,1);
      end
      cnstr = cnstr.setNmae(name);
      iCfun = [(1:6*(obj.N-1))';(1:6*(obj.N-1))'];
      jCvar = [(1:6*(obj.N-1))';6+(1:6*(obj.N-1))'];
      iCfun = [iCfun;reshape(bsxfun(@plus,[2;3;4;1;3;5;1;2;6;1;2;3],6*(0:(obj.N-2))),[],1)];
      jCvar = [jCvar;reshape(bsxfun(@plus,6*obj.N+[1;1;1;2;2;2;3;3;3;4;5;6],6*(0:(obj.N-2))),[],1)];
      iCfun = [iCfun;reshape(bsxfun(@plus,[2;3;4;1;3;5;1;2;6;1;2;3],6*(0:(obj.N-2))),[],1)];
      jCvar = [jCvar;reshape(bsxfun(@plus,6*obj.N+6+[1;1;1;2;2;2;3;3;3;4;5;6],6*(0:(obj.N-2))),[],1)];
      iCfun = [iCfun;reshape(bsxfun(@plus,[2;3;1;3;1;2],6*(0:(obj.N-2))),[],1)];
      jCvar = [jCvar;reshape(bsxfun(@plus,12*obj.N+[1;1;2;2;3;3],3*(0:(obj.N-2))),[],1)];
      iCfun = [iCfun;reshape(bsxfun(@plus,[2;3;1;3;1;2],6*(0:(obj.N-2))),[],1)];
      jCvar = [jCvar;reshape(bsxfun(@plus,12*obj.N+3+[1;1;2;2;3;3],3*(0:(obj.N-2))),[],1)];
      iCfun = [iCfun;(1:6*(obj.N-1))'];
      jCvar = [jCvar;reshape(bsxfun(@times,15*obj.N+(1:obj.N-1),ones(6,1)),[],1)];
      cnstr = cnstr.setSparseStructure(iCfun,jCvar);
      obj = obj.addConstraint(cnstr,[{obj.centroidal_momentum_inds(:)};{obj.world_momentum_dot_inds(:)};{obj.com_inds(:)};{obj.h_inds(:)}]);
    end
    
    function [c,dc] = momentumInterpolationFun(obj,centroidal_momentum,momentum_dot,com,dt)
      % Use mid-point interpolation for momentum
      % first compute the centroidal momentum dot
      centroidal_momentum = reshape(centroidal_momentum,6,obj.N);
      momentum_dot = reshape(momentum_dot,6,obj.N);
      com = reshape(com,3,obj.N);
      dt = reshape(dt,1,obj.N-1);
      hdot = [momentum_dot(4:6,:);momentum_dot(1:3,:)];
      hdot(1:3,:) = hdot(1:3,:)+cross(hdot(4:6,:),com);
      c = reshape(diff(centroidal_momentum,[],2)-0.5*(hdot(:,1:end-1)+hdot(:,2:end)).*bsxfun(@times,dt,ones(6,1)),[],1);
      dc_dcentroidal_momentum = -speye(6*(obj.N-1),6*obj.N)+[sparse(6*(obj.N-1),6) speye(6*(obj.N-1))];
      com_cross = [0 0 1;0 -1 0;0 0 -1;1 0 0;0 1 0;-1 0 0]*com;
      dc_dmomentum_dot = -0.5*sparse((1:6*(obj.N-1))',reshape(bsxfun(@plus,[4;5;6;1;2;3],6*(0:(obj.N-2))),[],1),reshape(bsxfun(@times,dt,ones(6,1)),[],1),6*(obj.N-1),6*obj.N)...
        -0.5*sparse((1:6*(obj.N-1))',reshape(bsxfun(@plus,[4;5;6;1;2;3],6*(1:(obj.N-1))),[],1),reshape(bsxfun(@times,dt,ones(6,1)),[],1),6*(obj.N-1),6*obj.N)...
        -0.5*sparse(reshape(bsxfun(@plus,[2;3;1;3;1;2],6*(0:(obj.N-2))),[],1),reshape(bsxfun(@plus,[1;1;2;2;3;3;],6*(0:(obj.N-2))),[],1),reshape(-com_cross(:,1:end-1).*bsxfun(@times,dt,ones(6,1)),[],1),6*(obj.N-1),6*obj.N)...
        -0.5*sparse(reshape(bsxfun(@plus,[2;3;1;3;1;2],6*(0:(obj.N-2))),[],1),reshape(bsxfun(@plus,6+[1;1;2;2;3;3;],6*(0:(obj.N-2))),[],1),reshape(-com_cross(:,2:end).*bsxfun(@times,dt,ones(6,1)),[],1),6*(obj.N-1),6*obj.N);
      f_cross = [0 0 1;0 -1 0;0 0 -1;1 0 0;0 1 0;-1 0 0]*momentum_dot(1:3,:);
      dc_dcom = -0.5*sparse(reshape(bsxfun(@plus,[2;3;1;3;1;2],6*(0:(obj.N-2))),[],1),reshape(bsxfun(@plus,[1;1;2;2;3;3;],3*(0:(obj.N-2))),[],1),reshape(f_cross(:,1:end-1).*bsxfun(@times,ones(6,1),dt),[],1),6*(obj.N-1),3*obj.N)...
        -0.5*sparse(reshape(bsxfun(@plus,[2;3;1;3;1;2],6*(0:(obj.N-2))),[],1),reshape(bsxfun(@plus,3+[1;1;2;2;3;3;],3*(0:(obj.N-2))),[],1),reshape(f_cross(:,2:end).*bsxfun(@times,ones(6,1),dt),[],1),6*(obj.N-1),3*obj.N);
      dc_ddt = sparse((1:6*(obj.N-1))',reshape(bsxfun(@times,ones(6,1),(1:obj.N-1)),[],1),-0.5*(hdot(:,1:end-1)+hdot(:,2:end)),6*(obj.N-1),obj.N-1);
      dc = [dc_dcentroidal_momentum dc_dmomentum_dot dc_dcom dc_ddt];
    end
  end
end