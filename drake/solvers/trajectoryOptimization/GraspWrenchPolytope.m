classdef GraspWrenchPolytope < RigidBodyContactWrench
  properties(SetAccess = protected)
    % The grasp wrench w satisfies Ain_wrench*w<=bin_wrench;
    % Aeq_wrench*w=beq_wrench and w in convexhull(wrench_vert)
    num_wrench_ineq
    num_wrench_eq
    Ain_wrench % A num_wrench_ineq x 6 matrix
    bin_wrench % A num_wrench_ineq x 1 vector
    Aeq_wrench % A num_wrench_eq x 6 matrix
    beq_wrench % A num_wrench_eq x 1 vector
    num_wrench_vert
    wrench_vert % A 6 x num_wrench_vert matrix
  end
  
  methods
    function obj = GraspWrenchPolytope(robot,body,grasp_pt,wrench_vert)
      % @param wrench_vert A 6 x num_wrench_vert matrix, the vertices of
      % the wrench polytope
      sizecheck(grasp_pt,[3,1]);
      obj = obj@RigidBodyContactWrench(robot,body,grasp_pt);
      m_num_wrench_vert = size(wrench_vert,2);
      sizecheck(wrench_vert,[6,m_num_wrench_vert]);
      [obj.Ain_wrench,obj.bin_wrench,obj.Aeq_wrench,obj.beq_wrench] = vert2lcon(wrench_vert');
      obj.num_wrench_ineq = size(obj.Ain_wrench,1);
      obj.num_wrench_eq = size(obj.Aeq_wrench,1);
      obj.wrench_vert = lcon2vert(obj.Ain_wrench,obj.bin_wrench,obj.Aeq_wrench,obj.beq_wrench)';
      obj.num_wrench_vert = size(obj.wrench_vert,2);
      obj.num_wrench_constraint = 0;
    end
    
    function A = force(obj)
      A = obj.robot.getMass()*9.81*speye(3,6);
    end
    
    function A = torque(obj)
      A = obj.robot.getMass()*9.81/100*[sparse(3,3) speye(3)];
    end
    
    function [pos,J] = contactPosition(obj,kinsol)
      [pos,J] = obj.robot.forwardKin(kinsol,obj.body,obj.body_pts,0);
    end
  end
  
  methods(Access = protected)
    function lincon = generateWrenchLincon(obj)
      A_force = obj.force();
      A_torque = obj.torque();
      lincon = LinearConstraint([-inf(obj.num_wrench_ineq,1);obj.beq_wrench],[obj.bin_wrench;obj.beq_wrench],[obj.Ain_wrench;obj.Aeq_wrench]*[A_force;A_torque]);
      lincon = lincon.setName(repmat({sprintf('%s_grasp_wrench_polytope_constraint',obj.body_name)},obj.num_wrench_ineq+obj.num_wrench_eq,1));
    end
  end
end